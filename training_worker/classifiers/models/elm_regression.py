import torch
import torch.nn as nn
from datetime import datetime
import os
import hashlib
import time


class ELMRegression():
    def __init__(self, device=None):
        self.model_type = 'elm-regression'
        self.date = datetime.now().strftime("%Y-%m-%d")
        self.tag_string = None
        self.model_file_path = None
        self.model_hash = None

        self._input_size = None
        self._hidden_layer_neuron_count = None
        self._output_size = None

        self._weight = None
        self._beta = None
        self._bias = None

        if not device and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        self._device = torch.device(device)

        self._activation = nn.Sigmoid()

        self.loss_func_name = "mse"
        self._loss = nn.MSELoss()

    def set_config(self, tag_string, input_size, hidden_layer_neuron_count, output_size=1, activation_func_name="sigmoid"):
        self.tag_string = tag_string
        self.model_file_path = ''
        self.model_hash = ''

        self.activation_func_name = activation_func_name
        self._activation = self.get_activation_func(activation_func_name)

        self._input_size = input_size
        self._hidden_layer_neuron_count = hidden_layer_neuron_count
        self._output_size = output_size

        self._weight = nn.init.uniform_(torch.empty(self._input_size, self._hidden_layer_neuron_count, device=self._device), a=-1.,
                                       b=1.)
        self._beta = nn.init.uniform_(torch.empty(self._hidden_layer_neuron_count, self._output_size, device=self._device), a=-1.,
                                      b=1.)
        self._bias = torch.zeros(self._hidden_layer_neuron_count, device=self._device)


    def train(self, training_feature_vector, training_targets):
        print("Training...")
        time_started = time.time()

        temp = training_feature_vector.mm(self._weight)
        H = self._activation(torch.add(temp, self._bias))

        H_pinv = torch.pinverse(H)
        self._beta = H_pinv.mm(training_targets)

        print("Finished training")
        print("Elapsed time: {}".format(time.time() - time_started))

    def classify(self, dataset_feature_vector):
        print("Classifying...")

        h = self._activation(torch.add(dataset_feature_vector.mm(self._weight), self._bias))
        out = h.mm(self._beta)

        return out

    def evaluate(self, dataset_feature_vector, dataset_targets):
        print("Evaluating...")

        predictions = self.classify(dataset_feature_vector)

        # calc loss
        loss = self._loss(predictions, dataset_targets)

        acc = torch.sum(torch.argmax(predictions, dim=1) == torch.argmax(dataset_targets, dim=1)).item() / len(dataset_targets)

        return predictions, loss, acc

    def get_model_hash(self):
        # TODO: make correction and verify better way to hash elm
        # model hash is sha256. hash of the pth file or the serialization
        model_str = str(self._weight) + str(self._beta) + str(self._bias)
        self.model_hash = hashlib.sha256(model_str.encode()).hexdigest()
        return self.model_hash

    def save_model(self, model_output_path):
        print("Saving model...")
        # Building path where model will be saved
        model_file_path = os.path.join(model_output_path, f'{self.tag_string}.pth')
        self.model_file_path = model_file_path

        # get model hash
        model_hash = self.get_model_hash()

        # make dir if it does not exist
        if not os.path.exists(model_output_path):
            os.makedirs(model_output_path)

        # Save checkpoint
        checkpoint = {
            'model-type': self.model_type,
            'tag-string': self.tag_string,
            'model-file-path': self.model_file_path,
            'model-hash': model_hash,
            'date': self.date,

            'input-size': self._input_size,
            'hidden-layer-neuron-count': self._hidden_layer_neuron_count,
            'output-size': self._output_size,
            'weight': self._weight,
            'beta': self._beta,
            'bias': self._bias,
            'activation-func': self.activation_func_name,
            'loss-func': self.loss_func_name,
        }

        torch.save(checkpoint, model_file_path)
        print(f"Model saved at '{model_file_path}'.")

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise Exception("Model path cannot be found")

        checkpoint = torch.load(model_path)

        self.model_type = checkpoint['model-type']
        self.tag_string = checkpoint['tag-string']
        self.model_file_path = checkpoint['model-file-path']
        self.model_hash = checkpoint['model-hash']
        self.date = checkpoint['date']

        self._input_size = checkpoint['input-size']
        self._hidden_layer_neuron_count = checkpoint['hidden-layer-neuron-count']
        self._output_size = checkpoint['output-size']

        self._weight = checkpoint['weight']
        self._beta = checkpoint['beta']
        self._bias = checkpoint['bias']

        self.loss_func_name = checkpoint['loss-func']
        self.activation_func_name = checkpoint['activation-func']
        self._activation = self.get_activation_func(self.activation_func_name)

        print("Model loaded: {}".format(model_path))

    def get_activation_func(self, activation_func_name):
        if activation_func_name == "sigmoid":
            return nn.Sigmoid()
        elif activation_func_name == "relu":
            return nn.ReLU()
        # TODO: add more activation func
        # elif activation_func_name == "sine":
        #     return nn.SiLU
