import torch
import torch.nn as nn
from datetime import datetime
import os
import sys
import hashlib
import time
from io import BytesIO
from safetensors.torch import save as safetensors_save
from safetensors.torch import load as safetensors_load

base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from utility.minio import cmd

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


    def train(self,
              training_feature_vector,
              training_targets,
              validation_feature_vector,
              validation_targets
              ):
        print("Training...")
        training_feature_vector = training_feature_vector.to(self._device)
        training_targets = training_targets.to(self._device)
        validation_feature_vector=validation_feature_vector.to(self._device)
        validation_targets=validation_targets.to(self._device)
        print("training feature vector shape=", training_feature_vector.shape)
        print("_weight shape=",self._weight.shape)
        time_started = time.time()

        temp = training_feature_vector.mm(self._weight)
        H = self._activation(torch.add(temp, self._bias))

        H_pinv = torch.pinverse(H)
        print("training targets shape=", training_targets.shape)
        print("h pinv shape=", H_pinv.shape)
        self._beta = H_pinv.mm(training_targets)

        print("Finished training")
        print("Elapsed time: {}".format(time.time() - time_started))

        # training loss and acc
        train_pred, training_loss, training_accuracy = self.evaluate(training_feature_vector, training_targets,
                                                                     threshold=0.5)

        # validation loss and acc
        validation_pred, validation_loss, validation_accuracy = self.evaluate(validation_feature_vector,
                                                                              validation_targets,
                                                                              threshold=0.5)

        return train_pred, training_loss, training_accuracy, validation_pred, validation_loss, validation_accuracy


    def classify(self, dataset_feature_vector):
        print("Classifying...")

        h = self._activation(torch.add(dataset_feature_vector.mm(self._weight), self._bias))
        out = h.mm(self._beta)

        return out

    def evaluate(self, validation_feature_vector, validation_targets, threshold=0.5):
        print("Evaluating...")

        predictions = self.classify(validation_feature_vector)

        # calc loss
        loss = self._loss(predictions, validation_targets)

        # if > threshold, will become true, then .float() converts to 1
        # true -> 1, false -> 0
        y_pred_converted = (predictions > threshold).float()
        validation_targets_converted = (validation_targets > threshold).float()

        acc = torch.sum(y_pred_converted == validation_targets_converted).item() / len(validation_targets)

        return predictions, loss, acc

    def get_model_hash(self):
        # TODO: make correction and verify better way to hash elm
        # model hash is sha256. hash of the pth file or the serialization
        model_str = str(self._weight) + str(self._beta) + str(self._bias)
        self.model_hash = hashlib.sha256(model_str.encode()).hexdigest()
        return self.model_hash

    def to_safetensors(self):
        # get model hash
        model_hash = self.get_model_hash()

        metadata = {
            'model-type': self.model_type,
            'tag-string': self.tag_string,
            'model-file-path': self.model_file_path,
            'model-hash': model_hash,
            'date': self.date,
            'input-size': "{}".format(self._input_size),
            'hidden-layer-neuron-count': "{}".format(self._hidden_layer_neuron_count),
            'output-size': "{}".format(self._output_size),
            'activation-func': self.activation_func_name,
            'loss-func': self.loss_func_name,
        }

        model_tensors = {
            'weight': self._weight,
            'beta': self._beta,
            'bias': self._bias,
        }

        return model_tensors, metadata

    def save_model(self, minio_client, datasets_bucket, model_output_path):
        # Hashing the model with its current configuration
        self.model_file_path = model_output_path

        # Preparing the model to be saved
        model, metadata = self.to_safetensors()

        # Saving the model to minio
        buffer = BytesIO()
        safetensors_buffer = safetensors_save(tensors=model,
                                              metadata=metadata)
        buffer.write(safetensors_buffer)
        buffer.seek(0)

        # upload the model
        cmd.upload_data(minio_client, datasets_bucket, model_output_path, buffer)

    def load_safetensors(self, model_buffer):
        data = model_buffer.read()
        safetensors_data = safetensors_load(data)

        self._weight = safetensors_data['weight']
        self._beta = safetensors_data['beta']
        self._bias = safetensors_data['bias']

        # load metadata
        n_header = data[:8]
        n = int.from_bytes(n_header, "little")
        metadata_bytes = data[8: 8 + n]
        header = json.loads(metadata_bytes)
        model = header.get("__metadata__", {})

        # Restoring model metadata
        self.model_type = model['model-type']
        self.tag_string = model['tag-string']
        self.model_file_path = model['model-file-path']
        self.model_hash = model['model-hash']
        self.date = model['date']

        self._input_size = model['input-size']
        self._hidden_layer_neuron_count = model['hidden-layer-neuron-count']
        self._output_size = model['output-size']

        self.loss_func_name = model['loss-func']
        self.activation_func_name = model['activation-func']
        self._activation = self.get_activation_func(self.activation_func_name)

    def get_activation_func(self, activation_func_name):
        if activation_func_name == "sigmoid":
            return nn.Sigmoid()
        elif activation_func_name == "relu":
            return nn.ReLU()
        # TODO: add more activation func
        # elif activation_func_name == "sine":
        #     return nn.SiLU
