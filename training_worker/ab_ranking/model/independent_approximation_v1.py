import os
import sys
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from datetime import datetime
import math
import threading
from io import BytesIO, StringIO
from tqdm import tqdm
import csv
from datetime import datetime
from pytz import timezone
from safetensors.torch import save as safetensors_save
base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from data_loader.independent_approximation_dataset_loader import IndependentApproximationDatasetLoader
from utility.minio import cmd


def get_mean_absolute_deviation(energy_vector):
    mean = energy_vector.mean()
    mean_absolute = energy_vector - mean
    abs_mean_absolute = mean_absolute.abs()
    mean_absolute_deviation = abs_mean_absolute.mean()

    energy_vector_mad = energy_vector / mean_absolute_deviation
    return energy_vector_mad


class IndependentApproximationV1Model(nn.Module):
    def __init__(self,
                 inputs_shape,
                 token_length_vector):
        super(IndependentApproximationV1Model, self).__init__()
        self.inputs_shape = inputs_shape
        self.token_length_vector = token_length_vector

        range_low = -1.0
        range_high = 1.0
        initial_energy_vector = (range_high - range_low) * torch.rand((1, inputs_shape), dtype=torch.float32) + range_low
        self.energy_vector = nn.Parameter(data=initial_energy_vector, requires_grad=True)

        initial_prompt_phrase_average_weight = torch.zeros(1, dtype=torch.float32)
        self.prompt_phrase_average_weight = nn.Parameter(data=initial_prompt_phrase_average_weight, requires_grad=True)

        # self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCELoss()

    # for score
    def forward(self, input):
        assert input.shape == (1, self.inputs_shape), "{} != {}".format(input.shape, (1, self.inputs_shape))

        # phrase average weight param
        average_weight_param_product = torch.sum(input, dim=1) * self.prompt_phrase_average_weight

        # phrase score
        sigma_energy_vector = get_mean_absolute_deviation(self.energy_vector)
        energy_per_token = torch.mul(input, sigma_energy_vector)
        energy_per_phrase = torch.mul(self.token_length_vector, energy_per_token)
        prompt_energy = torch.sum(energy_per_phrase, dim=1)

        prompt_energy = prompt_energy + average_weight_param_product

        output = prompt_energy.unsqueeze(1)

        assert output.shape == (1, 1)
        return output


class ABRankingIndependentApproximationV1Model:
    def __init__(self, inputs_shape,
                 dataset_loader: IndependentApproximationDatasetLoader = None,
                 token_length_vector = None,
                 input_type="positive"):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self._device = torch.device(device)

        self.dataset_loader = dataset_loader
        self.input_type = input_type

        if token_length_vector != None:
            token_length_vector = token_length_vector.to(self._device)
        self.model = IndependentApproximationV1Model(inputs_shape,
                                                     token_length_vector).to(self._device)
        self.model_type = 'ab-ranking-independent-approximation-v1'
        self.loss_func_name = ''
        self.file_path = ''
        self.model_hash = ''
        self.date = datetime.now().strftime("%Y-%m-%d")

        self.training_loss = 0.0
        self.validation_loss = 0.0
        self.mean = 0.0
        self.standard_deviation = 0.0

        # training hyperparameters
        self.epochs = None
        self.learning_rate = None
        self.train_percent = None
        self.training_batch_size = None
        self.weight_decay = None
        self.pooling_strategy = None
        self.add_loss_penalty = None
        self.target_option = None
        self.duplicate_flip_option = None
        self.randomize_data_per_epoch = None

    def _hash_model(self):
        """
        Hashes the current state of the model, and stores the hash in the
        instance of the classifier.
        """
        model_str = str(self.model.state_dict())
        self.model_hash = hashlib.sha256(model_str.encode()).hexdigest()

    def add_hyperparameters_config(self,
                                   epochs,
                                   learning_rate,
                                   train_percent,
                                   training_batch_size,
                                   weight_decay,
                                   pooling_strategy,
                                   add_loss_penalty,
                                   target_option,
                                   duplicate_flip_option,
                                   randomize_data_per_epoch):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.train_percent = train_percent
        self.training_batch_size = training_batch_size
        self.weight_decay = weight_decay
        self.pooling_strategy = pooling_strategy
        self.add_loss_penalty = add_loss_penalty
        self.target_option = target_option
        self.duplicate_flip_option = duplicate_flip_option
        self.randomize_data_per_epoch = randomize_data_per_epoch

    def to_safetensors(self):
        metadata = {
            "model-type": self.model_type,
            "file-path": self.file_path,
            "model-hash": self.model_hash,
            "date": self.date,
            "training-loss": "{}".format(self.training_loss),
            "validation-loss": "{}".format(self.validation_loss),
            "mean": "{}".format(self.mean),
            "standard-deviation": "{}".format(self.standard_deviation),
            "epochs": "{}".format(self.epochs),
            "learning-rate": "{}".format(self.learning_rate),
            "train-percent": "{}".format(self.train_percent),
            "training-batch-size": "{}".format(self.training_batch_size),
            "weight-decay": "{}".format(self.weight_decay),
            "pooling-strategy": "{}".format(self.pooling_strategy),
            "add-loss-penalty": "{}".format(self.add_loss_penalty),
            "target-option": "{}".format(self.target_option),
            "duplicate-flip-option": "{}".format(self.duplicate_flip_option),
            "randomize-data-per-epoch": "{}".format(self.randomize_data_per_epoch),
        }

        model = self.model.state_dict()
        return model, metadata

    def save(self, minio_client, datasets_bucket, model_output_path):
        # Hashing the model with its current configuration
        self._hash_model()
        self.file_path = model_output_path

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

    def upload_phrases_score_csv(self):
        print("Saving phrases score csv...")

        csv_buffer = StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow((["index", "phrase", "occurrences", "token length", "prompt_phrase_average_weight", "energy_per_token", "sigma_energy_per_token", "energy_per_phrase"]))
        average_weight = None
        for name, param in self.model.named_parameters():
            if name == "prompt_phrase_average_weight":
                average_weight = param.cpu().detach().squeeze().numpy()

        for name, param in self.model.named_parameters():
            if name == "energy_vector":
                energy_vector = param.cpu().detach().squeeze().numpy()
                if self.input_type == "positive":
                    index_phrase_dict = self.dataset_loader.phrase_vector_loader.index_positive_phrases_dict
                    index_phrase_info = self.dataset_loader.phrase_vector_loader.index_positive_prompt_phrase_info
                else:
                    index_phrase_dict = self.dataset_loader.phrase_vector_loader.index_negative_phrases_dict
                    index_phrase_info = self.dataset_loader.phrase_vector_loader.index_negative_prompt_phrase_info

                has_negative_index = False
                if -1 in index_phrase_dict:
                    has_negative_index = True

                sigma_energy_vector = get_mean_absolute_deviation(param.cpu().detach().squeeze())
                sigma_energy_vector = sigma_energy_vector.numpy()

                for i in range(len(energy_vector)):
                    if has_negative_index:
                        i -= 1
                    index = i
                    phrase = index_phrase_dict[i]
                    phrase_info = index_phrase_info[index]
                    occurrences = phrase_info.occurrences
                    token_length = phrase_info.token_length
                    energy_per_token = "{:f}".format(energy_vector[i])
                    sigma_energy_per_token = "{:f}".format(sigma_energy_vector[i])
                    energy_per_phrase = "{:f}".format(sigma_energy_vector[i] * float(token_length))
                    writer.writerow([index, phrase, occurrences, token_length, average_weight, energy_per_token, sigma_energy_per_token, energy_per_phrase])

                bytes_buffer = BytesIO(bytes(csv_buffer.getvalue(), "utf-8"))
                # upload the csv
                date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%Y-%m-%d')
                filename = "{}-{}-phrases-score.csv".format(date_now, self.input_type)
                csv_path = os.path.join(self.dataset_loader.dataset_name, "output/phrases-score-csv", filename)
                cmd.upload_data(self.dataset_loader.minio_client, 'datasets', csv_path, bytes_buffer)

    def load_pth(self, model_buffer):
        # Loading state dictionary
        model = torch.load(model_buffer)
        # Restoring model metadata
        self.model_type = model['model-type']
        self.file_path = model['file-path']
        self.model_hash = model['model-hash']
        self.date = model['date']
        self.model.load_state_dict(model['model_dict'])

        # new added fields not in past models
        # so check first
        if "training-loss" in model:
            self.training_loss = model['training-loss']
            self.validation_loss = model['validation-loss']

        if "mean" in model:
            self.mean = model['mean']
            self.standard_deviation = model['standard-deviation']

        if "epochs" in model:
            self.epochs = model['epochs']
            self.learning_rate = model['learning-rate']
            self.train_percent = model['train-percent']
            self.training_batch_size = model['training-batch-size']
            self.weight_decay = model['weight-decay']
            self.pooling_strategy = model['pooling-strategy']
            self.add_loss_penalty = model['add-loss-penalty']
            self.target_option = model['target-option']
            self.duplicate_flip_option = model['duplicate-flip-option']
            self.randomize_data_per_epoch = model['randomize-data-per-epoch']

    def load_safetensors(self, model_buffer):
        data = model_buffer.read()

        # Loading state dictionary
        self.model.load_state_dict(safetensors_load(data))

        # load metadata
        n_header = data[:8]
        n = int.from_bytes(n_header, "little")
        metadata_bytes = data[8: 8 + n]
        header = json.loads(metadata_bytes)
        model = header.get("__metadata__", {})

        # Restoring model metadata
        self.model_type = model['model-type']
        self.file_path = model['file-path']
        self.model_hash = model['model-hash']
        self.date = model['date']

        # new added fields not in past models
        # so check first
        if "training-loss" in model:
            self.training_loss = model['training-loss']
            self.validation_loss = model['validation-loss']

        if "mean" in model:
            self.mean = model['mean']
            self.standard_deviation = model['standard-deviation']

        if "epochs" in model:
            self.epochs = model['epochs']
            self.learning_rate = model['learning-rate']
            self.train_percent = model['train-percent']
            self.training_batch_size = model['training-batch-size']
            self.weight_decay = model['weight-decay']
            self.pooling_strategy = model['pooling-strategy']
            self.add_loss_penalty = model['add-loss-penalty']
            self.target_option = model['target-option']
            self.duplicate_flip_option = model['duplicate-flip-option']
            self.randomize_data_per_epoch = model['randomize-data-per-epoch']

    def train(self,
              training_batch_size=1,
              epochs=10,
              learning_rate=0.05,
              weight_decay=0.00,
              add_loss_penalty=True,
              randomize_data_per_epoch=True,
              debug_asserts=True):
        training_loss_per_epoch = []
        validation_loss_per_epoch = []

        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.model_type = 'image-pair-independent-approximation-v1'
        self.loss_func_name = "L1"

        # get validation data
        validation_features_x, \
            validation_features_y, \
            validation_targets = self.dataset_loader.get_validation_feature_vectors_and_target(self._device)

        # get total number of training features
        num_features = self.dataset_loader.get_len_training_ab_data()

        # get number of batches to do per epoch
        training_num_batches = math.ceil(num_features / training_batch_size)
        for epoch in tqdm(range(epochs), desc="Training epoch"):
            training_loss_arr = []
            validation_loss_arr = []
            epoch_training_loss = None
            epoch_validation_loss = None

            # Only train after 0th epoch
            if epoch != 0:
                for i in range(training_num_batches):
                    num_data_to_get = training_batch_size
                    # last batch
                    if i == training_num_batches - 1:
                        num_data_to_get = num_features - (i * (training_batch_size))

                    batch_features_x_orig, \
                        batch_features_y_orig, \
                        batch_targets_orig = self.dataset_loader.get_next_training_feature_vectors_and_target(
                        num_data_to_get, self._device)

                    if debug_asserts:
                        assert batch_features_x_orig.shape == (training_batch_size, self.model.inputs_shape)
                        assert batch_features_y_orig.shape == (training_batch_size, self.model.inputs_shape)
                        assert batch_targets_orig.shape == (training_batch_size, 1)

                    batch_features_x = batch_features_x_orig.clone().to(self._device)
                    batch_features_y = batch_features_y_orig.clone().to(self._device)
                    batch_targets = batch_targets_orig.clone().requires_grad_(True).to(self._device)

                    with torch.no_grad():
                        predicted_score_images_y = self.model.forward(batch_features_y)

                    optimizer.zero_grad()
                    predicted_score_images_x = self.model.forward(batch_features_x)

                    predicted_score_images_y_copy = predicted_score_images_y.clone().requires_grad_(True).to(
                        self._device)
                    batch_pred_probabilities = forward_bradley_terry(predicted_score_images_x,
                                                                     predicted_score_images_y_copy)

                    if debug_asserts:
                        assert batch_pred_probabilities.shape == batch_targets.shape

                    loss = self.model.bce_loss(batch_pred_probabilities, batch_targets)

                    if add_loss_penalty:
                        # add loss penalty
                        # neg_score = torch.multiply(predicted_score_images_x, -1.0)
                        # negative_score_loss_penalty = torch.relu(neg_score)
                        # loss = torch.add(loss, negative_score_loss_penalty)

                        # loss penalty = (relu(-x-1) + relu(x-1))
                        # https://www.wolframalpha.com/input?i=graph+for+x%3D-5+to+x%3D5%2C++relu%28+-x+-+1.0%29+%2B+ReLu%28x+-+1.0%29
                        loss_penalty = torch.relu(-predicted_score_images_x - 1.0) + torch.relu(predicted_score_images_x - 1.0)
                        loss = torch.add(loss, loss_penalty)


                    loss.backward()
                    optimizer.step()

                    training_loss_arr.append(loss.detach().cpu())

                if debug_asserts:
                    for name, param in self.model.named_parameters():
                        if torch.isnan(param.grad).any():
                            print("nan gradient found")
                            raise SystemExit
                        # print("param={}, grad={}".format(name, param.grad))

                if randomize_data_per_epoch:
                    self.dataset_loader.shuffle_training_data()

                self.dataset_loader.current_training_data_index = 0

            # Calculate Validation Loss
            with torch.no_grad():
                for i in range(len(validation_features_x)):
                    validation_feature_x = validation_features_x[i]
                    validation_feature_x = validation_feature_x.unsqueeze(0)
                    validation_feature_y = validation_features_y[i]
                    validation_feature_y = validation_feature_y.unsqueeze(0)

                    validation_target = validation_targets[i]
                    validation_target = validation_target.unsqueeze(0)

                    validation_feature_x = validation_feature_x.to(self._device)
                    validation_feature_y = validation_feature_y.to(self._device)
                    validation_target = validation_target.to(self._device)

                    predicted_score_image_x = self.model.forward(validation_feature_x)
                    with torch.no_grad():
                        predicted_score_image_y = self.model.forward(validation_feature_y)

                    validation_pred_probabilities = forward_bradley_terry(predicted_score_image_x,
                                                                          predicted_score_image_y)

                    if debug_asserts:
                        assert validation_pred_probabilities.shape == validation_target.shape

                    validation_loss = self.model.bce_loss(validation_pred_probabilities, validation_target)

                    if add_loss_penalty:
                        # add loss penalty
                        # neg_score = torch.multiply(predicted_score_image_x, -1.0)
                        # negative_score_loss_penalty = torch.relu(neg_score)
                        # validation_loss = torch.add(validation_loss, negative_score_loss_penalty)

                        # loss penalty = (relu(-x-1) + relu(x-1))
                        # https://www.wolframalpha.com/input?i=graph+for+x%3D-5+to+x%3D5%2C++relu%28+-x+-+1.0%29+%2B+ReLu%28x+-+1.0%29
                        loss_penalty = torch.relu(-predicted_score_image_x - 1.0) + torch.relu(
                            predicted_score_image_x - 1.0)
                        validation_loss = torch.add(validation_loss, loss_penalty)

                    # validation_loss = torch.add(validation_loss, negative_score_loss_penalty)
                    validation_loss_arr.append(validation_loss.detach().cpu())

            # calculate epoch loss
            # epoch's training loss
            if len(training_loss_arr) != 0:
                training_loss_arr = torch.stack(training_loss_arr)
                epoch_training_loss = torch.mean(training_loss_arr)

            # epoch's validation loss
            validation_loss_arr = torch.stack(validation_loss_arr)
            epoch_validation_loss = torch.mean(validation_loss_arr)

            if epoch_training_loss is None:
                epoch_training_loss = epoch_validation_loss
            print(
                f"Epoch {epoch}/{epochs} | Loss: {epoch_training_loss:.4f} | Validation Loss: {epoch_validation_loss:.4f}")
            training_loss_per_epoch.append(epoch_training_loss)
            validation_loss_per_epoch.append(epoch_validation_loss)

            self.training_loss = epoch_training_loss.detach().cpu()
            self.validation_loss = epoch_validation_loss.detach().cpu()

        # Calculate model performance
        with torch.no_grad():
            training_predicted_score_images_x = []
            training_predicted_score_images_y = []
            training_predicted_probabilities = []
            training_target_probabilities = []

            # get performance metrics
            for i in range(training_num_batches):
                num_data_to_get = training_batch_size
                if i == training_num_batches - 1:
                    num_data_to_get = num_features - (i * (training_batch_size))

                batch_features_x, \
                    batch_features_y, \
                    batch_targets = self.dataset_loader.get_next_training_feature_vectors_and_target(num_data_to_get,
                                                                                                       self._device)

                batch_predicted_score_images_x = self.model.forward(batch_features_x)
                batch_predicted_score_images_y = self.model.forward(batch_features_y)

                batch_pred_probabilities = forward_bradley_terry(batch_predicted_score_images_x,
                                                                 batch_predicted_score_images_y)
                if debug_asserts:
                    # assert pred(x,y) = 1- pred(y,x)
                    batch_pred_probabilities_inverse = forward_bradley_terry(batch_predicted_score_images_y,
                                                                             batch_predicted_score_images_x)
                    tensor_ones = torch.tensor([1.0] * len(batch_pred_probabilities_inverse)).to(self._device)
                    assert torch.allclose(batch_pred_probabilities,
                                          torch.subtract(tensor_ones, batch_pred_probabilities_inverse), atol=1e-05)

                training_predicted_score_images_x.extend(batch_predicted_score_images_x)
                training_predicted_score_images_y.extend(batch_predicted_score_images_y)
                training_predicted_probabilities.extend(batch_pred_probabilities)
                training_target_probabilities.extend(batch_targets)

            # validation
            validation_predicted_score_images_x = []
            validation_predicted_score_images_y = []
            validation_predicted_probabilities = []
            for i in range(len(validation_features_x)):
                validation_feature_x = validation_features_x[i]
                validation_feature_x = validation_feature_x.unsqueeze(0)
                validation_feature_y = validation_features_y[i]
                validation_feature_y = validation_feature_y.unsqueeze(0)

                validation_feature_x = validation_feature_x.to(self._device)
                validation_feature_y = validation_feature_y.to(self._device)

                predicted_score_image_x = self.model.forward(validation_feature_x)
                predicted_score_image_y = self.model.forward(validation_feature_y)
                pred_probability = forward_bradley_terry(predicted_score_image_x, predicted_score_image_y)

                if debug_asserts:
                    # assert pred(x,y) = 1- pred(y,x)
                    pred_probability_inverse = forward_bradley_terry(predicted_score_image_y, predicted_score_image_x)
                    tensor_ones = torch.tensor([1.0] * len(pred_probability_inverse)).to(self._device)
                    assert torch.allclose(pred_probability, torch.subtract(tensor_ones, pred_probability_inverse),
                                          atol=1e-05)

                validation_predicted_score_images_x.append(predicted_score_image_x)
                validation_predicted_score_images_y.append(predicted_score_image_y)
                validation_predicted_probabilities.append(pred_probability)

        return training_predicted_score_images_x, \
            training_predicted_score_images_y, \
            training_predicted_probabilities, \
            training_target_probabilities, \
            validation_predicted_score_images_x, \
            validation_predicted_score_images_y, \
            validation_predicted_probabilities, \
            validation_targets, \
            training_loss_per_epoch, \
            validation_loss_per_epoch

    # Deprecate: This will be replaced by
    # predict_average_pooling
    def predict(self, positive_input, negative_input):
        # get rid of the 1 dimension at start
        positive_input = positive_input.squeeze()
        negative_input = negative_input.squeeze()

        # make it [2, 77, 768]
        inputs = torch.stack((positive_input, negative_input))

        # make it [1, 2, 77, 768]
        inputs = inputs.unsqueeze(0)

        # do average pooling
        inputs = torch.mean(inputs, dim=2)

        # then concatenate
        inputs = inputs.reshape(len(inputs), -1)

        with torch.no_grad():
            outputs = self.model.forward(inputs).squeeze()

            return outputs

    # predict pooled embedding
    def predict_pooled_embeddings(self, positive_input_pooled_embeddings, negative_input_pooled_embeddings):
        # make it [2, 77, 768]
        inputs = torch.stack((positive_input_pooled_embeddings, negative_input_pooled_embeddings))

        # make it [1, 2, 77, 768]
        inputs = inputs.unsqueeze(0)

        # then concatenate
        inputs = inputs.reshape(len(inputs), -1)

        with torch.no_grad():
            outputs = self.model.forward(inputs).squeeze()

            return outputs

    def predict_average_pooling(self,
                                positive_input,
                                negative_input,
                                positive_attention_mask,
                                negative_attention_mask):
        # get rid of the 1 dimension at start
        positive_input = positive_input.squeeze()
        negative_input = negative_input.squeeze()

        # do average pooling
        positive_input = tensor_attention_pooling(positive_input, positive_attention_mask)
        negative_input = tensor_attention_pooling(negative_input, negative_attention_mask)

        # make it [2, 1, 768]
        inputs = torch.stack((positive_input, negative_input))

        # make it [1, 2, 1, 768]
        inputs = inputs.unsqueeze(0)

        # then concatenate
        inputs = inputs.reshape(len(inputs), -1)

        with torch.no_grad():
            outputs = self.model.forward(inputs).squeeze()

            return outputs

    def predict_positive_or_negative_only(self, inputs):
        # do average pooling
        inputs = torch.mean(inputs, dim=2)

        # then concatenate
        inputs = inputs.reshape(len(inputs), -1)

        with torch.no_grad():
            outputs = self.model.forward(inputs).squeeze()

            return outputs

    # accepts only pooled embeddings
    def predict_positive_or_negative_only_pooled(self, inputs):
        # then concatenate
        inputs = inputs.reshape(len(inputs), -1)

        with torch.no_grad():
            outputs = self.model.forward(inputs).squeeze()

            return outputs

    def predict_clip(self, inputs):
        # concatenate
        inputs = inputs.reshape(len(inputs), -1)

        with torch.no_grad():
            outputs = self.model.forward(inputs).squeeze()

            return outputs


def forward_bradley_terry(predicted_score_images_x, predicted_score_images_y, use_sigmoid=True):
    if use_sigmoid:
        # scale the score
        # scaled_score_image_x = torch.multiply(1000.0, predicted_score_images_x)
        # scaled_score_image_y = torch.multiply(1000.0, predicted_score_images_y)

        # prob = sigmoid( (x-y) / 100 )
        diff_predicted_score = torch.sub(predicted_score_images_x, predicted_score_images_y)
        res_predicted_score = torch.div(diff_predicted_score, 0.1)
        pred_probabilities = torch.sigmoid(res_predicted_score)
    else:
        epsilon = 0.000001

        # if score is negative N, make it 0
        # predicted_score_images_x = torch.max(predicted_score_images_x, torch.tensor([0.], device=self._device))
        # predicted_score_images_y = torch.max(predicted_score_images_y, torch.tensor([0.], device=self._device))

        # Calculate probability using Bradley Terry Formula: P(x>y) = score(x) / ( Score(x) + score(y))
        sum_predicted_score = torch.add(predicted_score_images_x, predicted_score_images_y)
        sum_predicted_score = torch.add(sum_predicted_score, epsilon)
        pred_probabilities = torch.div(predicted_score_images_x, sum_predicted_score)

    return pred_probabilities
