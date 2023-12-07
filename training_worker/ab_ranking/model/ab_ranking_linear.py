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
from io import BytesIO
from tqdm import tqdm
from safetensors.torch import save as safetensors_save
from safetensors.torch import load as safetensors_load
from safetensors import safe_open
from safetensors import deserialize as safetensors_deserialize
import json
base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from data_loader.ab_ranking_dataset_loader import ABRankingDatasetLoader
from utility.minio import cmd
# from utility.clip.clip_text_embedder import tensor_attention_pooling

class ABRankingLinearModel(nn.Module):
    def __init__(self, inputs_shape):
        super(ABRankingLinearModel, self).__init__()
        self.inputs_shape = inputs_shape
        self.linear = nn.Linear(inputs_shape, 1)
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.tanh = nn.Tanh()

    # for score
    def forward(self, input):
        assert input.shape == (1, self.inputs_shape)

        output = self.linear(input)

        assert output.shape == (1,1)
        return output


class ABRankingModel:
    def __init__(self, inputs_shape):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self._device = torch.device(device)

        self.model = ABRankingLinearModel(inputs_shape).to(self._device)
        self.model_type = 'ab-ranking-linear'
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
            "weight-decay": "{}".format( self.weight_decay),
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
              dataset_loader: ABRankingDatasetLoader,
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
        self.model_type = 'image-pair-ranking-linear'
        self.loss_func_name = "L1"

        # get validation data
        validation_features_x, \
            validation_features_y, \
            validation_targets = dataset_loader.get_validation_feature_vectors_and_target_linear(self._device)

        # get total number of training features
        num_features = dataset_loader.get_len_training_ab_data()

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
                        batch_targets_orig = dataset_loader.get_next_training_feature_vectors_and_target_linear(
                        num_data_to_get, self._device)

                    if debug_asserts:
                        assert batch_features_x_orig.shape == (training_batch_size, self.model.inputs_shape)
                        assert batch_features_y_orig.shape == (training_batch_size, self.model.inputs_shape)
                        assert batch_targets_orig.shape == (training_batch_size, 1)

                    batch_features_x = batch_features_x_orig.clone().requires_grad_(True).to(self._device)
                    batch_features_y = batch_features_y_orig.clone().requires_grad_(True).to(self._device)
                    batch_targets = batch_targets_orig.clone().requires_grad_(True).to(self._device)

                    with torch.no_grad():
                        predicted_score_images_y = self.model.forward(batch_features_y)

                    optimizer.zero_grad()
                    predicted_score_images_x = self.model.forward(batch_features_x)

                    predicted_score_images_y_copy = predicted_score_images_y.clone().requires_grad_(True).to(self._device)
                    batch_pred_probabilities = forward_bradley_terry(predicted_score_images_x,
                                                                          predicted_score_images_y_copy)

                    if debug_asserts:
                        assert batch_pred_probabilities.shape == batch_targets.shape

                    loss = self.model.l1_loss(batch_pred_probabilities, batch_targets)

                    if add_loss_penalty:
                        # add loss penalty
                        neg_score = torch.multiply(predicted_score_images_x, -1.0)
                        negative_score_loss_penalty = torch.relu(neg_score)
                        loss = torch.add(loss, negative_score_loss_penalty)

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
                    dataset_loader.shuffle_training_data()

                dataset_loader.current_training_data_index = 0

            # Calculate Validation Loss
            with torch.no_grad():
                for i in range(len(validation_features_x)):
                    validation_feature_x = validation_features_x[i]
                    validation_feature_x = validation_feature_x.unsqueeze(0)
                    validation_feature_y = validation_features_y[i]
                    validation_feature_y = validation_feature_y.unsqueeze(0)

                    validation_target = validation_targets[i]
                    validation_target = validation_target.unsqueeze(0)

                    predicted_score_image_x = self.model.forward(validation_feature_x)
                    with torch.no_grad():
                        predicted_score_image_y = self.model.forward(validation_feature_y)

                    validation_pred_probabilities = forward_bradley_terry(predicted_score_image_x,
                                                                          predicted_score_image_y)

                    if debug_asserts:
                        assert validation_pred_probabilities.shape == validation_target.shape

                    validation_loss = self.model.l1_loss(validation_pred_probabilities, validation_target)

                    if add_loss_penalty:
                        # add loss penalty
                        neg_score = torch.multiply(predicted_score_image_x, -1.0)
                        negative_score_loss_penalty = torch.relu(neg_score)
                        validation_loss = torch.add(validation_loss, negative_score_loss_penalty)

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
                    batch_targets = dataset_loader.get_next_training_feature_vectors_and_target_linear(num_data_to_get,
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
                    assert torch.allclose(batch_pred_probabilities, torch.subtract(tensor_ones, batch_pred_probabilities_inverse), atol=1e-05)

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

                predicted_score_image_x = self.model.forward(validation_feature_x)
                predicted_score_image_y = self.model.forward(validation_feature_y)
                pred_probability = forward_bradley_terry(predicted_score_image_x, predicted_score_image_y)

                if debug_asserts:
                    # assert pred(x,y) = 1- pred(y,x)
                    pred_probability_inverse = forward_bradley_terry(predicted_score_image_y, predicted_score_image_x)
                    tensor_ones = torch.tensor([1.0] * len(pred_probability_inverse)).to(self._device)
                    assert torch.allclose(pred_probability, torch.subtract(tensor_ones, pred_probability_inverse), atol=1e-05)

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
        res_predicted_score = torch.div(diff_predicted_score, 50.0)
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
