from io import BytesIO
import os
import sys
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import math
from tqdm import tqdm
from datetime import datetime
import threading

base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from training_worker.ab_ranking.model.ab_ranking_efficient_net_data_loader import ABRankingDatasetLoader
from utility.minio import cmd
from training_worker.ab_ranking.model.efficient_net_model import EfficientNet as efficientnet_pytorch


class EfficientNetModel(nn.Module):
    def __init__(self, efficient_net_version="b0", in_channels=1, num_classes=1):
        super(EfficientNetModel, self).__init__()
        self.efficient_net = efficientnet_pytorch(efficient_net_version, in_channels=in_channels, num_classes=num_classes)
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.efficient_net(x)
        x2 = self.sigmoid(x1)

        return x2


class ABRankingEfficientNetModel:
    def __init__(self, efficient_net_version="b0", in_channels=1, num_classes=1):
        print("efficient_net_version =", efficient_net_version)
        print("in_channels =", in_channels)
        print("num_classes =", num_classes)

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self._device = torch.device(device)

        self.model = EfficientNetModel(efficient_net_version, in_channels, num_classes).to(self._device)
        self.model_type = 'ab-ranking-efficient-net'
        self.loss_func_name = ''
        self.file_path = ''
        self.model_hash = ''
        self.date = datetime.now().strftime("%Y-%m-%d")

        self.training_loss = 0.0
        self.validation_loss = 0.0

    def _hash_model(self):
        """
        Hashes the current state of the model, and stores the hash in the
        instance of the classifier.
        """
        model_str = str(self.model.state_dict())
        self.model_hash = hashlib.sha256(model_str.encode()).hexdigest()

    def save(self, minio_client, datasets_bucket, model_output_path):
        # Hashing the model with its current configuration
        self._hash_model()
        self.file_path = model_output_path
        # Preparing the model to be saved
        model = {}
        model['model_dict'] = self.model.state_dict()
        # Adding metadata
        model['model-type'] = self.model_type
        model['file-path'] = self.file_path
        model['model-hash'] = self.model_hash
        model['date'] = self.date

        # Saving the model to minio
        buffer = BytesIO()
        torch.save(model, buffer)
        buffer.seek(0)
        # upload the model
        cmd.upload_data(minio_client, datasets_bucket, model_output_path, buffer)

    def load(self, model_buffer):
        # Loading state dictionary
        model = torch.load(model_buffer)
        # Restoring model metadata
        self.model_type = model['model-type']
        self.file_path = model['file-path']
        self.model_hash = model['model-hash']
        self.date = model['date']
        self.model.load_state_dict(model['model_dict'])


    def train(self,
              dataset_loader: ABRankingDatasetLoader,
              training_batch_size=4,
              epochs=100,
              learning_rate=0.001):
        training_loss_per_epoch = []
        validation_loss_per_epoch = []

        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.model_type = 'image-pair-ranking-efficient-net'
        self.loss_func_name = "bce"

        # get validation data
        validation_features_x, \
            validation_features_y, \
            validation_targets = dataset_loader.get_validation_feature_vectors_and_target()
        validation_features_x = validation_features_x.to(self._device)
        validation_features_y = validation_features_y.to(self._device)
        validation_targets = validation_targets.to(self._device)

        # num features * 2 bc we duplicate each ab data
        # (x, y, 1.0)
        # (y, x, 0.0)
        num_features = dataset_loader.get_len_training_ab_data() * 2

        # get number of batches to do per epoch
        training_num_batches = math.ceil(num_features / training_batch_size)

        for epoch in tqdm(range(epochs), desc="Training epoch"):
            # fill data buffer
            dataset_loader.spawn_filling_workers()

            for i in range(training_num_batches):
                num_data_to_get = training_batch_size
                # last batch
                if i == training_num_batches - 1:
                    num_data_to_get = num_features - (i * (training_batch_size))

                batch_features_x, \
                    batch_features_y,\
                    batch_targets = dataset_loader.get_next_training_feature_vectors_and_target(num_data_to_get, self._device)

                optimizer.zero_grad()
                predicted_score_images_x = self.model.forward(batch_features_x)
                with torch.no_grad():
                    predicted_score_images_y = self.model.forward(batch_features_y)

                predicted_score_images_y_copy = predicted_score_images_y.clone().detach().requires_grad_(True)
                batch_pred_probabilities = self.forward_bradley_terry(predicted_score_images_x, predicted_score_images_y_copy)
                loss = self.model.bce_loss(batch_pred_probabilities, batch_targets)
                loss.backward()
                optimizer.step()


            # Calculate Validation Loss
            with torch.no_grad():
                predicted_score_images_x = self.model.forward(validation_features_x)
                predicted_score_images_y = self.model.forward(validation_features_y)
                batch_pred_probabilities = self.forward_bradley_terry(predicted_score_images_x, predicted_score_images_y)
                validation_loss = self.model.bce_loss(batch_pred_probabilities, validation_targets)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4f} | Validation Loss: {validation_loss.item():.4f}")
            training_loss_per_epoch.append(loss.item())
            validation_loss_per_epoch.append(validation_loss.item())

            # refill training ab data
            dataset_loader.fill_training_ab_data()

        with torch.no_grad():
            # fill data buffer
            dataset_loader.spawn_filling_workers()

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
                    batch_features_y,\
                    batch_targets = dataset_loader.get_next_training_feature_vectors_and_target(num_data_to_get, self._device)

                batch_predicted_score_images_x = self.model.forward(batch_features_x)
                batch_predicted_score_images_y = self.model.forward(batch_features_y)
                batch_pred_probabilities = self.forward_bradley_terry(batch_predicted_score_images_x,
                                                             batch_predicted_score_images_y)
                loss = self.model.bce_loss(batch_pred_probabilities, batch_targets)

                training_predicted_score_images_x.extend(batch_predicted_score_images_x)
                training_predicted_score_images_y.extend(batch_predicted_score_images_y)
                training_predicted_probabilities.extend(batch_pred_probabilities)
                training_target_probabilities.extend(batch_targets)
            self.training_loss = loss

            validation_predicted_score_images_x = self.model.forward(validation_features_x)
            validation_predicted_score_images_y = self.model.forward(validation_features_y)
            validation_predicted_probabilities = self.forward_bradley_terry(validation_predicted_score_images_x,
                                                              validation_predicted_score_images_y)
            self.validation_loss = self.model.bce_loss(validation_predicted_probabilities, validation_targets)

        return training_predicted_score_images_x,\
            training_predicted_score_images_y, \
            training_predicted_probabilities,\
            training_target_probabilities,\
            validation_predicted_score_images_x, \
            validation_predicted_score_images_y,\
            validation_predicted_probabilities, \
            validation_targets,\
            training_loss_per_epoch, \
            validation_loss_per_epoch

    def forward_bradley_terry(self, predicted_score_images_x, predicted_score_images_y):
        epsilon = 0.000001

        # if score is negative N, make it 0
        predicted_score_images_x = torch.max(predicted_score_images_x, torch.tensor([0.], device=self._device))
        predicted_score_images_y = torch.max(predicted_score_images_y, torch.tensor([0.], device=self._device))

        # Calculate probability using Bradley Terry Formula: P(x>y) = score(x) / ( Score(x) + score(y))
        sum_predicted_score = torch.add(predicted_score_images_x, predicted_score_images_y)
        sum_predicted_score = torch.add(sum_predicted_score, epsilon)
        pred_probabilities = torch.div(predicted_score_images_x, sum_predicted_score)

        return pred_probabilities

    def predict_positive_negative(self, positive_input, negative_input):
        # get rid of the 1 dimension at start
        positive_input = positive_input.squeeze()
        negative_input = negative_input.squeeze()

        # make it [2, 77, 768]
        inputs = torch.stack((positive_input, negative_input))

        # make it [1, 2, 77, 768]
        inputs = inputs.unsqueeze(0)

        with torch.no_grad():
            outputs = self.model.forward(inputs).squeeze()

            return outputs


