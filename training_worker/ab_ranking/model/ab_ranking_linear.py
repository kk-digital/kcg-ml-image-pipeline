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
base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from training_worker.ab_ranking.model.ab_ranking_linear_data_loader import ABRankingDatasetLoader
from utility.minio import cmd


class ABRankingLinearModel(nn.Module):
    def __init__(self, inputs_shape):
        super(ABRankingLinearModel, self).__init__()
        self.linear = nn.Linear(inputs_shape, 1)
        self.identity = nn.Identity()
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = self.identity(x)
        # x = ((self.tanh(x) + 1.0)/2) * 100
        x = self.sigmoid(x) * 100

        return x


class ABRankingModel:
    def __init__(self, inputs_shape):
        print("inputs_shape=", inputs_shape)
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

    def train_batch(self,
                    optimizer,
                    loss_func,
                    model_copy,
                    training_image_x_features,
                    training_image_y_features,
                    training_target_probabilities):
        optimizer.zero_grad()
        predicted_score_images_x = self.model.forward(training_image_x_features)
        predicted_score_images_y = model_copy.forward(training_image_y_features)

        loss, _ = loss_func(predicted_score_images_x, predicted_score_images_y, training_target_probabilities)
        loss.backward()
        optimizer.step()

        return loss

    def train(self,
              dataset_loader: ABRankingDatasetLoader,
              training_batch_size=16,
              epochs=100,
              learning_rate=0.001):
        training_loss_per_epoch = []
        validation_loss_per_epoch = []

        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.model_type = 'image-pair-ranking-linear'
        self.loss_func_name = "ab_ranking_bradley_terry_loss"
        loss_func = self.ab_ranking_bradley_terry_loss

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
            # if buffer is empty, fill data
            fill_buffer_thread = threading.Thread(target=dataset_loader.fill_training_data_buffer)
            fill_buffer_thread.start()

            # get a copy of model
            model_copy = copy.deepcopy(self.model)
            # freeze weights of model_copy
            for param in model_copy.parameters():
                param.requires_grad = False

            for i in range(training_num_batches):
                num_data_to_get = training_batch_size
                if i == training_num_batches - 1:
                    num_data_to_get = num_features - (i * (training_batch_size))

                batch_features_x, \
                    batch_features_y,\
                    batch_targets = dataset_loader.get_next_training_feature_vectors_and_target(num_data_to_get, self._device)

                loss = self.train_batch(optimizer,
                                        loss_func,
                                        model_copy,
                                        batch_features_x,
                                        batch_features_y,
                                        batch_targets)

            # Validation step
            with torch.no_grad():
                predicted_score_images_x = self.model.forward(validation_features_x)
                predicted_score_images_y = self.model.forward(validation_features_y)
                validation_loss, _ = loss_func(predicted_score_images_x, predicted_score_images_y,
                                               validation_targets)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4f} | Validation Loss: {validation_loss.item():.4f}")
            training_loss_per_epoch.append(loss.item())
            validation_loss_per_epoch.append(validation_loss.item())

            # refill training ab data
            dataset_loader.fill_training_ab_data()

        with torch.no_grad():
            # fill data buffer
            # if buffer is empty, fill data
            fill_buffer_thread = threading.Thread(target=dataset_loader.fill_training_data_buffer)
            fill_buffer_thread.start()

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
                loss, batch_predicted_probabilities = loss_func(batch_predicted_score_images_x,
                                                             batch_predicted_score_images_y,
                                                             batch_targets)

                training_predicted_score_images_x.extend(batch_predicted_score_images_x)
                training_predicted_score_images_y.extend(batch_predicted_score_images_y)
                training_predicted_probabilities.extend(batch_predicted_probabilities)
                training_target_probabilities.extend(batch_targets)
            self.training_loss = loss

            validation_predicted_score_images_x = self.model.forward(validation_features_x)
            validation_predicted_score_images_y = self.model.forward(validation_features_y)
            self.validation_loss, validation_predicted_probabilities = loss_func(validation_predicted_score_images_x,
                                                              validation_predicted_score_images_y,
                                                              validation_targets)

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

    def ab_ranking_bradley_terry_loss(self, predicted_score_images_x, predicted_score_images_y, target_probabilities):
        epsilon = 0.000001

        # if score is negative N, make it 0
        predicted_score_images_x = torch.max(predicted_score_images_x, torch.tensor([0.]).to(self._device))
        predicted_score_images_y = torch.max(predicted_score_images_y, torch.tensor([0.]).to(self._device))

        # Calculate probability using Bradley Terry Formula: P(x>y) = score(x) / ( Score(x) + score(y))
        sum_predicted_score = predicted_score_images_x + predicted_score_images_y
        sum_predicted_score = torch.add(sum_predicted_score, epsilon)
        pred_probabilities = torch.div(predicted_score_images_x, sum_predicted_score)

        return self.model.bce_loss(pred_probabilities, target_probabilities), pred_probabilities

    def predict(self, inputs):
        with torch.no_grad():
            outputs = self.model.forward(inputs).squeeze()

            return outputs


