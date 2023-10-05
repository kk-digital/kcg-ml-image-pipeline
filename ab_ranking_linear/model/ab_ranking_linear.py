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
base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from ab_ranking_linear.model.ab_ranking_data_loader import ABRankingDatasetLoader


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
        self.model = ABRankingLinearModel(inputs_shape)
        self.model_type = 'linear-regression'
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

    def save(self, model_output_path):
        # Building path where model will be saved
        model_file_path = os.path.join(model_output_path, 'ab_ranking_linear.pth')
        if not os.path.exists(model_output_path):
            os.mkdir(model_output_path)

        # Hashing the model with its current configuration
        self._hash_model()
        self.file_path = model_file_path
        # Preparing the model to be saved
        model = {}
        model['model_dict'] = self.model.state_dict()
        # Adding metadata
        model['model-type'] = self.model_type
        model['file-path'] = self.file_path
        model['model-hash'] = self.model_hash
        model['date'] = self.date
        # Saving the model to disk
        torch.save(model, model_file_path)

    def load(self, model_path):
        # Loading state dictionary
        model = torch.load(model_path)
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

    # training_image_x_features - the feature vectors of selected image
    # training_image_y_features - the feature vector of the other/not selected image
    # validation_image_x_features - the feature vectors of selected image
    # validation_image_y_features - the feature vector of the other/not selected image
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
        # validation_features_x, \
            # validation_features_y, \
            # validation_targets = dataset_loader.get_next_training_feature_vectors_and_target(200)

        # get number of batches to do per epoch
        # num_features = len(training_image_x_features)
        num_features = 1000
        training_num_batches = math.ceil(num_features / training_batch_size)

        for epoch in range(epochs):
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
                    batch_targets = dataset_loader.get_next_training_feature_vectors_and_target(num_data_to_get)

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

        # # Calculating and storing performance metrics
        # training_predicted_score_images_x = self.model.forward(training_image_x_features)
        # training_predicted_score_images_y = self.model.forward(training_image_y_features)
        # validation_predicted_score_images_x = self.model.forward(validation_image_x_features)
        # validation_predicted_score_images_y = self.model.forward(validation_image_y_features)

        # # Storing loss
        # self.training_loss, training_pred_probabilities = loss_func(training_predicted_score_images_x,
        #                                                             training_predicted_score_images_y,
        #                                                             training_target_probabilities)
        # self.validation_loss, validation_pred_probabilities = loss_func(validation_predicted_score_images_x,
        #                                                                 validation_predicted_score_images_y,
        #                                                                 validation_target_probabilities)

        return training_loss_per_epoch, validation_loss_per_epoch

    def ab_ranking_bradley_terry_loss(self, predicted_score_images_x, predicted_score_images_y, target_probabilities):
        epsilon = 0.000001

        # if score is negative N, make it 0
        predicted_score_images_x = torch.max(predicted_score_images_x, torch.tensor([0.]))
        predicted_score_images_y = torch.max(predicted_score_images_y, torch.tensor([0.]))

        # Calculate probability using Bradley Terry Formula: P(x>y) = score(x) / ( Score(x) + score(y))
        sum_predicted_score = predicted_score_images_x + predicted_score_images_y
        sum_predicted_score = torch.add(sum_predicted_score, epsilon)
        pred_probabilities = torch.div(predicted_score_images_x, sum_predicted_score)

        return self.model.bce_loss(pred_probabilities, target_probabilities), pred_probabilities

    def predict(self, inputs):
        with torch.no_grad():
            outputs = self.model.forward(inputs).squeeze()

            return outputs


