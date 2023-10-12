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
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        x1 = self.efficient_net(x)

        return x1


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
              learning_rate=0.001,
              weight_decay=0.01):
        training_loss_per_epoch = []
        validation_loss_per_epoch = []

        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.model_type = 'image-pair-ranking-efficient-net'
        self.loss_func_name = "mse"

        # get validation data
        validation_features_x, \
            validation_features_y, \
            validation_targets = dataset_loader.get_validation_feature_vectors_and_target()
        validation_features_x = validation_features_x.to(self._device)
        validation_features_y = validation_features_y.to(self._device)
        validation_targets = validation_targets.to(self._device)

        # get total number of training features
        num_features = dataset_loader.get_len_training_ab_data()

        # get number of batches to do per epoch
        training_num_batches = math.ceil(num_features / training_batch_size)
        loss = None
        with torch.autograd.detect_anomaly():
            for epoch in tqdm(range(epochs), desc="Training epoch"):
                # Only train after 0th epoch
                if epoch != 0:
                    # fill data buffer
                    dataset_loader.spawn_filling_workers()

                    for i in range(training_num_batches):
                        num_data_to_get = training_batch_size
                        # last batch
                        if i == training_num_batches - 1:
                            num_data_to_get = num_features - (i * (training_batch_size))

                        batch_features_x_orig, \
                            batch_features_y_orig,\
                            batch_targets_orig = dataset_loader.get_next_training_feature_vectors_and_target(num_data_to_get, self._device)

                        batch_features_x = batch_features_x_orig.clone().requires_grad_(True).to(self._device)
                        batch_features_y = batch_features_y_orig.clone().requires_grad_(True).to(self._device)
                        batch_targets = batch_targets_orig.clone().requires_grad_(True).to(self._device)

                        with torch.no_grad():
                            predicted_score_images_y = self.model.forward(batch_features_y)

                        optimizer.zero_grad()
                        predicted_score_images_x = self.model.forward(batch_features_x)

                        batch_pred_probabilities = self.forward_bradley_terry(predicted_score_images_x, predicted_score_images_y)

                        # assert
                        for pred_prob in batch_pred_probabilities:
                            assert pred_prob.item() >= 0.0
                            assert pred_prob.item() <= 1.0

                        assert batch_targets.shape == batch_pred_probabilities.shape
                        loss = self.model.mse_loss(batch_pred_probabilities, batch_targets)
                        loss.backward()
                        optimizer.step()

                    for name, param in self.model.named_parameters():
                        if torch.isnan(param.grad).any():
                            print("nan gradient found")
                            raise SystemExit
                        print("param={}, grad={}".format(name, param.grad))
                    # refill training ab data
                    dataset_loader.fill_training_ab_data()

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
                        predicted_score_image_y = self.model.forward(validation_feature_y)
                        pred_probability = self.forward_bradley_terry(predicted_score_image_x, predicted_score_image_y)

                        assert validation_target.shape == pred_probability.shape
                        validation_loss = self.model.mse_loss(pred_probability, validation_target)


                # if epoch % 10 == 0:
                if loss is None:
                    loss = validation_loss
                print(
                        f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4f} | Validation Loss: {validation_loss.item():.4f}")
                training_loss_per_epoch.append(loss.item())
                validation_loss_per_epoch.append(validation_loss.item())

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
                assert batch_targets.shape ==batch_pred_probabilities.shape
                loss = self.model.mse_loss(batch_pred_probabilities, batch_targets)

                training_predicted_score_images_x.extend(batch_predicted_score_images_x)
                training_predicted_score_images_y.extend(batch_predicted_score_images_y)
                training_predicted_probabilities.extend(batch_pred_probabilities)
                training_target_probabilities.extend(batch_targets)
            self.training_loss = loss

            # validation
            validation_predicted_score_images_x = []
            validation_predicted_score_images_y = []
            validation_predicted_probabilities = []
            for i in range(len(validation_features_x)):
                validation_feature_x = validation_features_x[i]
                validation_feature_x = validation_feature_x.unsqueeze(0)
                validation_feature_y = validation_features_y[i]
                validation_feature_y = validation_feature_y.unsqueeze(0)

                validation_target = validation_targets[i]
                validation_target = validation_target.unsqueeze(0)

                predicted_score_image_x = self.model.forward(validation_feature_x)
                predicted_score_image_y = self.model.forward(validation_feature_y)
                pred_probability = self.forward_bradley_terry(predicted_score_image_x, predicted_score_image_y)
                assert validation_target.shape == pred_probability.shape
                validation_loss = self.model.mse_loss(pred_probability, validation_target)
                validation_predicted_score_images_x.append(predicted_score_image_x)
                validation_predicted_score_images_y.append(predicted_score_image_y)
                validation_predicted_probabilities.append(pred_probability)

            self.validation_loss = validation_loss

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
        # sum_predicted_score = torch.add(predicted_score_images_x, predicted_score_images_y)
        # sum_predicted_score = torch.add(sum_predicted_score, epsilon)
        # pred_probabilities = torch.div(predicted_score_images_x, sum_predicted_score)

        # prob = sigmoid( (x-y) / 100 )
        diff_predicted_score = torch.sub(predicted_score_images_x, predicted_score_images_y)
        res_predicted_score = torch.div(diff_predicted_score, 100.0)
        pred_probabilities = torch.sigmoid(res_predicted_score)

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


