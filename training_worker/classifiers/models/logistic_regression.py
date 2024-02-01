import os
import sys
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

base_directory = os.getcwd()
sys.path.insert(0, os.path.join(base_directory, 'utils'))


class LogisticRegression:
    def __init__(self, inputs_shape):
        self.model = nn.Sequential(
            nn.Linear(inputs_shape, 1),
            nn.Sigmoid()
        )
        self.model_type = 'logistic-regression'
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
        model_file_path = os.path.join(model_output_path, 'logistic_regression.pth')
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

    def train(self, training_inputs, training_targets, validation_inputs, validation_targets, epochs=100, learning_rate=0.001, loss_function='mse', normalize_feature_vector = False):
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        loss_func = nn.MSELoss()
        self.loss_func_name = loss_function

        # Creating feature tensors
        # training_inputs = torch.tensor(training_inputs)
        # training_targets = torch.tensor(training_targets)
        # validation_inputs = torch.tensor(validation_inputs)
        # validation_targets = torch.tensor(validation_targets)

        if normalize_feature_vector:
            training_inputs = normalize_feature_vector(training_inputs)
            training_targets = normalize_feature_vector(training_targets)
            validation_inputs = normalize_feature_vector(validation_inputs)
            validation_targets = normalize_feature_vector(validation_targets)

        for epoch in range(epochs):
            optimizer.zero_grad()
            training_targets = self.model(training_inputs)
            loss = loss_func(training_targets, training_targets)
            loss.backward()
            optimizer.step()

            # Validation step
            with torch.no_grad():
                validation_targets = self.model(validation_inputs)
                validation_loss = loss_func(validation_targets, validation_targets)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4f} | Validation Loss: {validation_loss.item():.4f}")

        # Calculating and storing performance metrics
        training_outputs = self.model(training_inputs)
        validation_outputs = self.model(validation_inputs)

        # Storing loss
        self.training_loss = loss_func(training_outputs, training_targets)
        self.validation_loss = loss_func(validation_outputs, validation_targets)

        return training_outputs, validation_outputs

    def predict(self, inputs, normalize_feature_vector = False):
        with torch.no_grad():
            # inputs = torch.tensor(inputs)
            if normalize_feature_vector:
                inputs = normalize_feature_vector(inputs)

            outputs = self.model(inputs).squeeze()
            print(outputs)
            return outputs

def normalize_feature_vector(feature_vector):
    return  feature_vector / torch.linalg.norm(feature_vector, dim=1, keepdim=True)
