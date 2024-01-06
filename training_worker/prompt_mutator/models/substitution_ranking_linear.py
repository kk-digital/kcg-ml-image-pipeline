from datetime import datetime
from io import BytesIO
import os
import sys
import tempfile
from matplotlib import pyplot as plt
import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader

base_directory = "./"
sys.path.insert(0, base_directory)
from utility.minio import cmd

class DatasetLoader(Dataset):
    def __init__(self, features, labels):
        """
        Initialize the dataset with features and labels.
        :param features: A NumPy array of the input features.
        :param labels: A NumPy array of the corresponding labels.
        """
        # Convert the data to torch.FloatTensor as it is the standard data type for floats in PyTorch
        self.features = torch.FloatTensor(np.array(features))
        self.labels = torch.FloatTensor(np.array(labels))

        print(f"features shape {self.features.shape}")
        print(f"targets shape {self.labels.shape}")

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Retrieve the features and label of the sample at the given index.
        :param idx: The index of the sample.
        """
        sample_features = self.features[idx]
        sample_label = self.labels[idx]
        return sample_features, sample_label


class LinearSubstitutionModel(nn.Module):
    def __init__(self, minio_client, input_size, hidden_sizes=[1024, 512], output_size=1, output_type="sigma_score", prompt_type="positive",
                 ranking_model="elm", operation="substitution", dataset="environmental", learning_rate=0.001, 
                 validation_split=0.2):
        
        super(LinearSubstitutionModel, self).__init__()
        # set device
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self._device = torch.device(device)

        # Define the multi-layered model architecture
        layers = [nn.Linear(input_size, hidden_sizes[0])]
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        # Adding the final layer (without an activation function for linear output)
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers).to(self._device)
        self.input_size= input_size
        self.minio_client= minio_client
        self.output_type= output_type
        self.prompt_type= prompt_type
        self.ranking_model=ranking_model
        self.operation=operation
        self.dataset=dataset
        self.date = datetime.now().strftime("%Y_%m_%d")
        self.local_path, self.minio_path=self.get_model_path()

        # Training parameters
        self.learning_rate = learning_rate
        self.validation_split = validation_split

    def get_model_path(self):
        local_path=f"output/{self.output_type}_prompt_mutator.pth"
        minio_path=f"{self.dataset}/models/prompt-generator/{self.operation}/{self.prompt_type}_prompts_only/{self.date}_linear_{self.output_type}_model.pth"

        return local_path, minio_path

    def train(self, inputs, outputs, num_epochs=100, batch_size=256):
        # load the dataset
        dataset= DatasetLoader(features=inputs, labels=outputs)
        # Split dataset into training and validation
        val_size = int(len(dataset) * self.validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create data loaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.L1Loss()  # Define the loss function
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)  # Define the optimizer

        # save loss for each epoch and features
        train_loss=[]
        val_loss=[]

        start = time.time()
        # Training and Validation Loop
        for epoch in range(num_epochs):
            self.model.eval()
            total_val_loss = 0
            total_val_samples = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self._device)
                    targets = targets.to(self._device)

                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)

                    total_val_loss += loss.item() * inputs.size(0)
                    total_val_samples += inputs.size(0)
                    
            self.model.train()
            total_train_loss = 0
            total_train_samples = 0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * inputs.size(0)
                total_train_samples += inputs.size(0)

            avg_train_loss = total_train_loss / total_train_samples
            avg_val_loss = total_val_loss / total_val_samples
            train_loss.append(avg_train_loss)
            val_loss.append(avg_val_loss)

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')
        
        end = time.time()
        training_time= end - start

        start = time.time()
        # Inference and calculate residuals on the training and validation set
        val_preds = self.inference(val_dataset, batch_size)
        train_preds = self.inference(train_dataset, batch_size)
        
        end = time.time()
        inference_speed=(train_size + val_size)/(end - start)
        print(f'Time taken for inference of {(train_size + val_size)} data points is: {end - start:.2f} seconds')

        # Extract the true values from the datasets
        y_train = torch.cat([y.unsqueeze(0) for _, y in train_dataset]).to(self._device)
        y_val = torch.cat([y.unsqueeze(0) for _, y in val_dataset]).to(self._device)

        # Calculate residuals
        val_residuals = y_val - val_preds
        train_residuals = y_train - train_preds

        val_preds= val_preds.cpu().numpy()
        y_val= y_val.cpu().numpy()
        val_residuals= val_residuals.cpu().numpy()
        train_residuals= train_residuals.cpu().numpy()
        
        self.save_graph_report(train_loss, val_loss, 
                               val_residuals, train_residuals, 
                               val_preds, y_val,
                               train_size, val_size)
        
        self.save_model_report(num_training=train_size,
                              num_validation=val_size,
                              training_time=training_time, 
                              train_loss=train_loss, 
                              val_loss=val_loss, 
                              inference_speed= inference_speed,
                              learning_rate=self.learning_rate)
        
        return val_loss[-1]
        
    def save_model_report(self,num_training,
                              num_validation,
                              training_time, 
                              train_loss, 
                              val_loss, 
                              inference_speed,
                              learning_rate):
        if self.operation=="substitution":
            input_type="[prompt_embeding[768], substituted_embeding[768], substitute_embedding[768], position_encoding[1], score_encoding[1]]"
        elif self.operation=="addition":
            input_type="[prompt_embeding[768], added_embedding[768], position_encoding[1]]"
        elif self.operation=="permutation":
            input_type="[prompt_embeding[768], first_embedding[768], second_embedding[768], first_position[1], second_position[1], score_encoding[1]]"

        report_text = (
            "================ Model Report ==================\n"
            f"Number of training datapoints: {num_training} \n"
            f"Number of validation datapoints: {num_validation} \n"
            f"Total training Time: {training_time:.2f} seconds\n"
            "Loss Function: L1 \n"
            f"Learning Rate: {learning_rate} \n"
            f"Training Loss: {train_loss[-1]} \n"
            f"Validation Loss: {val_loss[-1]} \n"
            f"Inference Speed: {inference_speed:.2f} predictions per second\n\n"
            "================ Input and output ==================\n"
            f"Input: {input_type} \n"
            f"Input Size: {self.input_size} \n" 
            f"Output: {self.output_type} \n\n"
        )

        # Define the local file path for the report
        local_report_path = 'output/model_report.txt'

        # Save the report to a local file
        with open(local_report_path, 'w') as report_file:
            report_file.write(report_text)

        # Read the contents of the local file
        with open(local_report_path, 'rb') as file:
            content = file.read()

        # Upload the local file to MinIO
        buffer = BytesIO(content)
        buffer.seek(0)

        cmd.upload_data(self.minio_client, 'datasets', self.minio_path.replace('.pth', '.txt'), buffer)

        # Remove the temporary file
        os.remove(local_report_path)

    def save_graph_report(self, train_mae_per_round, val_mae_per_round, 
                          val_residuals, train_residuals, 
                          predicted_values, actual_values,
                          training_size, validation_size):
        fig, axs = plt.subplots(3, 2, figsize=(12, 10))
        
        #info text about the model
        plt.figtext(0.02, 0.7, "Date = {}\n"
                            "Dataset = {}\n"
                            "Model = {}\n"
                            "Model type = {}\n"
                            "Input type = {}\n"
                            "Input shape = {}\n"
                            "Output type= {}\n\n"
                            ""
                            "Training size = {}\n"
                            "Validation size = {}\n"
                            "Training loss = {:.4f}\n"
                            "Validation loss = {:.4f}\n".format(self.date,
                                                            self.dataset,
                                                            f'Prompt {self.operation}',
                                                            'Linear',
                                                            f'{self.prompt_type}_clip_text_embedding',
                                                            f'{self.input_size}',
                                                            self.output_type,
                                                            training_size,
                                                            validation_size,
                                                            train_mae_per_round[-1],
                                                            val_mae_per_round[-1],
                                                            ))

        # Plot validation and training Rmse vs. Rounds
        axs[0][0].plot(range(1, len(train_mae_per_round) + 1), train_mae_per_round,'b', label='Training loss')
        axs[0][0].plot(range(1, len(val_mae_per_round) + 1), val_mae_per_round,'r', label='Validation loss')
        axs[0][0].set_title('MAE per Round')
        axs[0][0].set_ylabel('Loss')
        axs[0][0].set_xlabel('Rounds')
        axs[0][0].legend(['Training loss', 'Validation loss'])
        
        # Scatter Plot of actual values vs predicted values
        axs[0][1].scatter(predicted_values, actual_values, color='green', alpha=0.5)
        axs[0][1].set_title('Predicted values vs actual values')
        axs[0][1].set_ylabel('True')
        axs[0][1].set_xlabel('Predicted')

        # plot histogram of training residuals
        axs[1][0].hist(train_residuals, bins=30, color='blue', alpha=0.7)
        axs[1][0].set_xlabel('Residuals')
        axs[1][0].set_ylabel('Frequency')
        axs[1][0].set_title('Training Residual Histogram')

        # plot histogram of validation residuals
        axs[1][1].hist(val_residuals, bins=30, color='blue', alpha=0.7)
        axs[1][1].set_xlabel('Residuals')
        axs[1][1].set_ylabel('Frequency')
        axs[1][1].set_title('Validation Residual Histogram')
        
        # plot histogram of predicted values
        axs[2][0].hist(predicted_values, bins=30, color='blue', alpha=0.7)
        axs[2][0].set_xlabel('Predicted Values')
        axs[2][0].set_ylabel('Frequency')
        axs[2][0].set_title('Validation Predicted Values Histogram')
        
        # plot histogram of true values
        axs[2][1].hist(actual_values, bins=30, color='blue', alpha=0.7)
        axs[2][1].set_xlabel('Actual values')
        axs[2][1].set_ylabel('Frequency')
        axs[2][1].set_title('Validation True Values Histogram')

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0.7, wspace=0.3, left=0.3)

        plt.savefig(self.local_path.replace('.pth', '.png'))

        # Save the figure to a file
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # upload the graph report
        cmd.upload_data(self.minio_client, 'datasets', self.minio_path.replace('.pth', '.png'), buf)  

        # Clear the current figure
        plt.clf()

    def predict(self, features_array, batch_size=64):
        # Convert the features array into a PyTorch Tensor
        features_tensor = torch.Tensor(features_array).to(self._device)

        # Ensure the model is in evaluation mode
        self.model.eval()

        # List to hold all predictions
        predictions = []

        # Perform prediction in batches
        with torch.no_grad():
            for i in range(0, len(features_tensor), batch_size):
                batch = features_tensor[i:i + batch_size]  # Extract a batch
                outputs = self.model(batch)  # Get predictions for this batch
                predictions.append(outputs)

        # Concatenate all predictions and convert to a NumPy array
        predictions = torch.cat(predictions, dim=0).numpy()

        return predictions         

    def inference(self, dataset, batch_size=64):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()  # Set the model to evaluation mode
        predictions = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs= inputs.to(self._device)
                outputs = self.model(inputs)
                predictions.append(outputs)
        return torch.cat(predictions).squeeze()

    def load_model(self):
        minio_path=f"{self.dataset}/models/prompt-generator/{self.operation}/{self.prompt_type}_prompts_only/"
        file_name=f"_linear_{self.output_type}_model.pth"
        # get model file data from MinIO
        model_files=cmd.get_list_of_objects_with_prefix(self.minio_client, 'datasets', minio_path)
        most_recent_model = None

        for model_file in model_files:
            if model_file.endswith(file_name):
                most_recent_model = model_file

        if most_recent_model:
            model_file_data =cmd.get_file_from_minio(self.minio_client, 'datasets', most_recent_model)
        else:
            print("No .pth files found in the list.")
            return
        
        print(most_recent_model)

        # Create a temporary file and write the downloaded content into it
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            for data in model_file_data.stream(amt=8192):
                temp_file.write(data)

        # Load the model from the downloaded bytes
        self.model.load_state_dict(torch.load(temp_file.name))
        
        # Remove the temporary file
        os.remove(temp_file.name)

    def save_model(self):
         # Save the model locally
        torch.save(self.model.state_dict(), self.local_path)
        
        #Read the contents of the saved model file
        with open(self.local_path, "rb") as model_file:
            model_bytes = model_file.read()

        # Upload the model to MinIO
        cmd.upload_data(self.minio_client, 'datasets', self.minio_path, BytesIO(model_bytes))
        print(f'Model saved to {self.minio_path}')
