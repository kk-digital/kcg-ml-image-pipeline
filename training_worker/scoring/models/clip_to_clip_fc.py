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
import torch.nn.functional as F

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

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, outputs, targets):
        # Normalize vectors to have unit norm
        outputs = F.normalize(outputs, p=2, dim=1)
        targets = F.normalize(targets, p=2, dim=1)
        # Calculate cosine similarity and convert to loss
        cosine_loss = 1 - F.cosine_similarity(outputs, targets)
        return cosine_loss.mean()

class CliptoClipFCNetwork(nn.Module):
    def __init__(self, minio_client, input_size=1280, hidden_sizes=[2048, 1024, 512], input_type="input_clip" , output_size=1280, 
                 output_type="output_clip", dataset="environmental", loss_func="cosine"):
        
        super(CliptoClipFCNetwork, self).__init__()
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
        self.input_type= input_type
        self.output_type= output_type
        self.dataset=dataset
        self.loss_func= loss_func
        self.date = datetime.now().strftime("%Y_%m_%d")
        self.local_path, self.minio_path=self.get_model_path()

    def get_model_path(self):
        local_path=f"output/{self.output_type}_fc_{self.input_type}.pth"
        minio_path=f"{self.dataset}/models/latent-generator/{self.date}_{self.output_type}_fc_{self.input_type}.pth"

        return local_path, minio_path

    def train(self, inputs, outputs, learning_rate=0.001, validation_split=0.2, num_epochs=100, batch_size=256):
        # load the dataset
        dataset= DatasetLoader(features=inputs, labels=outputs)
        # Split dataset into training and validation
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create data loaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # Select the loss function based on the type specified
        if self.loss_func == 'cosine':
            criterion = CosineSimilarityLoss()
        else:
            criterion = nn.MSELoss()

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)  # Define the optimizer

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
                    inputs=inputs.to(self._device)
                    targets=targets.to(self._device)

                    outputs = self.model(inputs)
                    loss = criterion(outputs.squeeze(1), targets)

                    total_val_loss += loss.item() * inputs.size(0)
                    total_val_samples += inputs.size(0)
                    
            self.model.train()
            total_train_loss = 0
            total_train_samples = 0
            
            for inputs, targets in train_loader:
                inputs=inputs.to(self._device)
                targets=targets.to(self._device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs.squeeze(1), targets)
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
        val_cosine_similarities = self.get_cosine_similarities(val_dataset, batch_size).numpy()
        train_cosine_similarities = self.get_cosine_similarities(train_dataset, batch_size).numpy()
        
        end = time.time()
        inference_speed=(train_size + val_size)/(end - start)
        print(f'Time taken for inference of {(train_size + val_size)} data points is: {end - start:.2f} seconds')

        self.save_graph_report(train_loss, val_loss, 
                               train_cosine_similarities,
                               val_cosine_similarities,
                               train_size, val_size)
        
        self.save_model_report(num_training=train_size,
                              num_validation=val_size,
                              training_time=training_time, 
                              train_loss=train_loss, 
                              val_loss=val_loss, 
                              inference_speed= inference_speed,
                              learning_rate=learning_rate)
    
        return val_loss[-1]
        
    def save_model_report(self,num_training,
                              num_validation,
                              training_time, 
                              train_loss, 
                              val_loss, 
                              inference_speed,
                              learning_rate):

        if self.input_type=="input_clip":
            input_type="[input_clip_vector[1280]]"

        report_text = (
            "================ Model Report ==================\n"
            f"Number of training datapoints: {num_training} \n"
            f"Number of validation datapoints: {num_validation} \n"
            f"Total training Time: {training_time:.2f} seconds\n"
            f"Loss Function: {self.loss_func} \n"
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

    def save_graph_report(self, train_loss_per_round, val_loss_per_round, 
                          train_cosine_similarities,
                          val_cosine_similarities,
                          training_size, validation_size):
        fig, axs = plt.subplots(3, 1, figsize=(12, 10))
        
        #info text about the model
        plt.figtext(0.02, 0.7, "Date = {}\n"
                            "Dataset = {}\n"
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
                                                            'Fc_Network',
                                                            self.input_type,
                                                            self.input_size,
                                                            self.output_type,
                                                            training_size,
                                                            validation_size,
                                                            train_loss_per_round[-1],
                                                            val_loss_per_round[-1],
                                                            ))

        # Plot validation and training Rmse vs. Rounds
        axs[0][0].plot(range(1, len(train_loss_per_round) + 1), train_loss_per_round,'b', label='Training loss')
        axs[0][0].plot(range(1, len(val_loss_per_round) + 1), val_loss_per_round,'r', label='Validation loss')
        axs[0][0].set_title('Loss per Round')
        axs[0][0].set_ylabel('Loss')
        axs[0][0].set_xlabel('Rounds')
        axs[0][0].legend(['Training loss', 'Validation loss'])

        # plot histogram of training residuals
        axs[1][0].hist(train_cosine_similarities, bins=30, color='blue', alpha=0.7)
        axs[1][0].set_xlabel('Cosine Similarity')
        axs[1][0].set_ylabel('Frequency')
        axs[1][0].set_title('Training Cosine Similarity Histogram')
       
        # plot histogram of training residuals
        axs[2][0].hist(val_cosine_similarities, bins=30, color='blue', alpha=0.7)
        axs[2][0].set_xlabel('Cosine Similarity')
        axs[2][0].set_ylabel('Frequency')
        axs[2][0].set_title('Validation Cosine Similarity Histogram')

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

    def predict(self, data, batch_size=64):
        # Ensure the data tensor is on the correct device
        data = data.to(self._device)

        # Ensure the model is in evaluation mode
        self.model.eval()

        # List to hold all predictions
        predictions = []

        # Perform prediction in batches
        with torch.no_grad():
            for i in range(0, data.size(0), batch_size):
                batch = data[i:i+batch_size]  # Extract a batch
                outputs = self.model(batch)  # Get predictions for this batch
                predictions.append(outputs)

        # Concatenate all predictions
        predictions = torch.cat(predictions, dim=0)

        return predictions         

    def get_cosine_similarities(self, dataset, batch_size=64):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()  # Set the model to evaluation mode
        cosine_similarities = []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs= inputs.to(self._device)
                outputs = self.model(inputs).squeeze(1)
                # Normalize vectors to have unit norm
                outputs = F.normalize(outputs, p=2, dim=1)
                targets = F.normalize(targets, p=2, dim=1)
                # Calculate cosine similarity and convert to loss
                cosine_sim = F.cosine_similarity(outputs.cpu(), targets.cpu())
                cosine_similarities.append(cosine_sim) 
                
        return torch.cat(cosine_similarities).squeeze()

    def load_model(self):
        # get model file data from MinIO
        prefix= f"{self.dataset}/models/latent-generator/"
        suffix= f"_{self.output_type}_fc_{self.input_type}.pth"
        model_files=cmd.get_list_of_objects_with_prefix(self.minio_client, 'datasets', prefix)
        most_recent_model = None

        for model_file in model_files:
            if model_file.endswith(suffix):
                most_recent_model = model_file

        if most_recent_model:
            model_file_data =cmd.get_file_from_minio(self.minio_client, 'datasets', most_recent_model)
        else:
            print("No .pth files found in the list.")
            return None
        
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