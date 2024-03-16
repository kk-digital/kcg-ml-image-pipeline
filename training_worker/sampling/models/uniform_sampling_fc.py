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
from training_worker.sampling.scripts.uniform_sampling_dataset import UniformSphereGenerator
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


class SamplingFCNetwork(nn.Module):
    def __init__(self, minio_client, input_size=1281, hidden_sizes=[512, 256], input_type="input_clip" , output_size=12, 
                 output_type="score_distribution", dataset="environmental"):
        
        super(SamplingFCNetwork, self).__init__()
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
        
        # Adjusting the last layer to use LogSoftmax for KLDivLoss compatibility
        layers += [nn.Linear(hidden_sizes[-1], output_size), nn.LogSoftmax(dim=1)]

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers).to(self._device)
        self.input_size= input_size
        self.minio_client= minio_client
        self.input_type= input_type
        self.output_type= output_type
        self.dataset=dataset
        self.date = datetime.now().strftime("%Y_%m_%d")
        self.local_path, self.minio_path=self.get_model_path()

        # sphere dataloader
        self.dataloader= UniformSphereGenerator(minio_client, dataset)

    def get_model_path(self):
        local_path=f"output/{self.output_type}_fc_{self.input_type}.pth"
        minio_path=f"{self.dataset}/models/sampling/{self.date}_{self.output_type}_fc_{self.input_type}.pth"

        return local_path, minio_path

    def train(self, n_spheres, target_avg_points, learning_rate=0.001, validation_split=0.2, num_epochs=100, batch_size=256):
        # load the dataset
        inputs, outputs = self.dataloader.load_sphere_dataset(n_spheres,target_avg_points)

        dataset= DatasetLoader(features=inputs, labels=outputs)
        # Split dataset into training and validation
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create data loaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        criterion = nn.KLDivLoss(reduction='batchmean')  # Using KLDivLoss
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

                    print(inputs[0], inputs.shape)
                    print(targets[0], targets.shape)

                    outputs = self.model(inputs)
                    loss = criterion(outputs.squeeze(1), targets)

                    print(outputs[0], outputs.shape)
                    print(loss)

                    total_val_loss += loss.item() * inputs.size(0)
                    total_val_samples += inputs.size(0)
                    
            self.model.train()
            total_train_loss = 0
            total_train_samples = 0
            
            for inputs, targets in train_loader:
                inputs=inputs.to(self._device)
                targets=targets.to(self._device)

                print(inputs[0], inputs.shape)
                print(targets[0], targets.shape)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs.squeeze(1), targets)
                print(outputs[0], outputs.shape)
                print(loss)
                
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
        
        self.save_graph_report(train_loss, val_loss,
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
        if self.input_type=="output_clip":
            input_type="[output_image_clip_vector[1280]]"
        elif self.input_type=="input_clip":
            input_type="[input_clip_vector[1280]]"

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
                          training_size, validation_size):
        
        # Create a figure and a set of subplots
        fig, ax = plt.subplots()

        # Adjust the subplot parameters to give specified padding
        fig.subplots_adjust(left=0.3)  # Adjust as needed to fit your text

        # Info text about the model
        info_text = ("Date = {}\n"
                    "Dataset = {}\n"
                    "Model type = {}\n"
                    "Input type = {}\n"
                    "Input shape = {}\n"
                    "Output type= {}\n\n"
                    ""
                    "Training size = {}\n"
                    "Validation size = {}\n"
                    "Training loss = {:.4f}\n"
                    "Validation loss = {:.4f}\n").format(self.date,
                                                        self.dataset,
                                                        'Fc_Network',
                                                        self.input_type,
                                                        self.input_size,
                                                        self.output_type,
                                                        training_size,
                                                        validation_size,
                                                        train_mae_per_round[-1],
                                                        val_mae_per_round[-1])

        # Use figtext to place text to the left of the plot
        fig.text(0.02, 0.7, info_text)

        # Plot validation and training MAE vs. Rounds on the ax
        ax.plot(range(1, len(train_mae_per_round) + 1), train_mae_per_round, 'b', label='Training loss')
        ax.plot(range(1, len(val_mae_per_round) + 1), val_mae_per_round, 'r', label='Validation loss')
        ax.set_title('MAE per Round')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Rounds')
        ax.legend(['Training loss', 'Validation loss'])

        # Save the figure to a local file
        plt.savefig(self.local_path.replace('.pth', '.png'))

        # Save the figure to a buffer for uploading
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Upload the graph report
        cmd.upload_data(self.minio_client, 'datasets', self.minio_path.replace('.pth', '.png'), buf)  

        # Clear the figure to free up memory
        plt.close(fig)

    def predict(self, data, batch_size=64):
        # Convert the features array into a PyTorch Tensor
        features_tensor = torch.Tensor(np.array(data)).to(self._device)

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
        predictions = torch.cat(predictions, dim=0).cpu().numpy()

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
        # get model file data from MinIO
        prefix= f"{self.dataset}/models/sampling/"
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