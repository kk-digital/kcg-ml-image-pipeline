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
from training_worker.sampling.scripts.directional_uniform_sampling_dataset import DirectionalUniformSphereGenerator
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


class DirectionalSamplingFCRegressionNetwork(nn.Module):
    def __init__(self, minio_client, input_size=2560, hidden_sizes=[512, 256], input_type="directional_uniform_sphere",
                 output_size=1, output_type="mean_sigma_score", dataset="environmental", dataloader=None):
        
        super(DirectionalSamplingFCRegressionNetwork, self).__init__()
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

        # Define layers for the residual model
        residual_model_layers = [
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_size)
        ]
        self.residual_model = nn.Sequential(*residual_model_layers).to(self._device)
        # model metadata
        self.metadata=None
        self.residual_model_metadata=None

        self.input_size= input_size
        self.minio_client= minio_client
        self.input_type= input_type
        self.output_type= output_type
        self.dataset=dataset
        self.min_scaling_factors=None
        self.max_scaling_factors=None

        self.date = datetime.now().strftime("%Y_%m_%d")
        self.local_path, self.minio_path, self.residual_minio_path=self.get_model_path()

        # sphere dataloader
        self.dataloader = dataloader

    def get_model_path(self):
        local_path=f"output/{self.output_type}_fc_{self.input_type}.pth"
        minio_path=f"{self.dataset}/models/sampling/{self.date}_{self.output_type}_fc_{self.input_type}.pth"
        residual_minio_path=f"{self.dataset}/models/sampling/{self.date}_{self.output_type}_residual_fc_{self.input_type}.pth"

        return local_path, minio_path, residual_minio_path

    def train(self, n_spheres, 
              target_avg_points, 
              learning_rate=0.001, 
              validation_split=0.2, 
              num_epochs=100, 
              batch_size=256,
              generate_every_epoch=False,
              train_residual_model=False):

        # load datapoints from minio
        if self.dataloader is None:
            self.dataloader= DirectionalUniformSphereGenerator(self.minio_client, self.dataset)
            self.dataloader.load_data()

        # Define the loss function and optimizer
        criterion = nn.L1Loss()  
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # save loss for each epoch and features
        train_loss=[]
        val_loss=[]

        best_val_loss = float('inf')  # Initialize best validation loss as infinity
        best_train_loss = float('inf')  # Initialize best training loss as infinity
        best_epoch= 0
        start = time.time()
        # Training and Validation Loop
        for epoch in range(num_epochs):

            if(epoch==0 or generate_every_epoch):
                # generate dataset once or every epoch
                val_loader, train_loader, \
                val_size, train_size, \
                val_dataset, train_dataset= self.get_validation_and_training_features(validation_split,
                                                                                    batch_size,
                                                                                    n_spheres,
                                                                                    target_avg_points)

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

            # Update best model if current epoch's validation loss is the best
            if val_loss[-1] < best_val_loss:
                best_val_loss = val_loss[-1]
                best_train_loss = train_loss[-1]
                best_model_state = self.model
                best_epoch= epoch + 1

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')
        
        # save the model at the best epoch
        self.model= best_model_state

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
        val_residual_tensors = y_val - val_preds
        train_residual_tensors = y_train - train_preds

        val_preds= val_preds.cpu().numpy()
        y_val= y_val.cpu().numpy()
        val_residuals= val_residual_tensors.cpu().numpy()
        train_residuals= train_residual_tensors.cpu().numpy()
        
        self.save_graph_report(train_loss, val_loss, 
                               best_epoch, generate_every_epoch,
                               best_train_loss, best_val_loss, 
                               val_residuals, train_residuals, 
                               val_preds, y_val,
                               train_size, val_size)
        
        self.save_model_report(num_training=train_size,
                              num_validation=val_size,
                              training_time=training_time, 
                              train_loss=best_train_loss, 
                              val_loss=best_val_loss,  
                              inference_speed= inference_speed,
                              learning_rate=learning_rate)

        if(train_residual_model):
            print("training a residual model----------------")
            self.train_residual_model(n_spheres, 
                                    target_avg_points,
                                    learning_rate,
                                    validation_split, 
                                    num_epochs, 
                                    batch_size,
                                    generate_every_epoch)
        
        self.save_metadata(inputs, target_avg_points, learning_rate, num_epochs, batch_size, residual_model=train_residual_model)
        
        return best_val_loss
    
    def train_residual_model(self,
                            n_spheres, 
                            target_avg_points,
                            learning_rate,
                            validation_split, 
                            num_epochs, 
                            batch_size,
                            generate_every_epoch):

        # Define the loss function and optimizer
        criterion = nn.L1Loss()  
        optimizer = optim.Adam(self.residual_model.parameters(), weight_decay=1e-5, lr=learning_rate)

        # save loss for each epoch and features
        train_loss=[]
        val_loss=[]

        best_val_loss = float('inf')  # Initialize best validation loss as infinity
        best_train_loss = float('inf')  # Initialize best training loss as infinity
        best_epoch= 0
        start = time.time()
        # Training and Validation Loop
        for epoch in range(num_epochs):

            if(epoch==0 and generate_every_epoch):
                # generate dataset once or every epoch
                val_loader, train_loader, \
                val_size, train_size, \
                val_dataset, train_dataset= self.get_validation_and_training_features(validation_split,
                                                                                    batch_size,
                                                                                    n_spheres,
                                                                                    target_avg_points,
                                                                                    residual_model=True)

            self.residual_model.eval()
            total_val_loss = 0
            total_val_samples = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs=inputs.to(self._device)
                    targets=targets.to(self._device)

                    outputs = self.residual_model(inputs)
                    loss = criterion(outputs.squeeze(1), targets)

                    total_val_loss += loss.item() * inputs.size(0)
                    total_val_samples += inputs.size(0)
                    
            self.residual_model.train()
            total_train_loss = 0
            total_train_samples = 0
            
            for inputs, targets in train_loader:
                inputs=inputs.to(self._device)
                targets=targets.to(self._device)

                optimizer.zero_grad()
                outputs = self.residual_model(inputs)
                loss = criterion(outputs.squeeze(1), targets)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * inputs.size(0)
                total_train_samples += inputs.size(0)

            avg_train_loss = total_train_loss / total_train_samples
            avg_val_loss = total_val_loss / total_val_samples
            train_loss.append(avg_train_loss)
            val_loss.append(avg_val_loss)

            # Update best model if current epoch's validation loss is the best
            if val_loss[-1] < best_val_loss:
                best_val_loss = val_loss[-1]
                best_train_loss = train_loss[-1]
                best_model_state = self.residual_model
                best_epoch= epoch + 1

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')
        
        # save the model at the best epoch
        self.residual_model= best_model_state

        end = time.time()
        training_time= end - start

        start = time.time()
        # Inference and calculate residuals on the training and validation set
        val_preds = self.inference(val_dataset, batch_size, residual_model=True)
        train_preds = self.inference(train_dataset, batch_size, residual_model=True)
        
        end = time.time()
        inference_speed=(train_size + val_size)/(end - start)
        print(f'Time taken for inference of {(train_size + val_size)} data points is: {end - start:.2f} seconds')

        # Extract the true values from the datasets
        y_train = torch.cat([y.unsqueeze(0) for _, y in train_dataset]).to(self._device)
        y_val = torch.cat([y.unsqueeze(0) for _, y in val_dataset]).to(self._device)

        # Calculate residuals
        val_residual_tensors = y_val - val_preds
        train_residual_tensors = y_train - train_preds

        val_preds= val_preds.cpu().numpy()
        y_val= y_val.cpu().numpy()
        val_residuals= val_residual_tensors.cpu().numpy()
        train_residuals= train_residual_tensors.cpu().numpy()
        
        self.save_graph_report(train_loss, val_loss, 
                               best_epoch, generate_every_epoch,
                               best_train_loss, best_val_loss, 
                               val_residuals, train_residuals, 
                               val_preds, y_val,
                               train_size, val_size,
                               residual_model=True)
        
        self.save_model_report(num_training=train_size,
                              num_validation=val_size,
                              training_time=training_time, 
                              train_loss=best_train_loss, 
                              val_loss=best_val_loss,  
                              inference_speed= inference_speed,
                              learning_rate=learning_rate,
                              residual_model=True)
    
    def get_validation_and_training_features(self, validation_split, batch_size, n_spheres, target_avg_points, residual_model=False):
        # load inputs and targets
        inputs, outputs = self.dataloader.generate_spheres(n_spheres, target_avg_points, self.output_type)

        if residual_model:
            # calculate residuals
            predicted_outputs= self.predict(inputs, batch_size= batch_size).squeeze().cpu().numpy()
            residuals= np.abs(np.array(outputs) - predicted_outputs)
            # calculate mean and std of residuals
            residual_mean= np.mean(residuals)
            residual_std= np.std(residuals)
            # convert residuals to sigma residuals
            residuals= (residuals - residual_mean) / residual_std
            outputs= residuals.tolist()

        # load the dataset
        dataset= DatasetLoader(features=inputs, labels=outputs)
        # Split dataset into training and validation
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create data loaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        return val_loader, train_loader, val_size, train_size, val_dataset, train_dataset
    
    def save_metadata(self, spheres, points_per_sphere, learning_rate, num_epochs, training_batch_size, residual_model):
        # get min and max of spheres
        scaling_factors=[sphere[self.input_size//2:] for sphere in spheres]

        self.min_scaling_factors = [min(feature_values) for feature_values in zip(*scaling_factors)]
        self.max_scaling_factors = [max(feature_values) for feature_values in zip(*scaling_factors)]

        # Metadata
        self.metadata = {
            'points_per_sphere': points_per_sphere,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'training_batch_size': training_batch_size,
            'model_state': self.model.state_dict(),
            'min_scaling_factors': self.min_scaling_factors,
            'max_scaling_factors': self.max_scaling_factors
        }

        if residual_model:
            # Metadata
            self.residual_model_metadata = {
                'points_per_sphere': points_per_sphere,
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'training_batch_size': training_batch_size,
                'model_state': self.residual_model.state_dict()
            }
        
    def save_model_report(self,num_training,
                              num_validation,
                              training_time, 
                              train_loss, 
                              val_loss, 
                              inference_speed,
                              learning_rate,
                              residual_model=False):
        input_type="[input_clip_vector[1280], scaling_factors[1280]]"
        output_type= f"{self.output_type}_residual" if residual_model else self.output_type

        report_text = (
            "================ Model Report ==================\n"
            f"Number of training datapoints: {num_training} \n"
            f"Number of validation datapoints: {num_validation} \n"
            f"Total training Time: {training_time:.2f} seconds\n"
            "Loss Function: L1 \n"
            f"Learning Rate: {learning_rate} \n"
            f"Training Loss: {train_loss} \n"
            f"Validation Loss: {val_loss} \n"
            f"Inference Speed: {inference_speed:.2f} predictions per second\n\n"
            "================ Input and output ==================\n"
            f"Input: {input_type} \n"
            f"Input Size: {self.input_size} \n" 
            f"Output: {output_type} \n\n"
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

        minio_path= self.residual_minio_path if residual_model else self.minio_path 
        cmd.upload_data(self.minio_client, 'datasets', minio_path.replace('.pth', '.txt'), buffer)

        # Remove the temporary file
        os.remove(local_report_path)

    def save_graph_report(self, train_mae_per_round, val_mae_per_round,
                          saved_at_epoch, generate_every_epoch,
                          best_train_loss, best_val_loss,  
                          val_residuals, train_residuals, 
                          predicted_values, actual_values,
                          training_size, validation_size,
                          residual_model=False):
        fig, axs = plt.subplots(3, 2, figsize=(12, 10))
        
        output_type= f"{self.output_type}_residual" if residual_model else self.output_type
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
                            "Validation loss = {:.4f}\n"
                            "Saved at epoch = {}\n"
                            "Generation policy = {}\n".format(self.date,
                                                            self.dataset,
                                                            'Fc_Network',
                                                            self.input_type,
                                                            self.input_size,
                                                            output_type,
                                                            training_size,
                                                            validation_size,
                                                            best_train_loss,
                                                            best_val_loss,
                                                            saved_at_epoch,
                                                            "every epoch" if generate_every_epoch else "once"
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
        minio_path= self.residual_minio_path if residual_model else self.minio_path 
        cmd.upload_data(self.minio_client, 'datasets', minio_path.replace('.pth', '.png'), buf)  

        # Clear the current figure
        plt.clf()

    def predict(self, data, batch_size=64):
        if isinstance(data, np.ndarray) or isinstance(data, list):
            data= torch.FloatTensor(data)

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

    def inference(self, dataset, batch_size=64, residual_model=False):
        inference_model= self.residual_model if residual_model else self.model
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()  # Set the model to evaluation mode
        predictions = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs= inputs.to(self._device)
                outputs = inference_model(inputs)
                predictions.append(outputs)
        return torch.cat(predictions).squeeze()

    def load_model(self, load_residual_model=False):
        # get model file data from MinIO
        prefix= f"{self.dataset}/models/sampling/"
        model_suffix= f"_{self.output_type}_fc_{self.input_type}.pth"
        residual_model_suffix= f"_{self.output_type}_residual_fc_{self.input_type}.pth"

        model_files=cmd.get_list_of_objects_with_prefix(self.minio_client, 'datasets', prefix)
        most_recent_model = None
        most_recent_residual_model = None

        for model_file in model_files:
            if model_file.endswith(model_suffix):
                most_recent_model = model_file
            if model_file.endswith(residual_model_suffix) and load_residual_model:
                most_recent_residual_model = model_file

        if most_recent_model:
            model_file_data = cmd.get_file_from_minio(self.minio_client, 'datasets', most_recent_model)
            # Assuming get_file_from_minio returns a response object that can be read as bytes
            model_bytes = model_file_data.read()  # Read the entire content as bytes
            # Use BytesIO as a file-like object for torch.load
            model_buffer = BytesIO(model_bytes)
            self.metadata = torch.load(model_buffer)
            self.min_scaling_factors= self.metadata["min_scaling_factors"]
            self.max_scaling_factors= self.metadata["max_scaling_factors"]
            self.model.load_state_dict(self.metadata["model_state"])

            print(most_recent_model)
        else:
            print("No files found for this model.")
            return None

        # Similar approach for the residual model
        if most_recent_residual_model and load_residual_model:
            residual_model_file_data = cmd.get_file_from_minio(self.minio_client, 'datasets', most_recent_residual_model)
            residual_model_bytes = residual_model_file_data.read()  # Read as bytes
            residual_model_buffer = BytesIO(residual_model_bytes)
            self.residual_model_metadata = torch.load(residual_model_buffer)
            self.residual_model.load_state_dict(self.residual_model_metadata["model_state"])
        else:
            print("No files found for the residual model.")
            return None

    def save_model(self):
        if self.metadata is None:
            raise Exception("You have to train the model first before saving.")
        
        # Create an in-memory buffer
        model_buffer = BytesIO()
        # Save the model directly into the buffer
        torch.save(self.metadata, model_buffer)
        # Reset the buffer's position to the beginning
        model_buffer.seek(0)
        # Upload the buffer to MinIO
        cmd.upload_data(self.minio_client, 'datasets', self.minio_path, model_buffer)
        print(f'Model saved to {self.minio_path}')

        if self.residual_model_metadata is not None:
            # Create an in-memory buffer
            residual_model_buffer = BytesIO()
            # Save the model directly into the buffer
            torch.save(self.residual_model_metadata, residual_model_buffer)
            residual_model_buffer.seek(0)
            cmd.upload_data(self.minio_client, 'datasets', self.residual_minio_path, residual_model_buffer)
            print(f'Residual Model saved to {self.residual_minio_path}')