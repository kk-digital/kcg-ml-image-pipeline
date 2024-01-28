import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchsummary import summary
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import threading
from io import BytesIO
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn import MSELoss, CrossEntropyLoss, L1Loss
from torch.autograd import profiler
from datetime import datetime


architecture = 'treeConnect'



########################################################## Global parameters ##########################################################


# # Initialize the model, loss function, and optimizer
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# # Hyperparameters
# batch_size = 64
# learning_rate = 0.001
# epochs = 5
# criterion = nn.CrossEntropyLoss()



class tree_connect_architecture(nn.Module):
    def __init__(self, inputs_shape, output_shape):
        super(tree_connect_architecture, self).__init__()
        # Architecture
        # Convolutional layers with BatchNorm
        self.conv1 = nn.Conv2d(inputs_shape, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(128)

        # Locally connected layers with BatchNorm and Dropout
        self.lc1 = nn.Conv2d(128, 64, kernel_size=1, groups=2)  # Adjusted 2
        self.bn_lc1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.5)  # Changed probability to 0.5 for consistency
        self.lc2 = nn.Conv2d(64, 256, kernel_size=1, groups=4)  # Adjusted for 4
        self.bn_lc2 = nn.BatchNorm2d(256)
        self.dropout2 = nn.Dropout(0.5)  # Changed probability to 0.5 for consistency

        # Fully connected layer
        self.fc = nn.Linear(64 * 64, output_shape)  # Assuming 10 classes for CIFAR-10

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = F.relu(self.conv5(x))
        x = self.bn5(x)
        x = F.relu(self.conv6(x))
        x = self.bn6(x)
        x = F.relu(self.conv7(x))
        x = self.bn7(x)

        x = F.relu(self.lc1(x))
        x = self.bn_lc1(x)
        x = self.dropout1(x)
        x = F.relu(self.lc2(x))
        x = self.bn_lc2(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)





# Best version refactored
class TreeConnect_enhanced_BN_RF(nn.Module):
    def __init__(self,inputs_shape,output_shape):
        super(TreeConnect_enhanced_BN_RF, self).__init__()
        # init device
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self._device = torch.device(device)

        self.model = tree_connect_architecture(inputs_shape,output_shape).to(self._device)
        self.model_type = 'tree-connect'
        self.loss_func_name = ''
        self.file_path = ''
        self.model_hash = ''
        self.date = datetime.now().strftime("%Y-%m-%d")

        self.training_loss = 0.0
        self.validation_loss = 0.0
        self.mean = 0.0
        self.standard_deviation = 0.0


        # Training Parameters
        self.epochs = 50
        self.learning_rate = 0.001
        self.train_percent = 80
        self.training_batch_size = 64
        self.weight_decay = 0.0001
        self.pooling_strategy = None
        self.add_loss_penalty = None
        self.target_option = None
        self.duplicate_flip_option = None
        self.randomize_data_per_epoch = None

        # list of models per epoch
        self.models_per_epoch = []
        self.lowest_loss_model_epoch = None


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




    def train(self,
              training_batch_size=64,
              epochs=10,
              learning_rate=0.01,
              weight_decay=0.001,
              add_loss_penalty=False,
              randomize_data_per_epoch=False,
              debug_asserts=False,
              penalty_range=5.00,
              modelSaveName='',
              input_shape=3,
              output_shape=10):

      # Create model
      self.model = tree_connect_architecture(input_shape, output_shape).to(device=self._device)
      print(self.model)


      # Define data transformations
      transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

      # Download CIFAR-10 training dataset
      train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

      # Create a DataLoader for the training dataset
      train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

      # Training parameters
      self.criterion = nn.CrossEntropyLoss()
      # weight_decay=1e-4
      #optimizer = optim.Adam([{'params': model.parameters()}], lr=learning_rate,weight_decay=0.0001)
      #self.optimizer = optim.Adam(model.parameters(), lr= self.learning_rate)
      self.optimizer =optim.Adam([{'params': self.model.parameters()}], lr=self.learning_rate)
      #  optimizer = optim.Adam([{'params': model.parameters()}], lr=learning_rate, weight_decay=0.0001)
      scaler = GradScaler()
    # Lists to store training and validation metrics
      train_loss_list = []
      train_accuracy_list = []
      val_loss_list = []
      val_accuracy_list = []

        # Record the start time
      start_time = time.time()


      # Training loop
      for epoch in range(epochs):
          self.model.train()
          total_loss = 0.0
          correct_train = 0
          total_train = 0

          for inputs, labels in train_loader:
              self.optimizer.zero_grad()

              with autocast():
                  outputs = self.model(inputs.to(self._device))
                  loss = self.criterion(outputs.to(self._device), labels.to(self._device))

              scaler.scale(loss).backward()
              scaler.step(self.optimizer)
              scaler.update()

              total_loss += loss.item()
              _, predicted = outputs.max(1)
              total_train += labels.size(0)
              correct_train += predicted.eq(labels.to(self._device)).sum().item()


          # Calculate training accuracy and loss
          train_accuracy = 100.0 * correct_train / total_train
          train_loss = total_loss / len(train_loader)

          # Validation
          self.model.eval()
          total_val_loss = 0.0
          correct_val = 0
          total_val = 0


          # Load the validation set
          val_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

          # Create a DataLoader for the validation dataset
          batch_size_val = self.training_batch_size
          val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)

          # Run validation tests
          with torch.no_grad():
              for inputs, labels in val_loader:
                  outputs = self.model(inputs.to(device=self._device))
                  loss = self.criterion(outputs.to(device=self._device), labels.to(device=self._device))

                  total_val_loss += loss.item()
                  _, predicted = outputs.max(1)
                  total_val += labels.size(0)
                  correct_val += predicted.eq(labels.to(device=self._device)).sum().item()

          # Calculate validation accuracy and loss
          val_accuracy = 100.0 * correct_val / total_val
          val_loss = total_val_loss / len(val_loader)

          # Append metrics to lists
          train_loss_list.append(train_loss)
          train_accuracy_list.append(train_accuracy)
          val_loss_list.append(val_loss)
          val_accuracy_list.append(val_accuracy)

          # Print and display progress
          print(f'Epoch [{epoch+1}/{epochs}], '
                f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, '
                f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')



      # Record the end time
      end_time = time.time()

      # Calculate the elapsed time
      elapsed_time = end_time - start_time
      print("Time elapsed: ",elapsed_time)

      # Plotting the training and validation curves
      plt.figure(figsize=(12, 4))

      # Plotting the loss curves
      plt.subplot(1, 2, 1)
      plt.plot(train_loss_list, label='Training Loss', marker='o')
      plt.plot(val_loss_list, label='Validation Loss', marker='o')
      plt.title('Loss Curves')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.legend()

      # Plotting the accuracy curves
      plt.subplot(1, 2, 2)
      plt.plot(train_accuracy_list, label='Training Accuracy', marker='o')
      plt.plot(val_accuracy_list, label='Validation Accuracy', marker='o')
      plt.title('Accuracy Curves')
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy (%)')
      plt.legend()

      plt.tight_layout()
      plt.show()

      # Save the trained model
      torch.save(self.model.state_dict(), modelSaveName)


      # Evaluate
    def evalualte(self,
                  model_path,
                  batchSize):
      # Define the transformation for your images
      transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

      # Validation dataset
      val_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

      # Create a DataLoader for the validation dataset
      batch_size = batchSize  # BS
      test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
      test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

      total_val_loss = 0.0
      correct_val = 0
      total_val = 0

      # load model
      # Load the pre-trained weights
      self.model.load_state_dict(torch.load(model_path))

      #Run validation tests
      with torch.no_grad():
          for inputs, labels in test_loader:
              inputs, labels = inputs.to(self._device), labels.to(self._device)
              outputs = self.model(inputs)
              loss = self.add_loss_penaltycriterion(outputs, labels)

              total_val_loss += loss.item()
              _, predicted = outputs.max(1)
              total_val += labels.size(0)
              correct_val += predicted.eq(labels).sum().item()

      # Calculate accuracy and average loss
      test_accuracy = 100.0 * correct_val / total_val
      val_loss = total_val_loss / len(test_loader)

      # print Validation Accuracy
      print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {test_accuracy:.2f}%')



# Best version
class TreeConnect_enhanced_BN(nn.Module):
    def __init__(self):
        super(TreeConnect_enhanced_BN, self).__init__()
        # Convolutional layers with BatchNorm
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(128)

        # Locally connected layers with BatchNorm and Dropout
        self.lc1 = nn.Conv2d(128, 64, kernel_size=1, groups=2)  # Adjusted for 4 groups with 4 channels each
        self.bn_lc1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.5)  # Changed probability to 0.5 for consistency
        self.lc2 = nn.Conv2d(64, 256, kernel_size=1, groups=4)  # Adjusted for 4 groups with 64 channels each
        self.bn_lc2 = nn.BatchNorm2d(256)
        self.dropout2 = nn.Dropout(0.5)  # Changed probability to 0.5 for consistency

        # Fully connected layer
        self.fc = nn.Linear(64 * 64, 10)  # Assuming 10 classes for CIFAR-10

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = F.relu(self.conv5(x))
        x = self.bn5(x)
        x = F.relu(self.conv6(x))
        x = self.bn6(x)
        x = F.relu(self.conv7(x))
        x = self.bn7(x)

        x = F.relu(self.lc1(x))
        x = self.bn_lc1(x)
        x = self.dropout1(x)
        x = F.relu(self.lc2(x))
        x = self.bn_lc2(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# Best version







# OLD Standard Version
class PyTorchModelTreeConnect(nn.Module):
    def __init__(self):
        super(PyTorchModelTreeConnect, self).__init__()
        # Convolutional layers:
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)  # Assuming input channels = 3
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)

        # # Flattening:
        self.flatten = nn.Flatten()

        # Dense layers:
        self.dense1 = nn.Linear(2048, 256)  # Assuming output of reshape and local_conv2 is 4096
        self.dense2 = nn.Linear(256, 128)
        self.dense3 = nn.Linear(128, 10)  # Correct number of outputs


    def forward(self, x):
        # Forward pass through the layers:
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.conv4(x)
        x = nn.functional.relu(x)
        x = self.conv5(x)
        x = nn.functional.relu(x)
        x = self.conv6(x)
        x = nn.functional.relu(x)
        x = x.flatten(start_dim=1)
        x = self.dense1(x)
        x = nn.functional.relu(x)
        x = self.dense2(x)
        x = nn.functional.relu(x)

        # Standard
        x = self.dense3(x)
        return F.log_softmax(x, dim=1)


# General Class Used to compare the architectures
class GeneralModel(nn.Module):
    def __init__(self):
        super(GeneralModel, self).__init__()
        # Standard Conv Block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1)


        if architecture == 'treeConnect':
          # Locally conntected layers
          self.lc1 = nn.Conv2d(128, 64, kernel_size=1, groups=2)  # Adjusted for 2
          self.lc2 = nn.Conv2d(64, 256, kernel_size=1, groups=4)  # Adjusted for 4
          self.fc = nn.Linear(64 * 64, 10)  # Last layer

        else: # it's FC
           self.fc1 = nn.Linear(2048, 256) # FC
           self.fc2 = nn.Linear(256, 256) # FC
           self.fc = nn.Linear(256, 10)  # Last layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        if architecture == 'treeConnect':
          x = F.relu(self.lc1(x))
          x = F.relu(self.lc2(x))
        else:
          # Flatter the output
          x = x.view(x.size(0), -1)
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))

        # Standard
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)



class TreeConnect(nn.Module):
    def __init__(self):
        super(TreeConnect, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1)

        # self.lc2 = nn.Conv2d(16128, 256, kernel_size=1, groups=16)
        # Locally connected layers
        self.lc1 = nn.Conv2d(128, 64, kernel_size=1, groups=2)  # Adjusted for 2
        self.lc2 = nn.Conv2d(64, 256, kernel_size=1, groups=4)  # Adjusted for 4 groups with 64 channels each

        # Fully connected layers
        #self.fc = nn.Linear(64 * 64, 10)
        #self.fc1 = nn.Linear(16 * 128, 256)
        self.fc = nn.Linear(64 * 64, 10)  # Assuming 10 classes for CIFAR-10

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        #x = x.view(x.size(0), 128, -1)
        x = F.relu(self.lc1(x))
        #x = x.view(x.size(0), 16, -1).permute(0, 2, 1)
        x = F.relu(self.lc2(x))

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class TreeConnect_enhanced(nn.Module):
    def __init__(self):
        super(TreeConnect_enhanced, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1)

        # Locally connected layers
        self.lc1 = nn.Conv2d(128, 64, kernel_size=1, groups=2)  # Adjusted for 4 groups with 4 channels each
        self.dropout1 = nn.Dropout(0.6)  # Add dropout with a probability of 0.5
        self.lc2 = nn.Conv2d(64, 256, kernel_size=1, groups=4)  # Adjusted for 4 groups with 64 channels each
        self.dropout2 = nn.Dropout(0.6)  # Add dropout with a probability of 0.5

        # Fully connected layers
        #self.fc = nn.Linear(64 * 64, 10)
        #self.fc1 = nn.Linear(16 * 128, 256)
        self.fc = nn.Linear(64 * 64, 10)  # Assuming 10 classes for CIFAR-10

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        #x = x.view(x.size(0), 128, -1)
        x = F.relu(self.lc1(x))
        self.dropout1 = nn.Dropout(0.6)  # Add dropout with a probability of 0.5
        #x = x.view(x.size(0), 16, -1).permute(0, 2, 1)
        x = F.relu(self.lc2(x))
        self.dropout2 = nn.Dropout(0.6)  # Add dropout with a probability of 0.5
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)




#######################


class FullConnect(nn.Module):
    def __init__(self):
        super(FullConnect, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(2048, 256)  # Updated to match the number of features
        self.fc2 = nn.Linear(256, 256)
        self.fc = nn.Linear(256, 10)  # Assuming 10 classes for CIFAR-10

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        # Flatten the output
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        #x = x.view(x.size(0), 16, -1).permute(0, 2, 1)
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)




class FullConnect_fat(nn.Module):
    def __init__(self):
        super(FullConnect_fat, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(2048, 2048)  # Updated to match the number of features
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.fc4 = nn.Linear(2048, 2048)
        self.fc5 = nn.Linear(2048, 2048)
        self.fc6 = nn.Linear(2048, 2048)
        self.fc7 = nn.Linear(2048, 256)
        self.fc8 = nn.Linear(256, 256)
        self.fc9 = nn.Linear(256, 256)
        self.fc10 = nn.Linear(256, 256)
        self.fc11 = nn.Linear(256, 256)
        self.fc12 = nn.Linear(256, 256)
        self.fc13 = nn.Linear(256, 256)
        self.fc = nn.Linear(256, 10)  # Assuming 10 classes for CIFAR-10

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        # Flatten the output
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        #x = x.view(x.size(0), 16, -1).permute(0, 2, 1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc12(x))
        x = F.relu(self.fc13(x))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)



class FullConnect_enhanced(nn.Module):
    def __init__(self):
        super(FullConnect_enhanced, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        # Fully connected layers
        self.fc1 = nn.Linear(2048, 256)  # Updated to match the number of features
        self.dropout1 = nn.Dropout(0.5)  # Add dropout with a probability of 0.5
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.5)  # Add dropout with a probability of 0.5
        self.fc = nn.Linear(256, 10)  # Assuming 10 classes for CIFAR-10

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = F.relu(self.conv5(x))
        x = self.bn5(x)
        x = F.relu(self.conv6(x))
        x = self.bn6(x)
        x = F.relu(self.conv7(x))
        x = self.bn7(x)

        # Flatten the output
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x) # Dropout 1
        #x = x.view(x.size(0), 16, -1).permute(0, 2, 1)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x) # Dropout 2
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class RegularSparseLocalLayer(nn.Module):
    def __init__(self, in_features, out_features, distance_factor):
        super(RegularSparseLocalLayer, self).__init__()

        self.mask = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        self.distance_factor = distance_factor


        # Set a regular sparsity pattern based on distance
        for i in range(out_features):
            start = max(0, i - distance_factor)  # Ensure start index is within bounds
            end = min(in_features, i + distance_factor + 1)  # Ensure end index is within bounds
            self.mask[i, start:end] = 1


        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.bias = nn.Parameter(torch.Tensor(out_features))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        masked_weight = self.mask * self.weight
        return F.linear(x, masked_weight, self.bias)

class RegularSparseLayer(nn.Module):
    def __init__(self, in_features, out_features, sparsity_factor):
        super(RegularSparseLayer, self).__init__()

        self.mask = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        self.sparsity_factor = sparsity_factor

        # Set a regular sparsity pattern where each neuron is connected to a fixed number of neurons
        for i in range(out_features):
            self.mask[i, i * sparsity_factor : (i + 1) * sparsity_factor] = 1

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.bias = nn.Parameter(torch.Tensor(out_features))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        masked_weight = self.mask * self.weight
        return F.linear(x, masked_weight, self.bias)

class FractionalSparseLayer(nn.Module):
    def __init__(self, in_features, out_features, connection_fraction):
        super(FractionalSparseLayer, self).__init__()

        self.mask = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        self.connection_fraction = connection_fraction

        # Calculate the number of connections to keep for each neuron
        num_connections_to_keep = int(in_features * connection_fraction)

        # Set a random sparsity pattern for each neuron
        for i in range(out_features):
            indices_to_keep = torch.randperm(in_features)[:num_connections_to_keep]
            self.mask[i, indices_to_keep] = 1

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.bias = nn.Parameter(torch.Tensor(out_features))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        masked_weight = self.mask * self.weight
        return F.linear(x, masked_weight, self.bias)



class RandomSparseLayer(nn.Module):
    def __init__(self, in_features, out_features, percentage_masked):
        super(RandomSparseLayer, self).__init__()

        self.mask = nn.Parameter(torch.ones(out_features, in_features), requires_grad=False)
        self.prune_percentage = percentage_masked

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.bias = nn.Parameter(torch.Tensor(out_features))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

        self.reset_mask()

    def reset_mask(self):
        num_elements = int(self.mask.numel() * (self.prune_percentage / 100.0))
        indices_to_prune = torch.randperm(self.mask.numel())[:num_elements]
        self.mask.view(-1)[indices_to_prune] = 0

    def forward(self, x):
        masked_weight = self.mask * self.weight
        return F.linear(x, masked_weight, self.bias)



class Random_Sparse_enhanced(nn.Module):
    def __init__(self):
        super(Random_Sparse_enhanced, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(128)




        # # Fractional Sparse Layers
        # connection_fraction1 = 0.5  # Connect each neuron to half of the original connections
        # self.sparse_layer1 = FractionalSparseLayer(2048, 256, connection_fraction1)

        # Random Sparse Layers RegularSparseLocalLayer
        self.sparse_layer1 = RegularSparseLocalLayer(2048, 256, distance_factor=64)
        #self.dropout1 = nn.Dropout(0.6)
        self.sparse_layer2 = RegularSparseLocalLayer(256, 256, distance_factor=32)
        #self.dropout2 = nn.Dropout(0.6)


        self.fc = nn.Linear(256, 10)  # Assuming 10 classes for CIFAR-10

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = F.relu(self.conv5(x))
        x = self.bn5(x)
        x = F.relu(self.conv6(x))
        x = self.bn6(x)
        x = F.relu(self.conv7(x))
        x = self.bn7(x)

        # Flatten the output
        x = x.view(x.size(0), -1)

        x = F.relu(self.sparse_layer1(x))
        #x = self.dropout1(x)  # Dropout 1
        x = F.relu(self.sparse_layer2(x))
        #x = self.dropout2(x)  # Dropout 2
        x = self.fc(x)
        return F.log_softmax(x, dim=1)



def TrainModel_Automated_Mixed_Precision(modelclass,modelSaveName):


  # Create model
  model = modelclass().to(device=device)
  print(model)


  # Define data transformations
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  # Download CIFAR-10 training dataset
  train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

  # Create a DataLoader for the training dataset
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

  # Training parameters
  criterion = nn.CrossEntropyLoss()
  # weight_decay=1e-4
  #optimizer = optim.Adam([{'params': model.parameters()}], lr=learning_rate,weight_decay=0.0001)
  optimizer = optim.SGD(model.parameters(), lr=0.01 )
  scaler = GradScaler()
# Lists to store training and validation metrics
  train_loss_list = []
  train_accuracy_list = []
  val_loss_list = []
  val_accuracy_list = []




  # Record the start time
  start_time = time.time()

  # Training loop
  for epoch in range(epochs):
      model.train()
      total_loss = 0.0
      correct_train = 0
      total_train = 0

      for inputs, labels in train_loader:
          optimizer.zero_grad()
          with autocast(dtype=torch.float16):
            output = model(input)
            loss = criterion(outputs.to(device), labels.to(device))


          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()

          total_loss += loss.item()
          _, predicted = outputs.max(1)
          total_train += labels.size(0)
          correct_train += predicted.eq(labels.to(device)).sum().item()

      # Calculate training accuracy and loss
      train_accuracy = 100.0 * correct_train / total_train
      train_loss = total_loss / len(train_loader)

      # Validation
      model.eval()
      total_val_loss = 0.0
      correct_val = 0
      total_val = 0


      # Load the validation set
      val_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

      # Create a DataLoader for the validation dataset
      batch_size_val = 64
      val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)

      # Run validation tests
      with torch.no_grad():
          for inputs, labels in val_loader:
              outputs = model(inputs.to(device=device))
              loss = criterion(outputs.to(device=device), labels.to(device=device))

              total_val_loss += loss.item()
              _, predicted = outputs.max(1)
              total_val += labels.size(0)
              correct_val += predicted.eq(labels.to(device=device)).sum().item()

      # Calculate validation accuracy and loss
      val_accuracy = 100.0 * correct_val / total_val
      val_loss = total_val_loss / len(val_loader)

      # Append metrics to lists
      train_loss_list.append(train_loss)
      train_accuracy_list.append(train_accuracy)
      val_loss_list.append(val_loss)
      val_accuracy_list.append(val_accuracy)

      # Print and display progress
      print(f'Epoch [{epoch+1}/{epochs}], '
            f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, '
            f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')



  # Record the end time
  end_time = time.time()

  # Calculate the elapsed time
  elapsed_time = end_time - start_time
  print("Time elapsed: ",elapsed_time)

  # Plotting the training and validation curves
  plt.figure(figsize=(12, 4))

  # Plotting the loss curves
  plt.subplot(1, 2, 1)
  plt.plot(train_loss_list, label='Training Loss', marker='o')
  plt.plot(val_loss_list, label='Validation Loss', marker='o')
  plt.title('Loss Curves')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()

  # Plotting the accuracy curves
  plt.subplot(1, 2, 2)
  plt.plot(train_accuracy_list, label='Training Accuracy', marker='o')
  plt.plot(val_accuracy_list, label='Validation Accuracy', marker='o')
  plt.title('Accuracy Curves')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy (%)')
  plt.legend()

  plt.tight_layout()
  plt.show()

  # Save the trained model
  torch.save(model.state_dict(), modelSaveName)



########################################################################
# Train and plot (AMP V2)
########################################################################


def TrainModel_AMP_V2(modelclass,modelSaveName):


  # Create model
  model = modelclass().to(device=device)
  print(model)


  # Define data transformations
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  # Download CIFAR-10 training dataset
  train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

  # Create a DataLoader for the training dataset
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)



# Lists to store training and validation metrics
  train_loss_list = []
  train_accuracy_list = []
  val_loss_list = []
  val_accuracy_list = []

  # Training parameters
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam([{'params': model.parameters()}], lr=learning_rate, weight_decay=0.0001)

  scaler = GradScaler()

  # Record the start time
  start_time = time.time()


  # Training loop
  for epoch in range(epochs):
      model.train()
      total_loss = 0.0
      correct_train = 0
      total_train = 0

      for inputs, labels in train_loader:
          optimizer.zero_grad()

          with autocast():
              outputs = model(inputs.to(device))
              loss = criterion(outputs.to(device), labels.to(device))

          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()

          total_loss += loss.item()
          _, predicted = outputs.max(1)
          total_train += labels.size(0)
          correct_train += predicted.eq(labels.to(device)).sum().item()


      # Calculate training accuracy and loss
      train_accuracy = 100.0 * correct_train / total_train
      train_loss = total_loss / len(train_loader)

      # Validation
      model.eval()
      total_val_loss = 0.0
      correct_val = 0
      total_val = 0


      # Load the validation set
      val_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

      # Create a DataLoader for the validation dataset
      batch_size_val = 64
      val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)

      # Run validation tests
      with torch.no_grad():
          for inputs, labels in val_loader:
              outputs = model(inputs.to(device=device))
              loss = criterion(outputs.to(device=device), labels.to(device=device))

              total_val_loss += loss.item()
              _, predicted = outputs.max(1)
              total_val += labels.size(0)
              correct_val += predicted.eq(labels.to(device=device)).sum().item()

      # Calculate validation accuracy and loss
      val_accuracy = 100.0 * correct_val / total_val
      val_loss = total_val_loss / len(val_loader)

      # Append metrics to lists
      train_loss_list.append(train_loss)
      train_accuracy_list.append(train_accuracy)
      val_loss_list.append(val_loss)
      val_accuracy_list.append(val_accuracy)

      # Print and display progress
      print(f'Epoch [{epoch+1}/{epochs}], '
            f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, '
            f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')



  # Record the end time
  end_time = time.time()

  # Calculate the elapsed time
  elapsed_time = end_time - start_time
  print("Time elapsed: ",elapsed_time)

  # Plotting the training and validation curves
  plt.figure(figsize=(12, 4))

  # Plotting the loss curves
  plt.subplot(1, 2, 1)
  plt.plot(train_loss_list, label='Training Loss', marker='o')
  plt.plot(val_loss_list, label='Validation Loss', marker='o')
  plt.title('Loss Curves')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()

  # Plotting the accuracy curves
  plt.subplot(1, 2, 2)
  plt.plot(train_accuracy_list, label='Training Accuracy', marker='o')
  plt.plot(val_accuracy_list, label='Validation Accuracy', marker='o')
  plt.title('Accuracy Curves')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy (%)')
  plt.legend()

  plt.tight_layout()
  plt.show()

  # Save the trained model
  torch.save(model.state_dict(), modelSaveName)



########################################################################
# Train and plot (Standard)
########################################################################

def TrainModel(modelclass,modelSaveName):


  # Create model
  model = modelclass().to(device=device)
  print(model)


  # Define data transformations
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  # Download CIFAR-10 training dataset
  train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

  # Create a DataLoader for the training dataset
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

  # Training parameters
  criterion = nn.CrossEntropyLoss()
  # weight_decay=1e-4
  optimizer = optim.Adam([{'params': model.parameters()}], lr=learning_rate,weight_decay=0.0001)


# Lists to store training and validation metrics
  train_loss_list = []
  train_accuracy_list = []
  val_loss_list = []
  val_accuracy_list = []


  # Record the start time
  start_time = time.time()

  # Training loop
  for epoch in range(epochs):
      model.train()
      total_loss = 0.0
      correct_train = 0
      total_train = 0

      for inputs, labels in train_loader:
          optimizer.zero_grad()
          outputs = model(inputs.to(device))
          loss = criterion(outputs.to(device), labels.to(device))
          loss.backward()
          optimizer.step()

          total_loss += loss.item()
          _, predicted = outputs.max(1)
          total_train += labels.size(0)
          correct_train += predicted.eq(labels.to(device)).sum().item()

      # Calculate training accuracy and loss
      train_accuracy = 100.0 * correct_train / total_train
      train_loss = total_loss / len(train_loader)

      # Validation
      model.eval()
      total_val_loss = 0.0
      correct_val = 0
      total_val = 0


      # Load the validation set
      val_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

      # Create a DataLoader for the validation dataset
      batch_size_val = 64
      val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)

      # Run validation tests
      with torch.no_grad():
          for inputs, labels in val_loader:
              outputs = model(inputs.to(device=device))
              loss = criterion(outputs.to(device=device), labels.to(device=device))

              total_val_loss += loss.item()
              _, predicted = outputs.max(1)
              total_val += labels.size(0)
              correct_val += predicted.eq(labels.to(device=device)).sum().item()

      # Calculate validation accuracy and loss
      val_accuracy = 100.0 * correct_val / total_val
      val_loss = total_val_loss / len(val_loader)

      # Append metrics to lists
      train_loss_list.append(train_loss)
      train_accuracy_list.append(train_accuracy)
      val_loss_list.append(val_loss)
      val_accuracy_list.append(val_accuracy)

      # Print and display progress
      print(f'Epoch [{epoch+1}/{epochs}], '
            f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, '
            f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')



  # Record the end time
  end_time = time.time()

  # Calculate the elapsed time
  elapsed_time = end_time - start_time
  print("Time elapsed: ",elapsed_time)

  # Plotting the training and validation curves
  plt.figure(figsize=(12, 4))

  # Plotting the loss curves
  plt.subplot(1, 2, 1)
  plt.plot(train_loss_list, label='Training Loss', marker='o')
  plt.plot(val_loss_list, label='Validation Loss', marker='o')
  plt.title('Loss Curves')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()

  # Plotting the accuracy curves
  plt.subplot(1, 2, 2)
  plt.plot(train_accuracy_list, label='Training Accuracy', marker='o')
  plt.plot(val_accuracy_list, label='Validation Accuracy', marker='o')
  plt.title('Accuracy Curves')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy (%)')
  plt.legend()

  plt.tight_layout()
  plt.show()

  # Save the trained model
  torch.save(model.state_dict(), modelSaveName)





######################################################################################################
#                                Run inference -> Evaluation
######################################################################################################

def evalualte_model(self,model_path,batchSize):
  # Define the transformation for your images
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  # Validation dataset
  val_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

  # Create a DataLoader for the validation dataset
  batch_size = batchSize  # BS
  test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  total_val_loss = 0.0
  correct_val = 0
  total_val = 0

  # load model
  # Load the pre-trained weights
  self.model.load_state_dict(torch.load(model_path))

  #Run validation tests
  with torch.no_grad():
      for inputs, labels in test_loader:
          inputs, labels = inputs.to(self.device), labels.to(self.device)
          outputs = self.model(inputs)
          loss = self.criterion(outputs, labels)

          total_val_loss += loss.item()
          _, predicted = outputs.max(1)
          total_val += labels.size(0)
          correct_val += predicted.eq(labels).sum().item()

  # Calculate accuracy and average loss
  test_accuracy = 100.0 * correct_val / total_val
  val_loss = total_val_loss / len(test_loader)

  # print Validation Accuracy
  print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {test_accuracy:.2f}%')


######################################################################################################
#                                Run inference -> Speed
######################################################################################################

def calculate_model_speed(modelclass,modelName,batchsize):

  # Define the transformation for your images
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


  # Create a DataLoader for the validation dataset
  batch_size = batchsize  # BS
  # Test data 10000
  test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  # Check images number
  total_images = len(test_loader.dataset)
  print(f'Total number of images in the test set: {total_images}')

  # load model
  model = modelclass().to(device=device) # FullConnect() #  TreeConnect()
  # Load the pre-trained weights
  model.load_state_dict(torch.load(modelName))
  # Start inference time
  inf_start_time = time.time()
  #Evaluate the model
  model.eval()
  criterion = nn.CrossEntropyLoss()

  # Run inference on the test dataset
  # Lists to store predictions and ground truth labels
  all_predictions = []
  all_labels = []
  with torch.no_grad():
      for inputs , labels in test_loader:
          inputs, labels = inputs.to(device), labels.to(device)
          outputs = model(inputs)
          _, predicted = outputs.max(1)

  # End inference time
  inf_end_time = time.time()

  # Calculate the elapsed time
  inf_elapsed_time = inf_end_time - inf_start_time
  print("Inference time elapsed: ",inf_elapsed_time)
  return inf_elapsed_time


def calculate_inference_speed_forXruns(modelName,batchsize,numberOfRuns):
  infTime = 0
  for i in range(numberOfRuns):
    infTime += calculate_model_speed(modelName,batchsize)
  return (infTime/numberOfRuns)

#Includes saving time
def calculate_model_speedv2(modelclass, modelName, batchsize):
    # Define the transformation for your images
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    # Create a DataLoader for the validation dataset
    batch_size = batchsize  # BS
    # Test data 10000
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Check images number
    total_images = len(test_loader.dataset)
    print(f'Total number of images in the test set: {total_images}')

    # load model
    model = modelclass().to(device=device) # FullConnect() #  TreeConnect()
    # Load the pre-trained weights
    model.load_state_dict(torch.load(modelName))

    # Start inference time
    inf_start_time = time.time()

    # Lists to store predictions and ground truth labels
    all_predictions = []
    all_labels = []

    # Evaluate the model
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Run inference on the test dataset
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            # Append predictions and labels to the lists
            all_predictions.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # End inference time
    inf_end_time = time.time()

    # Calculate the elapsed time
    inf_elapsed_time = inf_end_time - inf_start_time
    print("Inference time elapsed: ", inf_elapsed_time)

    return inf_elapsed_time



######################################################################################################
#                                Run functions here
######################################################################################################






# Best version Run here



# # Create model
# # 3 channels input , 10 outputs
# newmodel = TreeConnect_enhanced_BN_RF(3,10)
# newmodel.training_batch_size=64
# newmodel.epochs=3
# newmodel.learning_rate=0.01
# newmodel.weight_decay=0.001
# #Train and save
# newmodel.train(modelSaveName='blabla2.pth')






loadedmodel = TreeConnect_enhanced_BN_RF(3,10)
loadedmodel.train(modelSaveName='blabla3.pth',epochs=20)
loadedmodel.evalualte(model_path="blabla3.pth", batchSize=64)  # Correct call







#Load model
#evalualte_model(TreeConnect_enhanced_BN,'tc_bn_v1.pth',64)


# TrainModel_AMP_V2(TreeConnect_enhanced_BN,"TC_automated_MP.pth")
# #TrainModel(Random_Sparse_enhanced,'sp_locally_30_.pth') #FullConnect Random_Sparse_enhanced
# #evalualte_model(TreeConnect_enhanced_BN,'tc_bn_v1.pth',64)

# #evalualte_model(TreeConnect_enhanced_BN,'TC_automated_MP.pth',1024)


# # Profile the forward pass
# with profiler.profile(record_shapes=True, use_cuda=torch.cuda.is_available()) as prof:
#   TrainModel_AMP_V2(TreeConnect_enhanced_BN,"TC_automated_MP_v2.pth")





  #evalualte_model(TreeConnect_enhanced_BN,'TC_automated_MP.pth',1024)
  # rangex = 30
  # inf_elapsed_timeT =0
  # for i in range(rangex):
  #   inf_elapsed_timeT += calculate_model_speed(TreeConnect_enhanced_BN,'TC_automated_MP.pth',1024)
  # print("inference time is: ", inf_elapsed_timeT/(rangex) )












# # load model
# architecture = 'treeConnect'
# modelx =  TreeConnect_enhanced_BN().to(device=device) # FullConnect # TreeConnect()
# # Load the pre-trained weights
# modelx.load_state_dict(torch.load('TC_automated_MP.pth'))
# print(summary(modelx.to(device=device), input_size=(3, 32, 32)))






#calculate_model_speed('fullconnect_model_v4_g.pth',1024)
#evalualte_model('treeconnect_model_v4_g.pth',1024) # fullconnect_model_v3  treeconnect_model_v3

# architecture = 'treeConnect'
# print("The average inference time for tree connect on batch size 10000 is: ",calculate_inference_speed_forXruns('treeconnect_model_v6_g.pth',10000,100))


# architecture = 'fullConnect'
# print("The average inference time for fully connected on batch size 16 is: ",calculate_inference_speed_forXruns('fullconnect_model_v6_g.pth',16,100))


# architecture = 'treeConnect'
# calculate_model_speed('treeconnect_model_v6_g.pth',16)

# architecture = 'fullConnect'
# calculate_model_speed('fullconnect_model_v6_g.pth',16)


