import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import time
import numpy as np
from torchsummary import summary

architecture = 'fullConnect'

# General Class
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
          self.lc1 = nn.Conv2d(128, 64, kernel_size=1, groups=2)  # Adjusted for 4 groups with 4 channels each
          self.lc2 = nn.Conv2d(64, 256, kernel_size=1, groups=4)  # Adjusted for 4 groups with 64 channels each
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

        # Locally connected part
        # self.lc1 = nn.Conv2d(128, 16128, kernel_size=1, groups=128)
        # self.lc2 = nn.Conv2d(16128, 256, kernel_size=1, groups=16)
        # Locally connected layers
        self.lc1 = nn.Conv2d(128, 64, kernel_size=1, groups=2)  # Adjusted for 4 groups with 4 channels each
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








# class TreeConnectVx(nn.Module):
#     def init(self):
#         super(TreeConnectVx, self).init()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.conv6 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)

#         # Add more layers as in your Keras model...



#         self.lc1 = nn.Conv2d(256, 16128, kernel_size=1, groups=128)
#         self.lc2 = nn.Conv2d(16, 1616, kernel_size=1, groups=16)
#         self.fc = nn.Linear(16*16, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = F.relu(self.conv5(x))
#         x = F.relu(self.conv6(x))


#         x = x.view(x.size(0), 128, -1)
#         x = F.relu(self.lc1(x))
#         x = x.view(x.size(0), 16, -1).permute(0, 2, 1)
#         x = F.relu(self.lc2(x))

#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return F.log_softmax(x, dim=1)



# ######################



# class TreeConnectModelBasic(nn.Module):
#     def __init__(self):
#         super(TreeConnectModelBasic, self).__init__()

#         # Reshape input to 128x128 matrix
#         #self.reshape = nn.Reshape((128, 128))
#         self.reshape = lambda x: x.view(-1, 32, 32)

#         # First locally connected layer
#         self.local1 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0,groups=2)
#         self.relu1 = nn.ReLU()

#         # Permute dimensions
#         #self.permute = nn.Permute((2, 1))
#         self.permute = lambda x: x.permute(0, 2, 1)

#         # Second locally connected layer
#         self.local2 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0,groups=2)
#         self.relu2 = nn.ReLU()

#         # Flatten before dense layers
#         self.flatten = nn.Flatten()

#         # Fully connected layers
#         self.fc1 = nn.Linear(16 * 128, 256)
#         self.fc2 = nn.Linear(256, 10)  # Assuming 10 classes for CIFAR-10

#         # Softmax activation
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.reshape(x)
#         x = self.relu1(self.local1(x))
#         x = self.permute(x)
#         x = self.relu2(self.local2(x))
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.softmax(x)
#         return x


# Define CIFAR-10 specific parameters
# in_channels = 3  # RGB images
# out_channels = 64
# kernel_size = 3
# num_classes = 10

# class ParallelConvolutionalModel(nn.Module):
#     def __init__(self, in_channels = 3, out_channels = 48, kernel_size = 3, num_classes = 10):
#         super(ParallelConvolutionalModel, self).__init__()

#         # First parallel convolutional layer
#         self.conv1_1 = nn.Conv2d(in_channels, 48, kernel_size, stride=1, padding=1, groups=3)

#         # Second parallel convolutional layer
#         self.conv1_2 = nn.Conv2d(in_channels, 48, kernel_size, stride=1, padding=1, groups=3)

#         # Common subsequent layers
#         self.relu = nn.ReLU(inplace=True)
#         self.pool = nn.MaxPool2d(2, 2)
#         # Corrected input channels for conv2 layer
#         self.conv2 = nn.Conv2d(48 * 2, out_channels, kernel_size, stride=1, padding=1)
#         self.fc = nn.Linear(out_channels * 8 * 8, num_classes)

#     def forward(self, x):
#         # Branch 1
#         out1 = self.conv1_1(x)
#         out1 = self.relu(out1)
#         out1 = self.pool(out1)

#         # Branch 2
#         out2 = self.conv1_2(x)
#         out2 = self.relu(out2)
#         out2 = self.pool(out2)

#         # Concatenate the outputs of both branches
#         out = torch.cat((out1, out2), dim=1)

#         # Common layers
#         out = self.conv2(out)
#         out = self.relu(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)

#         return out



############################################## Train here



# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Instantiate the model
#tree_connect_model = TreeConnect().to(device=device)
gModel = GeneralModel().to(device=device)

# Display the model architecture
#print(tree_connect_model)
print(gModel)


# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 50

# # Load CIFAR-10 data
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Define data transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# Download CIFAR-10 training dataset
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

# Create a DataLoader for the training dataset
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)



#model = TreeConnect().to(device=device)
model = GeneralModel().to(device=device)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.Adam([{'params': model.parameters()}], lr=learning_rate)


################################################### Training loop ##########################################








# import matplotlib.pyplot as plt
# # Lists to store training and validation metrics
# train_loss_list = []
# train_accuracy_list = []
# val_loss_list = []
# val_accuracy_list = []



# # Record the start time
# start_time = time.time()

# # Training loop
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0.0
#     correct_train = 0
#     total_train = 0

#     for inputs, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs.to(device))
#         loss = criterion(outputs.to(device), labels.to(device))
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total_train += labels.size(0)
#         correct_train += predicted.eq(labels.to(device)).sum().item()

#     # Calculate training accuracy and loss
#     train_accuracy = 100.0 * correct_train / total_train
#     train_loss = total_loss / len(train_loader)

#     # Validation
#     model.eval()
#     total_val_loss = 0.0
#     correct_val = 0
#     total_val = 0


#     # Assuming you have a validation dataset (val_dataset)
#     val_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

#     # Create a DataLoader for the validation dataset
#     batch_size = 64  # You can adjust this based on your needs
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             outputs = model(inputs.to(device=device))
#             loss = criterion(outputs.to(device=device), labels.to(device=device))

#             total_val_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total_val += labels.size(0)
#             correct_val += predicted.eq(labels.to(device=device)).sum().item()

#     # Calculate validation accuracy and loss
#     val_accuracy = 100.0 * correct_val / total_val
#     val_loss = total_val_loss / len(val_loader)

#     # Append metrics to lists
#     train_loss_list.append(train_loss)
#     train_accuracy_list.append(train_accuracy)
#     val_loss_list.append(val_loss)
#     val_accuracy_list.append(val_accuracy)

#     # Print and display progress
#     print(f'Epoch [{epoch+1}/{epochs}], '
#           f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, '
#           f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')



# # Record the end time
# end_time = time.time()

# # Calculate the elapsed time
# elapsed_time = end_time - start_time
# print("Time elapsed: ",elapsed_time)

# # Plotting the training and validation curves
# plt.figure(figsize=(12, 4))

# # Plotting the loss curves
# plt.subplot(1, 2, 1)
# plt.plot(train_loss_list, label='Training Loss', marker='o')
# plt.plot(val_loss_list, label='Validation Loss', marker='o')
# plt.title('Loss Curves')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# # Plotting the accuracy curves
# plt.subplot(1, 2, 2)
# plt.plot(train_accuracy_list, label='Training Accuracy', marker='o')
# plt.plot(val_accuracy_list, label='Validation Accuracy', marker='o')
# plt.title('Accuracy Curves')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (%)')
# plt.legend()

# plt.tight_layout()
# plt.show()

# # Save the trained model
# torch.save(model.state_dict(), 'fullconnect_model_v4_g.pth')













######################################################################################################
#                                Run inference -> Evaluation
######################################################################################################

def evalualte_model(modelName,batchSize):
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
  model =  GeneralModel() # FullConnect # TreeConnect()
  # Load the pre-trained weights
  model.load_state_dict(torch.load(modelName))

  #Run validation tests
  with torch.no_grad():
      for inputs, labels in test_loader:
          outputs = model(inputs)
          loss = criterion(outputs, labels)

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

def calculate_model_speed(modelName,batchsize):

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
  model = GeneralModel() # FullConnect() #  TreeConnect()
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
      for inputs, labels in test_loader:
          #inputs, labels = inputs.to(device), labels.to(device)
          outputs = model(inputs)
          _, predicted = outputs.max(1)

  # End inference time
  inf_end_time = time.time()

  # Calculate the elapsed time
  inf_elapsed_time = inf_end_time - inf_start_time
  print("Inference time elapsed: ",inf_elapsed_time)




######################################################################################################
#                                Run functions here
######################################################################################################
calculate_model_speed('fullconnect_model_v4_g.pth',1024)
#evalualte_model('treeconnect_model_v4_g.pth',1024) # fullconnect_model_v3  treeconnect_model_v3




# load model
# architecture = 'treeConnect'
# modelx =  GeneralModel().to(device=device) # FullConnect # TreeConnect()
# # Load the pre-trained weights
# modelx.load_state_dict(torch.load('treeconnect_model_v3.pth'))
# print(summary(modelx.to(device=device), input_size=(3, 32, 32)))

