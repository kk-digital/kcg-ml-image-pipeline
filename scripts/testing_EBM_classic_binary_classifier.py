## Standard libraries
import os
import json
import math
import numpy as np
import random

## Imports for plotting
import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib.colors import to_rgb
import matplotlib
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from torch.utils.data import random_split, DataLoader

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
# PyTorch Lightning
try:
    import pytorch_lightning as pl
except ModuleNotFoundError: # Google Colab does not have PyTorch Lightning installed by default. Hence, we do it here if necessary
    #!pip install --quiet pytorch-lightning>=1.4
    import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint



# ADDED BY ME
from datetime import datetime
from pytz import timezone
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.datasets import SVHN
import torchvision.transforms as transforms
from io import BytesIO
import io
import os
import sys
base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from data_loader.ab_ranking_dataset_loader import ABRankingDatasetLoader
from utility.minio import cmd
from utility.clip.clip_text_embedder import tensor_attention_pooling

import urllib.request
from urllib.error import HTTPError
from torchvision.datasets import SVHN
from torch.utils.data import random_split, DataLoader
from PIL import Image
import requests
import msgpack 





########################################### Initialize the cuda device 
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


########################################### Initialize minio 

date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%d-%m-%Y %H:%M:%S')
print(date_now)


minio_client = cmd.get_minio_client("D6ybtPLyUrca5IdZfCIM",
            "2LZ6pqIGOiZGcjPTR6DZPlElWBkRTkaLkyLIBt4V",
            None)
minio_path="environmental/output/my_test"



###########################################

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.path import separate_bucket_and_file_path
from data_loader.utils import get_object

API_URL = "http://192.168.3.1:8111"








def get_tag_jobs(tag_id):
    response = requests.get(f'{API_URL}/tags/get-images-by-tag-id/?tag_id={tag_id}')
    
    # Check if the response is successful (status code 200)
    if response.status_code == 200:
        try:
            # Parse the JSON response
            response_data = json.loads(response.content)

            # Check if 'images' key is present in the JSON response
            if 'images' in response_data.get('response', {}):
                # Extract file paths from the 'images' key
                file_paths = [job['file_path'] for job in response_data['response']['images']]
                return file_paths
            else:
                print("Error: 'images' key not found in the JSON response.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    else:
        print(f"Error: HTTP request failed with status code {response.status_code}")

    # Return an empty list or appropriate value to indicate an error
    return []

########################################### Get clip vectors

def get_clip_vectors(file_paths):
        clip_vectors=[]
        for path in file_paths:
            clip_path= path.replace(".jpg","_clip_kandinsky.msgpack")
            bucket, features_vector_path= separate_bucket_and_file_path(clip_path) 
            features_data = get_object(minio_client, features_vector_path)
            features = msgpack.unpackb(features_data)["clip-feature-vector"]
            clip_vectors.append(features)
        return clip_vectors    



########################################### get images

def get_image(file_path: str):
    # get image from minio server
    bucket_name, file_path = separate_bucket_and_file_path(file_path)
    try:
        response = minio_client.get_object(bucket_name, file_path)
        image_data = BytesIO(response.data)
        img = Image.open(image_data)
        img = img.convert("RGB")
    except Exception as e:
        raise e
    finally:
        response.close()
        response.release_conn()

    return img


########################################### DATA Augmentation


def data_augmentation(images_tensor, num_of_passes):
    # Define probabilities for each transformation
    prob_mirror = 0.9
    prob_zoom = 0.5
    prob_rotation = 0.2
    prob_contrast = 0.5
    prob_brightness = 0.5

    # Apply data augmentation to each image in the array
    augmented_images = []

    for img in images_tensor:
        for _ in range(num_of_passes):
            transformed_img = img.clone()

            # Apply mirror transformation
            random_mirror = random.random()
            if random_mirror < prob_mirror:
                transformed_img = transforms.RandomHorizontalFlip()(transformed_img)

            # Apply zoom transformation
            random_zoom = random.random()
            if random_zoom < prob_zoom:
                transformed_img = transforms.RandomResizedCrop(size=(512, 512), scale=(0.9, 1.1), ratio=(0.9, 1.1))(transformed_img)

            # Apply rotation transformation
            random_rotation = random.random()
            if random_rotation < prob_rotation:
                transformed_img = transforms.RandomRotation(degrees=(-20, 20))(transformed_img)


            # New Augments
                
            # Apply contrast transformation
            random_contrast = random.random()
            if random_contrast < prob_contrast:
                transformed_img = transforms.ColorJitter(contrast=(0.5, 1.5))(transformed_img)

            # Apply brightness transformation
            random_brightness = random.random()
            if random_brightness < prob_brightness:
                transformed_img = transforms.ColorJitter(brightness=(0.2, 2))(transformed_img)



            augmented_images.append(transformed_img)

    # Convert the list of augmented images to a PyTorch tensor
    augmented_images_tensor = torch.stack(augmented_images)

    # Concatenate original and augmented images
    combined_images = images_tensor + list(augmented_images_tensor)

    return combined_images





def get_dataset_from_id(id_class,data_augment_passes,label_value):

    images_paths = get_tag_jobs(id_class)
    ocult_images = []


    for path in images_paths:
        ocult_images.append(get_image(path))


    # Transforme into tansors
    ocult_images = [transform(img) for img in ocult_images]


    # Call your data_augmentation function
    ocult_images = data_augmentation(ocult_images, data_augment_passes)


    print("Occult lenght : ",len(ocult_images))


    # Create labels
    # label_value = label_value
    # labels_occult = [label_value] * len(ocult_images)
    if label_value == 1:
        label_value = torch.tensor([1, 0], dtype=torch.float32)
    else:
        label_value = torch.tensor([0, 1], dtype=torch.float32)




    data_occcult = []
    for image in ocult_images:
        data_occcult.append((image, label_value))

    ocult_images = data_occcult
    num_samples_ocult = len(ocult_images)
    print("the number of samples in ocult ", num_samples_ocult)
    train_size_ocult = int(0.8 * num_samples_ocult)
    val_size_ocult = num_samples_ocult - train_size_ocult
    train_set_ocult, val_set_ocult = random_split(ocult_images, [train_size_ocult, val_size_ocult])
    return train_set_ocult,val_set_ocult




def get_combined_adv_dataset_from_id_array(id_classes,data_augment_passes,label_value):
    i = 0
    for class_id in id_classes:
        images_paths[i] = get_tag_jobs(class_id)
        i += 1



    ocult_images = []

    for j in range(i):
        for path in images_paths[j]:
            ocult_images.append(get_image(path))


    # Transforme into tansors
    ocult_images = [transform(img) for img in ocult_images]


    # Call your data_augmentation function
    ocult_images = data_augmentation(ocult_images, data_augment_passes)


    print("Occult lenght : ",len(ocult_images))


    # Create labels
    label_value = label_value
    labels_occult = [label_value] * len(ocult_images)

    if label_value == 1:
        label_value = torch.tensor([1, 0], dtype=torch.float32)
    else:
        label_value = torch.tensor([0, 1], dtype=torch.float32)


    data_occcult = []
    for image in ocult_images:
        data_occcult.append((image, label_value))

    ocult_images = data_occcult
    num_samples_ocult = len(ocult_images)
    print("the number of samples in ocult ", num_samples_ocult)
    train_size_ocult = int(0.8 * num_samples_ocult)
    val_size_ocult = num_samples_ocult - train_size_ocult
    train_set_ocult, val_set_ocult = random_split(ocult_images, [train_size_ocult, val_size_ocult])
    return train_set_ocult,val_set_ocult





##################### Load images

# Transformations: # don't use greyscale
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize after grayscale conversion
])


##################### Load images

batchsize_x = 64

#cybernetic: 35, occult: 39


################################################################### Load occults images
images_paths = get_tag_jobs(39)

ocult_images = []


for path in images_paths:
    ocult_images.append(get_image(path))


# Transforme into tansors
ocult_images = [transform(img) for img in ocult_images]


# Call your data_augmentation function
ocult_images = data_augmentation(ocult_images, 5)


print("Occult lenght : ",len(ocult_images))


# Create labels
label_value = 1
labels_occult = [label_value] * len(ocult_images)

data_occcult = []
for image in ocult_images:
    data_occcult.append((image, 1))

ocult_images = data_occcult
num_samples_ocult = len(ocult_images)
print("the number of samples in ocult ", num_samples_ocult)
train_size_ocult = int(0.8 * num_samples_ocult)
val_size_ocult = num_samples_ocult - train_size_ocult


#train_set_ocult, val_set_ocult = random_split(ocult_images, [train_size_ocult, val_size_ocult])
train_set_ocult, val_set_ocult = get_dataset_from_id(39,5,1)
train_loader_set_ocult = data.DataLoader(train_set_ocult, batch_size=batchsize_x, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
val_loader_set_ocult= data.DataLoader(val_set_ocult, batch_size=batchsize_x, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)




train_advtrain, val_advtrain =  get_combined_adv_dataset_from_id_array([7,8,9,15,20,21,22],5,0)
train_loader_advtrain = data.DataLoader(train_advtrain, batch_size=batchsize_x, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
val_loader_advtrain= data.DataLoader(val_advtrain, batch_size=batchsize_x, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)


################################################################################ Load cybernetics images
images_paths = get_tag_jobs(35)

cyber_images = []


for path in images_paths:
    cyber_images.append(get_image(path))

# Transform into tansors
cyber_images = [transform(img) for img in cyber_images]

# Call your data_augmentation function
cyber_images = data_augmentation(cyber_images, 9)

print("Cyber lenght : ",len(cyber_images))


# Create labels
label_value = 0
labels_cyber = [label_value] * len(cyber_images)

data_cyber = []
for image in cyber_images:
    data_cyber.append((image, 0))


cyber_images = data_cyber

print("cyber images 1 : ", cyber_images[0])

num_samples_cyber = len(cyber_images)
print("the number of samples in cyber ", num_samples_cyber)
train_size_cyber = int(0.8 * num_samples_cyber)
val_size_cyber= num_samples_cyber - train_size_cyber


train_set_cyber, val_set_cyber = random_split(cyber_images, [train_size_cyber, val_size_cyber])
train_loader_set_cyber = data.DataLoader(train_set_cyber, batch_size=batchsize_x, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
val_loader_set_cyber= data.DataLoader(val_set_cyber,batch_size=batchsize_x, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)





# ############ OOD
# oodset = SVHN(root='./data',  transform=transform, download=True)
# num_samples_ood = len(oodset)
# print("the number of ood samples is ", num_samples_ood)
# train_size_ood = int(0.8 * num_samples_ood)
# val_size_ood = num_samples_ood - train_size_ood

# train_set_ood, val_set_ood = random_split(oodset, [train_size_ood, val_size_ood])
# train_ood_loader = data.DataLoader(train_set_ood, batch_size=batchsize_x, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
# val_ood_loader = data.DataLoader(val_set_ood, batch_size=batchsize_x, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)



# ############### CIFAR
# cifarset = CIFAR10(root='./data',  transform=transform, download=True)
# num_samples_cifarset = len(cifarset)
# print("the number of ood samples is ", num_samples_cifarset)
# train_size_cifarset = int(0.8 * num_samples_cifarset)
# val_size_cifarset= num_samples_cifarset- train_size_cifarset

# train_set_cifarset, val_set_cifarset = random_split(cifarset, [train_size_cifarset, val_size_cifarset])
# train_cifarset_loader = data.DataLoader(train_set_cifarset, batch_size=batchsize_x, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
# val_cifarset_loader = data.DataLoader(val_set_cifarset, batch_size=batchsize_x, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)



train_loader = train_loader_set_ocult
val_loader = val_loader_set_ocult
adv_loader = val_loader_advtrain





########### Model Architecture

class MultiClassCNNModel(nn.Module):
    def __init__(self, input_channels=3, input_size=512, num_classes=2):
        super(MultiClassCNNModel, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the fully-connected layer input based on the architecture and input size
        fc_input_size = 64 * (input_size // 4) * (input_size // 4)

        # Change the output size to num_classes
        self.fc = nn.Linear(fc_input_size, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor before passing it to the fully-connected layer
        output = self.fc(x)  # Linear layer with num_classes outputs
        output = F.softmax(output, dim=1)  # Apply softmax activation along the second dimension
        return output


import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score




train_set_ocult, val_set_ocult
train_advtrain, val_advtrain 


combined_dataset_pos = torch.utils.data.ConcatDataset([train_set_ocult, val_set_ocult])
combined_dataset_neg = torch.utils.data.ConcatDataset([train_advtrain, val_advtrain])
combined_dataset = torch.utils.data.ConcatDataset([combined_dataset_pos, combined_dataset_neg])

print("dataset lenght ", len(combined_dataset))

# Split the combined dataset into training and testing sets
train_size = int(0.8 * len(combined_dataset))
test_size = len(combined_dataset) - train_size
train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])

# Create DataLoader for training and testing sets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)





# Initialize the binary classification model and move it to the GPU
model = MultiClassCNNModel().to(device)
# Iterate through model parameters
for name, param in model.named_parameters():
    print(f"Parameter: {name}, Size: {param.size()}, Type: {param.dtype}")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 15

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Print inputs and labels for debugging
        # print(f"Inputs: {images}")
        # print(f"Labels: {labels}")

        optimizer.zero_grad()
        outputs = model(images)

        # Print outputs for debugging
        print(f"Outputs: {outputs}")

        loss = criterion(outputs, labels.float().view(-1, 1))

        # Print loss for debugging
        print(f"Loss: {loss.item()}")

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Evaluation loop
model.eval()
all_labels = []
all_predictions = []
all_confidences = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predictions = (torch.sigmoid(outputs) > 0.5).float()  # Convert logits to probabilities and then to binary predictions
        confidence = torch.sigmoid(outputs).squeeze().cpu().numpy()  # Confidence scores
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_confidences.extend(confidence)

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_predictions)

# Print results
print(f"Accuracy: {accuracy * 100:.2f}%")