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
from kandinsky.models.clip_image_encoder.clip_image_encoder import KandinskyCLIPImageEncoder
import tempfile
import csv
import pandas as pd
from torch.utils.data import ConcatDataset
import argparse
from safetensors.torch import load_model, save_model

# ------------------------------------------------- Parameters BIS -------------------------------------------------
base_directory = "./"
sys.path.insert(0, base_directory)

from utility.path import separate_bucket_and_file_path
from data_loader.utils import get_object

API_URL = "http://192.168.3.1:8111"
minio_client = cmd.get_minio_client("D6ybtPLyUrca5IdZfCIM", "2LZ6pqIGOiZGcjPTR6DZPlElWBkRTkaLkyLIBt4V",None)

date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%d-%m-%Y %H:%M:%S')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")
    parser.add_argument('--class-id', type=int, help='id number of the class to train', default=35)
    parser.add_argument('--class-name', type=str, help='the name of the class to train or to load', default='cyber')
    parser.add_argument('--training-batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--num-samples', type=int, default=30000)
    parser.add_argument('--save-name', type=str, default='new-model')

    return parser.parse_args()


class EBM_Single_Class_Trainer:
    def __init__(self,
                
                model,
                class_name,
                minio_access_key,
                minio_secret_key,
                save_name,
                dataset,
                class_id,
                training_batch_size=64,
                num_samples=30000,
                learning_rate = 0.001,
                epochs=25):
        # get minio client
        self.minio_client = cmd.get_minio_client(minio_access_key=minio_access_key,
                                            minio_secret_key=minio_secret_key)
        # get device
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = torch.device(device)

        self.dataset= dataset
        self.save_name= save_name
        self.class_id= class_id
        self.num_samples= num_samples
        self.training_batch_size= training_batch_size
        self.model = DeepEnergyModel(adv_loader=None, img_shape=(1280,))
        self.classe_name = class_name
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = model

    def load_EBM_model(self):
        self.load_model_to_minio(self.model,self.classe_name)

    def train(self):
        
        all_tags = list(range(1, 51))
        print("all tag : ",all_tags)
        class_tag = self.class_id
        print("class tag : ",  class_tag)
        target_paths, adv_paths = self.get_all_tag_jobs(all_tags,class_tag)
        # Create dataloader of target class
        train_loader_automated, val_loader_automated = self.get_clip_embeddings_by_path(target_paths,1)

        # Create dataloader of adversarial classes
        train_loader_clip_ood, val_loader_clip_ood = self.get_clip_embeddings_by_path(adv_paths,0)
        # init the loader
        train_loader = train_loader_automated
        val_loader = val_loader_automated
        adv_loader = train_loader_clip_ood

        # Train
        model = train_model(img_shape=(1,1280),
                            batch_size=self.training_batch_size,
                            lr=self.learning_rate,
                            beta1=0.0,
                            train_loader = train_loader,
                            val_loader = val_loader,
                            adv_loader =adv_loader )
        
        self.save_model_to_minio(model,self.save_name,'temp_model.safetensors')


        # up loader graphs

        # # Plot

        # ############### Plot graph
        epochs = range(1, len(self.model.total_losses) + 1)  

        # Create subplots grid (3 rows, 1 column)
        fig, axes = plt.subplots(4, 1, figsize=(10, 24))

        # Plot each loss on its own subplot
        axes[0].plot(epochs, self.model.total_losses, label='Total Loss')
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Total Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(epochs, self.model.cdiv_losses, label='Contrastive Divergence Loss')
        axes[1].set_xlabel('Steps')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Contrastive Divergence Loss')
        axes[1].legend()
        axes[1].grid(True)


        axes[2].plot(epochs, self.model.reg_losses , label='Regression Loss')
        axes[2].set_xlabel('Steps')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Regression Loss')
        axes[2].legend()
        axes[2].grid(True)

        # Plot real and fake scores on the fourth subplot
        axes[3].plot(epochs, self.model.real_scores_s, label='Real Scores')
        axes[3].plot(epochs, self.model.fake_scores_s, label='Fake Scores')
        axes[3].set_xlabel('Steps')
        axes[3].set_ylabel('Score')  # Adjust label if scores represent a different metric
        axes[3].set_title('Real vs. Fake Scores')
        axes[3].legend()
        axes[3].grid(True)

        # Adjust spacing between subplots for better visualization
        plt.tight_layout()

        plt.savefig("output/loss_tracking_per_step.png")

        # Save the figure to a file
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # upload the graph report
        minio_path="environmental/output/my_tests"
        minio_path= minio_path + "/loss_tracking_per_step_1_cd_p2_regloss_isometric_training" +date_now+".png"
        cmd.upload_data(minio_client, 'datasets', minio_path, buf)
        # Remove the temporary file
        os.remove("output/loss_tracking_per_step.png")
        # Clear the current figure
        plt.clf()



    def get_all_tag_jobs(class_ids,target_id):
        all_data = {}  # Dictionary to store data for all class IDs
        
        for class_id in class_ids:
            response = requests.get(f'{API_URL}/tags/get-images-by-tag-id/?tag_id={class_id}')
            
            # Check if the response is successful (status code 200)
            if response.status_code == 200:
                try:
                    # Parse the JSON response
                    response_data = json.loads(response.content)

                    # Check if 'images' key is present in the JSON response
                    if 'images' in response_data.get('response', {}):
                        # Extract file paths from the 'images' key
                        file_paths = [job['file_path'] for job in response_data['response']['images']]
                        all_data[class_id] = file_paths
                    else:
                        print(f"Error: 'images' key not found in the JSON response for class ID {class_id}.")
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON for class ID {class_id}: {e}")
            else:
                print(f"Error: HTTP request failed with status code {response.status_code} for class ID {class_id}")
        

        # Separate data for a specific class ID (e.g., class_id = X) from all the rest
        target_class_data = all_data.get(target_id, [])
        rest_of_data = {class_id: data for class_id, data in all_data.items() if class_id != target_id}
        return target_class_data , rest_of_data

    def get_clip_embeddings_by_path(self,images_paths,label_value):
        ocult_clips = []
        ocult_clips = get_clip_vectors(images_paths)

        # Create labels
        data_occcult_clips = [(clip, label_value) for clip in ocult_clips]
        print("Clip embeddings array lenght : ",len(data_occcult_clips))

        # Split

        num_samples = len(data_occcult_clips)
        train_size = int(0.8 * num_samples)
        val_size = num_samples - train_size
        train_set, val_set = random_split(data_occcult_clips, [train_size, val_size])

        train_loader_clip = data.DataLoader(train_set, batch_size= self.training_batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        val_loader_clip = data.DataLoader(val_set, batch_size=self.training_batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)

        return train_loader_clip, val_loader_clip



    # ------------------------------------------------- Save Model --------------------------------------------------

    def save_model_to_minio(model,name,local_path):
            # Save the model locally pth
            save_model(model, local_path)
            
            #Read the contents of the saved model file
            with open(local_path, "rb") as model_file:
                model_bytes = model_file.read()

            # Upload the model to MinIO
            minio_client = cmd.get_minio_client("D6ybtPLyUrca5IdZfCIM", "2LZ6pqIGOiZGcjPTR6DZPlElWBkRTkaLkyLIBt4V",None)
            minio_path="environmental/output/my_tests"
            date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%d-%m-%Y %H:%M:%S')
            minio_path= minio_path + "/model-"+name+'_'+date_now+".safetensors"
            cmd.upload_data(minio_client, 'datasets', minio_path, BytesIO(model_bytes))
            print(f'Model saved to {minio_path}')


    # ------------------------------------------------- Load Model--------------------------------------------------
    def load_model_to_minio(model,type):
            # get model file data from MinIO
            prefix= "environmental/output/my_tests/model-"+type
            suffix= ".safetensors"
            # minio_client = cmd.get_minio_client("D6ybtPLyUrca5IdZfCIM", "2LZ6pqIGOiZGcjPTR6DZPlElWBkRTkaLkyLIBt4V",None)
            model_files=cmd.get_list_of_objects_with_prefix(minio_client, 'datasets', prefix)
            most_recent_model = None

            for model_file in model_files:
                print(model_file)
                if model_file.endswith(suffix):
                    print("yep found one",model_file)
                    most_recent_model = model_file

            if most_recent_model:
                model_file_data =cmd.get_file_from_minio(minio_client, 'datasets', most_recent_model)
                print("yep save : ",model_file)
            else:
                print("No .safetensors files found in the list.")
                return None
            
            print(most_recent_model)

            # Create a temporary file and write the downloaded content into it
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                for data in model_file_data.stream(amt=8192):
                    temp_file.write(data)

            # Load the model from the downloaded bytes
            #model.load_state_dict(torch.load(temp_file.name))
            load_model(model, temp_file.name)
            # Remove the temporary file
            os.remove(temp_file.name)
        




# ------------------------------------------------- Neural Net Architecutre --------------------------------------------------


class Clip_NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Clip_NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ------------------------------------------------- Deep Energy Based Model --------------------------------------------------
    
class DeepEnergyModel(pl.LightningModule):


    def __init__(self, img_shape,adv_loader, batch_size = 64, alpha=0.1, lr=1e-4, beta1=0.0, **CNN_args):
        super().__init__()
        self.save_hyperparameters()
        self.adv_loader = adv_loader
        self.cnn = Clip_NN(input_size = 1280, hidden_size = 512, output_size =1) 
        self.example_input_array = torch.zeros(1, *img_shape)
        self.total_losses = []
        self.class_losses = []
        self.cdiv_losses = []
        self.reg_losses = []
        self.real_scores_s = []
        self.fake_scores_s = []


    def forward(self, x):
        z = self.cnn(x)
        return z

    def configure_optimizers(self):
        # Energy models can have issues with momentum as the loss surfaces changes with its parameters.
        # Hence, we set it to 0 by default.
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97) # Exponential decay over epochs
        return [optimizer], [scheduler]

    def training_step(self, batch): #maybe add the adv loader
        # We add minimal noise to the original images to prevent the model from focusing on purely "clean" inputs
        # get device
    
        if torch.cuda.is_available():
            device_name = 'cuda'
        else:
            device_name = 'cpu'
            device = torch.device(device_name)

        real_imgs, _ = batch
        #print("the _ is ",_)
        small_noise = torch.randn_like(real_imgs) * 0.005
        real_imgs.add_(small_noise).clamp_(min=-1.0, max=1.0)

        # Obtain samples #Give more steps later
        fake_imgs, fake_labels = next(iter(self.adv_loader))
        fake_imgs = fake_imgs.to(device)
        fake_labels = fake_labels.to(device)

        _.to(device)
        all_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        all_scores = self.cnn(all_imgs)

        # Separate real and fake scores and probabilities
        real_scores, fake_scores = all_scores.chunk(2, dim=0)


        # Calculate CD loss
        cdiv_loss = fake_scores.mean() - real_scores.mean()

        # regression loss
        reg_loss =(real_scores ** 2 + fake_scores ** 2).mean()

        # Combine losses and backpropagate
        alphaW = 1  # Adjust weight for cdiv_loss
        alphaY = 0.8  # Adjust weight for reg_loss
        total_loss =  ((alphaW) * cdiv_loss) + (alphaY * reg_loss)


        # Logging
        self.log('total loss', total_loss)
        self.log('loss_contrastive_divergence', cdiv_loss)
        self.log('metrics_avg_real', 0)
        self.log('metrics_avg_fake', 0)


        self.total_losses.append(total_loss.item())
        self.cdiv_losses.append(cdiv_loss.item())
        self.reg_losses.append(reg_loss.item())

        self.real_scores_s.append(real_scores.mean().item())
        self.fake_scores_s.append(fake_scores.mean().item())
        return total_loss
    
    def validation_step(self, batch):

      # Validate with real images only (no noise/fakes)
      real_imgs, labels = batch

      # Pass through model to get scores and probabilities
      all_scores = self.cnn(real_imgs)

      # Calculate CD loss (optional, adjust if needed)
      cdiv = all_scores.mean()  # Modify based on scores or probabilities

      # Log metrics
      self.log('val_contrastive_divergence', cdiv)






def train_model(train_loader,val_loader, adv_loader, **kwargs):


    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=20,
                         gradient_clip_val=0.1,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor='val_contrastive_divergence'),
                                    LearningRateMonitor("epoch")
                                   ])

    pl.seed_everything(42)
    model = DeepEnergyModel(adv_loader =adv_loader ,**kwargs)
    trainer.fit(model, train_loader, val_loader)

    return model



# From multiples image paths
def get_clip_vectors(file_paths):


    clip_vectors = []

    for path in file_paths:

        try:
            print("path : " , path)
            clip_path = path.replace(".jpg", "_clip_kandinsky.msgpack")
            bucket, features_vector_path = separate_bucket_and_file_path(clip_path)

            features_data = get_object(minio_client, features_vector_path)
            features = msgpack.unpackb(features_data)["clip-feature-vector"]
            features = torch.tensor(features)
            clip_vectors.append(features)
        except Exception as e:
            # Handle the specific exception (e.g., FileNotFoundError, ConnectionError) or a general exception.
            print(f"Error processing clip at path {path}: {e}")
            # You might want to log the error for further analysis or take alternative actions.

    return clip_vectors



def main():
    args = parse_args()

    training_pipeline=EBM_Single_Class_Trainer(minio_access_key=args.minio_access_key,
                                minio_secret_key=args.minio_secret_key,
                                dataset= args.dataset,
                                class_name= args.class_name,
                                model = None,
                                save_name = args.save_name,
                                class_id = args.class_id,
                                training_batch_size=args.training_batch_size,
                                num_samples= args.num_samples,
                                epochs= args.epochs,
                                learning_rate= args.learning_rate)

    # do self training
    training_pipeline.train()
    

if __name__ == "__main__":
    main()
