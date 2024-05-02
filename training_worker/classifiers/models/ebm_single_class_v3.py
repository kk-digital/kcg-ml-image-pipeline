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
from training_worker.classifiers.models.reports.get_model_card import get_model_card_buf
from os.path import basename



API_URL = "http://192.168.3.1:8111"
date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%d-%m-%Y %H:%M:%S')

minio_client = cmd.get_minio_client("D6ybtPLyUrca5IdZfCIM",
            "2LZ6pqIGOiZGcjPTR6DZPlElWBkRTkaLkyLIBt4V",
            None)

from utility.path import separate_bucket_and_file_path
from data_loader.utils import get_object
from utility.http import request


def remove_duplicates(list_a, list_b):
    # Convert lists to sets to remove duplicates
    # set_a = set(list_a)
    # set_b = set(list_b)


    result = []
    # Remove elements from set_b that are in set_a
    # set_b -= set_a

    for element_b in list_b:
        exist = False   
        for element_a in list_a:
            
            if element_b == element_a:
             print(f"{list_b} exist") 
             exist = True
             break
        if exist == False:
            result.append(element_b)

    # Convert sets back to lists
    # unique_list_a = list(set_a)
    # unique_list_b = list(set_b)

    print(f'before {len(list_b)}, after {len(result)}')

    return result

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

image_embedder= KandinskyCLIPImageEncoder(device="cuda")
image_embedder.load_submodels()



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

def get_clip_embeddings_by_tag(id_classes,label_value):
    images_paths = get_tag_jobs(id_classes[0])
    i = 1
    for i in range(1,len(id_classes)):
        images_paths = images_paths + get_tag_jobs(id_classes[i])
       
 
    
    ocult_clips = get_clip_vectors(images_paths)


    # Create labels
    data_occcult_clips = [(clip, label_value) for clip in ocult_clips]
    print("Clip embeddings array lenght : ",len(data_occcult_clips))

    # Split

    num_samples = len(data_occcult_clips)
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    train_set, val_set = random_split(data_occcult_clips, [train_size, val_size])

    train_loader_clip = data.DataLoader(train_set, batch_size=16, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader_clip = data.DataLoader(val_set, batch_size=16, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)

    return train_loader_clip, val_loader_clip


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
        # optional
        #x = torch.tanh(x) 
        return x


# ------------------------------------------------- EBM Standard Class --------------------------------------------------

    
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
    
        real_imgs, _ = batch
        #print("the _ is ",_)
        small_noise = torch.randn_like(real_imgs) * 0.005
        real_imgs.add_(small_noise).clamp_(min=-1.0, max=1.0)

        # Obtain samples #Give more steps later
        fake_imgs, fake_labels = next(iter(self.adv_loader))
        fake_imgs = fake_imgs.to(self.device)
        fake_labels = fake_labels.to(self.device)

        _.to(self.device)
        all_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        all_scores = self.cnn(all_imgs)

        # Separate real and fake scores and probabilities
        real_scores, fake_scores = all_scores.chunk(2, dim=0)


        # Calculate CD loss
        cdiv_loss = fake_scores.mean() - real_scores.mean()

        # regression loss
        #reg_loss =(real_scores ** 2 + fake_scores ** 2).mean()
        reg_loss =((real_scores + fake_scores )** 2) .mean()

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



# ------------------------------------------------- EBM Single Class --------------------------------------------------


class EBM_Single_Class:
    def __init__(self,
                
                model,
                class_name,
                minio_access_key,
                minio_secret_key,
                save_name,
                dataset,
                class_id,
                # train_loader,
                # val_loader,
                # adv_loader,
                training_batch_size=16,
                num_samples=30000,
                learning_rate = 0.001,
                epochs=25,


                ):
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
        self.train_loader = None
        self.val_loader = None
        self.adv_loader = None

    def train(self,**kwargs):

    
        trainer = pl.Trainer(
                            accelerator="gpu" if str(self.device).startswith("cuda") else "cpu",
                            devices=1,
                            max_epochs=self.epochs,
                            gradient_clip_val=0.1,
                            callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor='val_contrastive_divergence'),
                                        LearningRateMonitor("epoch")
                                    ])

        pl.seed_everything(42)
        self.model = DeepEnergyModel(adv_loader =self.adv_loader,img_shape=(1280,) ,**kwargs)
        
        trainer.fit(self.model , self.train_loader, self.val_loader)        
        self.save_model_to_minio(self.save_name,'temp_model.safetensors')

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
        minio_path= minio_path + "/loss_tracking_per_step_1_cd_p2_regloss_"+ self.classe_name + "_" +date_now+".png"
        cmd.upload_data(minio_client, 'datasets', minio_path, buf)
        # Remove the temporary file
        os.remove("output/loss_tracking_per_step.png")
        # Clear the current figure
        plt.clf()



    def train_v2(self,**kwargs):


        ##################### Standard method ##########################
        print("class name ", self.classe_name)
        all_tags = get_unique_tag_ids()
        print("all tag : ",all_tags)
        class_tag = get_tag_id_by_name(self.classe_name)
        print("class tag : ",  class_tag)
        target_paths, adv_paths = get_all_tag_jobs(class_ids = all_tags, target_id =class_tag) 


        # new addition same input size
        # min_data_size = min(len(target_paths), len(adv_paths))
        # print(f"the minimum is {min}, of  target: {len(target_paths)} and adv: {len(adv_paths)} ")
        # target_paths = random.sample(target_paths, min_data_size)
        # adv_paths = random.sample(adv_paths, min_data_size)
        # new addition same input size



        # minimize dataset

        target_paths = random.sample(target_paths,   min(len(target_paths),800))
        
        # minimize dataset


        print("target_paths lenght : ", len(target_paths))
        print("adv_paths lenght : ", len(adv_paths))
        
        # for path in target_paths:
        #     print(" Path t :", path)
        # for path in adv_paths:
        #     print(" Path adv :", path)
        #Create dataloader of target class
        train_loader_automated, val_loader_automated = get_clip_embeddings_by_path(target_paths,1)

        # Create dataloader of adversarial classes
        train_loader_clip_ood, val_loader_clip_ood = get_clip_embeddings_by_path(adv_paths,0)
        ##################### Standard method ##########################


        ##################### OLD method ##########################
        # # Create dataloader of target class
        # train_loader_automated, val_loader_automated = get_clip_embeddings_by_path(get_tag_jobs(get_tag_id_by_name(self.classe_name)),1)

        # # Create dataloader of adversarial classes
        # train_loader_clip_ood, val_loader_clip_ood = get_clip_embeddings_by_tag([3,5,7,8,9,15,20,21,34,39],0)
        ##################### OLD method ##########################



        # init the loader
        train_loader = train_loader_automated
        val_loader = val_loader_automated
        adv_loader = train_loader_clip_ood
    
        trainer = pl.Trainer(
                            accelerator="gpu" if str(self.device).startswith("cuda") else "cpu",
                            devices=1,
                            max_epochs=self.epochs,
                            gradient_clip_val=0.1,
                            callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor='val_contrastive_divergence'),
                                        LearningRateMonitor("epoch")
                                    ])

        pl.seed_everything(42)
        self.model = DeepEnergyModel(adv_loader =adv_loader,img_shape=(1280,) ,**kwargs)
        
        trainer.fit(self.model , train_loader, val_loader)        
        self.save_model_to_minio(self.save_name,'temp_model.safetensors')

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
        minio_path= minio_path + "/loss_tracking_per_step_1_cd_p2_regloss_"+ self.classe_name + "_" +date_now+".png"
        cmd.upload_data(minio_client, 'datasets', minio_path, buf)
        # Remove the temporary file
        os.remove("output/loss_tracking_per_step.png")
        # Clear the current figure
        plt.clf()
   
   
   





    def train_v3(self,**kwargs):


        ##################### Standard method ##########################
        print("class name ", self.classe_name)
        all_tags = get_unique_tag_ids()
        print("all tag : ",all_tags)
        class_tag = get_tag_id_by_name(self.classe_name)
        print("class tag : ",  class_tag)
        target_paths, adv_paths = get_all_tag_jobs(class_ids = all_tags, target_id =class_tag) 



        adv_paths = remove_duplicates(target_paths,adv_paths)

        # new addition same input size
        # min_data_size = min(len(target_paths), len(adv_paths))
        # print(f"the minimum is {min}, of  target: {len(target_paths)} and adv: {len(adv_paths)} ")
        # target_paths = random.sample(target_paths, min_data_size)
        # adv_paths = random.sample(adv_paths, min_data_size)
        # new addition same input size



        # minimize dataset

        target_paths = random.sample(target_paths,   min(len(target_paths),800))
        
        # minimize dataset


        print("target_paths lenght : ", len(target_paths))
        print("adv_paths lenght : ", len(adv_paths))
        
        # for path in target_paths:
        #     print(" Path t :", path)
        # for path in adv_paths:
        #     print(" Path adv :", path)
        #Create dataloader of target class
        train_loader_automated, val_loader_automated = get_clip_embeddings_by_path(target_paths,1)

        # Create dataloader of adversarial classes
        train_loader_clip_ood, val_loader_clip_ood = get_clip_embeddings_by_path(adv_paths,0)
        ##################### Standard method ##########################


        ##################### OLD method ##########################
        # # Create dataloader of target class
        # train_loader_automated, val_loader_automated = get_clip_embeddings_by_path(get_tag_jobs(get_tag_id_by_name(self.classe_name)),1)

        # # Create dataloader of adversarial classes
        # train_loader_clip_ood, val_loader_clip_ood = get_clip_embeddings_by_tag([3,5,7,8,9,15,20,21,34,39],0)
        ##################### OLD method ##########################



        # init the loader
        train_loader = train_loader_automated
        val_loader = val_loader_automated
        adv_loader = train_loader_clip_ood
    
        trainer = pl.Trainer(
                            accelerator="gpu" if str(self.device).startswith("cuda") else "cpu",
                            devices=1,
                            max_epochs=self.epochs,
                            gradient_clip_val=0.1,
                            callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor='val_contrastive_divergence'),
                                        LearningRateMonitor("epoch")
                                    ])

        pl.seed_everything(42)
        self.model = DeepEnergyModel(adv_loader =adv_loader,img_shape=(1280,) ,**kwargs)
        
        trainer.fit(self.model , train_loader, val_loader)        
        self.save_model_to_minio(self.save_name,'temp_model.safetensors')

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
        minio_path= minio_path + "/loss_tracking_per_step_1_cd_p2_regloss_"+ self.classe_name + "_" +date_now+".png"
        cmd.upload_data(minio_client, 'datasets', minio_path, buf)
        # Remove the temporary file
        os.remove("output/loss_tracking_per_step.png")
        # Clear the current figure
        plt.clf()
   
   

    def save_model_to_minio(self,name,local_path):
            # Save the model locally pth
            save_model(self.model, local_path)
            
            #Read the contents of the saved model file
            with open(local_path, "rb") as model_file:
                model_bytes = model_file.read()

            # init config
                
            date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%Y-%m-%d')
            print("Current datetime: {}".format(datetime.now(tz=timezone("Asia/Hong_Kong"))))
            bucket_name = "datasets"
            network_type = "energy-based-model"
            output_type = "energy"
            input_type = 'clip-h'
            dataset_name = 'environmental'
            tag_name = self.classe_name

            output_path = "{}/models/classifiers/{}".format(dataset_name, tag_name)
            sequence = 0
            filename = "{}-{:02}-{}-{}-{}-{}".format(date_now, sequence, tag_name, output_type, network_type, input_type)

            # if exist, increment sequence
            while True:
                filename = "{}-{:02}-{}-{}-{}-{}".format(date_now, sequence, tag_name, output_type, network_type, input_type)
                exists = cmd.is_object_exists(minio_client, bucket_name,
                                            os.path.join(output_path, filename + ".safetensors"))
                if not exists:
                    break

                sequence += 1

            model_name = "{}.safetensors".format(filename)
            model_output_path = os.path.join(output_path, model_name)
            print("file path : ",filename)
            # upload model
            
            minio_model_path = output_path + '/' + filename +".safetensors"
            print("minio model path ",minio_model_path)
            cmd.upload_data(minio_client, bucket_name, minio_model_path, BytesIO(model_bytes))
            # Upload the model to MinIO

            cmd.is_object_exists(minio_client, bucket_name,
                                      os.path.join(output_path, filename + ".safetensors"))
            
          
            # get model card and upload
            classifier_name="{}-{}-{}-{}".format(self.classe_name, output_type, network_type, input_type)
            model_card_name = "{}.json".format(filename)
            model_card_name_output_path = os.path.join(output_path, model_card_name)
            model_card_buf, model_card = get_model_card_buf(classifier_name= classifier_name,
                                                            tag_id= self.class_id,
                                                            latest_model= filename,
                                                            model_path= model_output_path,
                                                            creation_time=date_now)
            cmd.upload_data(minio_client, bucket_name, model_card_name_output_path, model_card_buf)

            # add model card
            request.http_add_classifier_model(model_card)      
    def load_model_from_minio(self, minio_client, dataset_name, tag_name, model_type):
            # get model file data from MinIO
            #datasets/environmental/models/classifiers/concept-cybernetic
            prefix= f"{dataset_name}/models/classifiers/"
            suffix= ".safetensors"

            model_files=cmd.get_list_of_objects_with_prefix(minio_client, 'datasets', prefix)
            most_recent_model = None

            for model_file in model_files:
                #print("model path : ",model_file)
                if tag_name in model_file and model_type in model_file and model_file.endswith(suffix):
                    #print("yep found one",model_file)
                    most_recent_model = model_file

            print("most recent model is : ",  most_recent_model)
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
            self.model = DeepEnergyModel(train_loader = None,val_loader = None, adv_loader = None,img_shape=(1280,))
            load_model(self.model, temp_file.name)
            # Remove the temporary file
            os.remove(temp_file.name)

    # Via clip-H
    def classify(self, dataset_feature_vector):
        print("Evaluate energy...")
        #print("da vector ", dataset_feature_vector)
        #dataset_feature_vector = dataset_feature_vector.to(self._device)
        energy = self.model.cnn(dataset_feature_vector.unsqueeze(0).to(self.model.device)).cpu()
        return energy
    



# Uitilities #################################
    


def get_tag_id_by_name(tag_name):
    response = requests.get(f'{API_URL}/tags/get-tag-id-by-tag-name?tag_string={tag_name}')
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        json_data = response.json()

        # Get the value of "response" from the JSON data
        response_value = json_data.get('response')
        tag_id = response_value.get('tag_id')
        # Print or use the response value
        #print("The tag id is:", response_value, " the tag id is : ",tag_id )
        return tag_id
    else:
        print("Error:", response.status_code)


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
    

    # # Separate data for a specific class ID (e.g., class_id = X) from all the rest
    # target_class_data = all_data.get(target_id, [])
    # rest_of_data = {class_id: data for class_id, data in all_data.items() if class_id != target_id}
    # #return target_class_data , rest_of_data


    # Separate data for a specific class ID (e.g., class_id = X) from all the rest
    target_class_data = all_data.get(target_id, [])
    rest_of_data = [path for class_id, paths in all_data.items() if class_id != target_id for path in paths]

    return target_class_data, rest_of_data


def get_all_classes_paths(class_ids,target_id):
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
    

    # # Separate data for a specific class ID (e.g., class_id = X) from all the rest
    # target_class_data = all_data.get(target_id, [])
    # rest_of_data = {class_id: data for class_id, data in all_data.items() if class_id != target_id}
    # #return target_class_data , rest_of_data

    print("the full data ",  len(all_data.items()))
    # Separate data for a specific class ID (e.g., class_id = X) from all the rest
    target_class_data = all_data.get(target_id, [])
    rest_of_data = [path for class_id, paths in all_data.items() if class_id != target_id for path in paths]

    return target_class_data, rest_of_data




# Get train and validation loader from images paths and the label value
def get_clip_embeddings_by_path(images_paths,label_value):
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

    train_loader_clip = data.DataLoader(train_set, batch_size=16, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader_clip = data.DataLoader(val_set, batch_size=16, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)

    return train_loader_clip, val_loader_clip

# From multiples image paths
def get_clip_vectors(file_paths):
    clip_vectors = []

    for path in file_paths:

        try:
            #print("path : " , path)
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


def get_clip_and_image_from_path(image_path):
    image=get_image(image_path)
    clip_embedding =  image_embedder.get_image_features(image)
    #clip_embedding = torch.tensor(clip_embedding)
    return image,clip_embedding.float()

def get_clip_from_path(image_path):
    image=get_image(image_path)
    clip_embedding =  image_embedder.get_image_features(image)
    #clip_embedding = torch.tensor(clip_embedding)
    return clip_embedding.float()

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


def get_unique_tag_ids():
    response = requests.get(f'{API_URL}/tags/list-tag-definitions')
    if response.status_code == 200:
        try:
            json_data = response.json()
            tags = json_data.get('response', {}).get('tags', [])

            # Extract unique tag IDs
            unique_tag_ids = set(tag.get('tag_id') for tag in tags)
            
            # Convert the set of unique tag IDs to a list
            unique_tag_ids_list = list(unique_tag_ids)
            
            return unique_tag_ids_list
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    else:
        print(f"Error: HTTP request failed with status code {response.status_code}")




def get_unique_tag_names():
    response = requests.get(f'{API_URL}/tags/list-tag-definitions')
    if response.status_code == 200:
        try:
            json_data = response.json()
            tags = json_data.get('response', {}).get('tags', [])

            # Extract unique tag IDs
            unique_tag_name = set(tag.get('tag_string') for tag in tags)
            
            # Convert the set of unique tag IDs to a list
            unique_tag_name_list = list(unique_tag_name)
            
            return unique_tag_name_list
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    else:
        print(f"Error: HTTP request failed with status code {response.status_code}")





def load_model_elm(minio_client, model_dataset, tag_name, model_type, scoring_model, not_include, device=None):
    input_path = f"{model_dataset}/models/classifiers/{tag_name}/"
    file_suffix = ".safetensors"

    # Use the MinIO client's list_objects method directly with recursive=True
    model_files = [obj.object_name for obj in minio_client.list_objects('datasets', prefix=input_path, recursive=True) if obj.object_name.endswith(file_suffix) and model_type in obj.object_name and scoring_model in obj.object_name and not_include not in obj.object_name ]
    
    if not model_files:
        print(f"No .safetensors models found for tag: {tag_name}")
        return None

    # Assuming there's only one model per tag or choosing the first one
    model_files.sort(reverse=True)
    model_file = model_files[0]
    print(f"Loading model: {model_file}")

    return load_model_with_filename(minio_client, model_file, tag_name, device)




def load_model_with_filename(minio_client, model_file, device, model_info=None):
    model_data = minio_client.get_object('datasets', model_file)
    
    clip_model = ELMRegression(device= device )
    
    # Create a BytesIO object from the model data
    byte_buffer = BytesIO(model_data.data)
    clip_model.load_safetensors(byte_buffer)

    print(f"Model loaded for tag: {model_info}")
    
    return clip_model, basename(model_file)


###################### main

def main():
    args = parse_args()
    class_names = get_unique_tag_names()
    all_tags = get_unique_tag_ids()
    print("all tags : ", all_tags )
    print("all tags length : ", len(all_tags) )
    # all_data,_ = get_all_classes_paths(class_ids = all_tags,target_id=1)
    # print(all_data)
    # print("all data length  : ", len(all_data) )



    ##################### Basic method ##########################

    for class_name in class_names:
        
        class_tag = get_tag_id_by_name(class_name)
        print("Initiating training of : ", class_name)

        target_paths, adv_paths = get_all_tag_jobs(class_ids = all_tags, target_id =class_tag)



        if len(target_paths) >= 16: 
            
            # train_loader_automated, val_loader_automated = get_clip_embeddings_by_path(target_paths,1)
            # # Create dataloader of adversarial classes
            # train_loader_clip_ood, val_loader_clip_ood = get_clip_embeddings_by_path(adv_paths,0)
            print("Training starated for  ", class_name," with ",len(target_paths)," data points.")

            training_pipeline=EBM_Single_Class(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        dataset= args.dataset,
                                        class_name= class_name,
                                        model = None,
                                        save_name = args.save_name,
                                        class_id =  get_tag_id_by_name(args.class_name),
                                        training_batch_size=args.training_batch_size,
                                        num_samples= args.num_samples,
                                        epochs= args.epochs,
                                        learning_rate= args.learning_rate)

            # do self training
            training_pipeline.train_v2()
        else:
            print("There isn't enough data for : ", class_name)


    ##################### Standard method ##########################

    # for class_name in class_names:
        
    #     class_tag = get_tag_id_by_name(class_name)
    #     print("Initiating training of : ", class_name)

    #     target_paths = all_data.get(class_tag, [])
    #     adv_paths = [path for class_id, paths in all_data.items() if class_id != class_tag for path in paths]

    #     train_loader_automated, val_loader_automated = get_clip_embeddings_by_path(target_paths,1)
    #     # Create dataloader of adversarial classes
    #     train_loader_clip_ood, val_loader_clip_ood = get_clip_embeddings_by_path(adv_paths,0)


    #     if len(train_loader_automated) != 0: 
    #         print("Training starated for  ", class_name," with ",len(train_loader_automated)," data points.")
    #         training_pipeline=EBM_Single_Class(minio_access_key=args.minio_access_key,
    #                                     minio_secret_key=args.minio_secret_key,
    #                                     dataset= args.dataset,
    #                                     class_name= args.class_name,
    #                                     model = None,
    #                                     save_name = args.save_name,
    #                                     class_id =  get_tag_id_by_name(args.class_name),
    #                                     training_batch_size=args.training_batch_size,
    #                                     num_samples= args.num_samples,
    #                                     epochs= args.epochs,
    #                                     learning_rate= args.learning_rate,
    #                                     train_loader = train_loader_automated,
    #                                     val_loader  = val_loader_automated,
    #                                     adv_loader = train_loader_clip_ood)
    #         # do self training
    #         training_pipeline.train()
    #     else:
    #         print("There isn't enough data for : ", class_name)

        # # init the loader
        # train_loader = train_loader_automated
        # val_loader = val_loader_automated
        # adv_loader = train_loader_clip_ood

    # training_pipeline.load_model_from_minio(minio_client, dataset_name = "environmental", tag_name ="topic-space" , model_type = "energy-based-model")
    # #datasets/test-generations/0024/023128.jpg

    # print("defect-split-pane-image 1 : , ",training_pipeline.evalute_energy(get_clip_from_path('datasets/environmental/0241/240499.jpg')).item())
    # print("defect-split-pane-image 2: , ",training_pipeline.evalute_energy(get_clip_from_path('datasets/environmental/0188/187132.jpg')).item())
    # print("defect-split-pane-image 3: , ",training_pipeline.evalute_energy(get_clip_from_path('datasets/environmental/0058/057942.jpg')).item())


    # print("Cyber image 1 : , ",training_pipeline.evalute_energy(get_clip_from_path('datasets/test-generations/0024/023123.jpg')).item())
    # print("Cyber image 2 : , ",training_pipeline.evalute_energy(get_clip_from_path('datasets/environmental/0208/207925.jpg')).item())
    # print("Cyber image 3 : , ",training_pipeline.evalute_energy(get_clip_from_path('datasets/environmental/0330/329625.jpg')).item())

    # print("occult image 1 : , ",training_pipeline.evalute_energy(get_clip_from_path('datasets/test-generations/0024/023128.jpg')).item())
    # print("occult image 2 : , ",training_pipeline.evalute_energy(get_clip_from_path('datasets/environmental/0124/123017.jpg')).item())
    # print("occult image 3 : , ",training_pipeline.evalute_energy(get_clip_from_path('datasets/environmental/0367/366210.jpg')).item())

    # print("Aquatic image 1 : , ",training_pipeline.evalute_energy(get_clip_from_path('datasets/environmental/0300/299693.jpg')).item())
    # print("Aquatic image 2  : , ",training_pipeline.evalute_energy(get_clip_from_path('datasets/environmental/0042/041848.jpg')).item())
    # print("Aquatic image 3  : , ",training_pipeline.evalute_energy(get_clip_from_path('datasets/environmental/0277/276058.jpg')).item())

# if __name__ == "__main__":
#     main()



# Get all the tags from all the classes:
    


# ELM VS EBM
# 2024-02-29-00-topic-aquatic-score-elm-regression-clip-h.safetensors






###########################################################################################################################
##################################################  ELM VS EBM     ########################################################
###########################################################################################################################

# idi = ['datasets/environmental/0042/041848.jpg','datasets/environmental/0263/262253.jpg','datasets/environmental/0056/055126.jpg']
# ood = ['datasets/environmental/0058/057516.jpg','datasets/environmental/0214/213301.jpg','datasets/environmental/0063/062805.jpg']



# target_image = 'datasets/environmental/0214/213301.jpg' # idi[0]

# _, clip_h_vector = get_clip_and_image_from_path(idi[0])

# print(clip_h_vector)



# #EBM
# args = parse_args()
# original_model=EBM_Single_Class(minio_access_key=args.minio_access_key,
#                             minio_secret_key=args.minio_secret_key,
#                             dataset= args.dataset,
#                             class_name= "topic-aquatic" ,
#                             model = None,
#                             save_name = args.save_name,
#                             class_id =  get_tag_id_by_name(args.class_name),
#                             training_batch_size=args.training_batch_size,
#                             num_samples= args.num_samples,
#                             epochs= args.epochs,
#                             learning_rate= args.learning_rate)

# #original_model = EBM_Single_Class(train_loader = None,val_loader = None, adv_loader = None,img_shape=(1280,))
# # Load the last occult trained model
# original_model.load_model_from_minio(minio_client, dataset_name = "environmental", tag_name ="topic-aquatic" , model_type = "energy-based-model")
# #score = original_model.cnn(clip_h_vector.unsqueeze(0).to(original_model.device)).cpu()
# score = original_model.evalute_energy(clip_h_vector)


# # ELM
# from training_worker.classifiers.models.elm_regression import ELMRegression
# # elm_model = ELMRegression()
# #def load_model(self, minio_client, model_dataset, tag_name, model_type, scoring_model, not_include, device=None):

# elm_model, _ = load_model_elm(device = original_model.device, minio_client = minio_client, model_dataset = "environmental",scoring_model = 'score' ,tag_name = "topic-aquatic", model_type = "elm-regression-clip-h", not_include= 'batatatatatata')



# print("the EBM score is : ",score.item())

# print("the ELM score is : ", (elm_model.classify(clip_h_vector)).item())

import statistics


# class_names = get_unique_tag_names()
# all_tags = get_unique_tag_ids()
# print("all tags : ", all_tags )
# print("all tags length : ", len(all_tags) )
# target_data , ood_data = get_all_classes_paths(class_ids = all_tags,target_id=35)


# target_scores_EBM = []
# ood_scores_EBM = []

# target_scores_ELM = []
# ood_scores_ELM = []

# for target in target_data:
#     vector = get_clip_from_path(target)
#     ebm_score = (original_model.evalute_energy(vector).item())
#     elm_score = (elm_model.classify(vector)).item()
#     target_scores_EBM.append(ebm_score)
#     target_scores_ELM.append(elm_score)
#     print(f'The score for EBM is {ebm_score} , and the score of ELM is {elm_score}')


# for ood_image in ood_data:
#     vector = get_clip_from_path(ood_image)
#     ebm_score = (original_model.evalute_energy(vector).item())
#     elm_score = (elm_model.classify(vector)).item()
#     ood_scores_EBM.append(ebm_score)
#     ood_scores_ELM.append(elm_score)
#     print(f'The score for EBM is {ebm_score} , and the score of ELM is {elm_score}')



# print(f'Average EBM score is {statistics.mean(target_scores_EBM)} , and average ELM score is {statistics.mean(target_scores_ELM)}')
# print(f'Standard diviation EBM: {statistics.stdev(target_scores_EBM)} , ELM {statistics.stdev(target_scores_ELM)}')


# print(f'Average OOD EBM score is {statistics.mean(ood_scores_EBM)} , and average OOD ELM score is {statistics.mean(ood_scores_ELM)}')
# print(f'Standard diviation OOD EBM: {statistics.stdev(ood_scores_EBM)} , ELM {statistics.stdev(ood_scores_ELM)}')


###########################################################################################################################
##################################################  ELM VS EBM     ########################################################
###########################################################################################################################


def plot_images_with_scores_hasheless(sorted_dataset,name):
    minio_client = cmd.get_minio_client("D6ybtPLyUrca5IdZfCIM",
            "2LZ6pqIGOiZGcjPTR6DZPlElWBkRTkaLkyLIBt4V",
            None)
    # Number of images
    num_images = len(sorted_dataset)
    
    # Fixed columns to 4
    cols = 4
    # Calculate rows needed for 4 images per row
    rows = math.ceil(num_images / cols)

    # Create figure with subplots
    # Adjust figsize here: width, height in inches. Increase for larger images.
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))  # 4 inches per image in each dimension
    fig.tight_layout(pad=3.0)  # Adjust padding as needed
    # Flatten axes array for easy indexing
    axes = axes.flatten()

    # Loop over sorted dataset and plot each image with its score
    for i, (image_path, _, score, image_tensor) in enumerate(sorted_dataset):
        # Check if image_tensor is a PIL Image; no need to convert if already a numpy array
        if not isinstance(image_tensor, np.ndarray):
            # Convert PIL Image to a format suitable for matplotlib
            image = np.array(image_tensor)
        
        # Plot the image
        axes[i].imshow(image)
        axes[i].set_title(f"Score: {score:.2f}")
        axes[i].axis('off')  # Hide axis ticks and labels

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.savefig("output/rank.png")

    # Save the figure to a file
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # upload the graph report
    minio_path="environmental/output/my_tests"
    minio_path= minio_path + "/ranking_ds_"+ name + '_' +date_now+".png"
    cmd.upload_data(minio_client, 'datasets', minio_path, buf)
    # Remove the temporary file
    os.remove("output/rank.png")
    # Clear the current figure
    plt.clf()




# Try it for dictionary
def plot_images_with_scores_hasheless_v2(sorted_dataset,name):
    minio_client = cmd.get_minio_client("D6ybtPLyUrca5IdZfCIM",
            "2LZ6pqIGOiZGcjPTR6DZPlElWBkRTkaLkyLIBt4V",
            None)
    # Number of images
    num_images = len(sorted_dataset)
    
    # Fixed columns to 4
    cols = 4
    # Calculate rows needed for 4 images per row
    rows = math.ceil(num_images / cols)

    # Create figure with subplots
    # Adjust figsize here: width, height in inches. Increase for larger images.
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))  # 4 inches per image in each dimension
    fig.tight_layout(pad=3.0)  # Adjust padding as needed
    # Flatten axes array for easy indexing
    axes = axes.flatten()

    # Loop over sorted dataset and plot each image with its score
    i = 0
    for element in sorted_dataset:
        if not isinstance(element["image_tensor"], np.ndarray):
            # Convert PIL Image to a format suitable for matplotlib
            image = np.array(element["image_tensor"])
            score = element["score"]
        

    # for i, (image_path, _, score, image_tensor) in enumerate(sorted_dataset):
    #     # Check if image_tensor is a PIL Image; no need to convert if already a numpy array
    #     if not isinstance(image_tensor, np.ndarray):
    #         # Convert PIL Image to a format suitable for matplotlib
    #         image = np.array(image_tensor)
        
        # Plot the image
        axes[i].imshow(image)
        axes[i].set_title(f"Score: {score:.2f}")
        axes[i].axis('off')  # Hide axis ticks and labels
        i = i+1
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.savefig("output/rank.png")

    # Save the figure to a file
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # upload the graph report
    minio_path="environmental/output/my_tests"
    minio_path= minio_path + "/ranking_ds_"+ name + '_' +date_now+".png"
    cmd.upload_data(minio_client, 'datasets', minio_path, buf)
    # Remove the temporary file
    os.remove("output/rank.png")
    # Clear the current figure
    plt.clf()

def process_and_sort_dataset(images_paths, model):
    # Initialize an empty list to hold the structure for each image
    structure = []

    # Process each image path
    for image_path in images_paths:
        # Extract embedding and image tensor from the image path
        image, embedding = get_clip_and_image_from_path(image_path)
        
        # Compute the score by passing the image tensor through the model
        # Ensure the tensor is in the correct shape, device, etc.
        score = model.classify(embedding).cpu()
        #score = model.evalute_energy(embedding).cpu()
        
        # Append the path, embedding, and score as a tuple to the structure list
        structure.append((image_path, embedding, score.item(),image))  # Assuming score is a tensor, use .item() to get the value

    # Sort the structure list by the score in descending order (for ascending, remove 'reverse=True')
    # The lambda function specifies that the sorting is based on the third element of each tuple (index 2)
    sorted_structure = sorted(structure, key=lambda x: x[2], reverse=True)

    return sorted_structure



def process_and_sort_dataset_form_multiplemodels(images_paths, model_list):
    # Initialize an empty list to hold the structure for each image
    structure = []

    # Process each image path
    for image_path in images_paths:
        # Extract embedding and image tensor from the image path
        image, embedding = get_clip_and_image_from_path(image_path)
        
        # Compute the score by passing the image tensor through the model
        # Ensure the tensor is in the correct shape, device, etc.
        score = 0
        for model in model_list:
            score = score + model.classify(embedding).cpu()

        #score = model.evalute_energy(embedding).cpu()
        
        # Append the path, embedding, and score as a tuple to the structure list
        structure.append((image_path, embedding, score.item(),image))  # Assuming score is a tensor, use .item() to get the value

    # Sort the structure list by the score in descending order (for ascending, remove 'reverse=True')
    # The lambda function specifies that the sorting is based on the third element of each tuple (index 2)
    sorted_structure = sorted(structure, key=lambda x: x[2], reverse=True)

    return sorted_structure


def process_and_sort_dataset_elm(images_paths, model):
    # Initialize an empty list to hold the structure for each image
    structure = []

    # Process each image path
    for image_path in images_paths:
        # Extract embedding and image tensor from the image path
        image, embedding = get_clip_and_image_from_path(image_path)
        
        # Compute the score by passing the image tensor through the model
        # Ensure the tensor is in the correct shape, device, etc.
        #score = model.evalute_energy(embedding).cpu()
        score = model.classify(embedding).cpu()
        
        # Append the path, embedding, and score as a tuple to the structure list
        structure.append((image_path, embedding, score.item(),image))  # Assuming score is a tensor, use .item() to get the value

    # Sort the structure list by the score in descending order (for ascending, remove 'reverse=True')
    # The lambda function specifies that the sorting is based on the third element of each tuple (index 2)
    sorted_structure = sorted(structure, key=lambda x: x[2], reverse=True)

    return sorted_structure


def plot_samples_hashless(loaded_model,dataset_name, number_of_samples,tag_name):

    images_paths = get_file_paths(dataset_name,number_of_samples)
    
    # Process the images
    sorted_images_and_hashes = process_and_sort_dataset(images_paths, loaded_model) 

    rank = 1
    for image in sorted_images_and_hashes:
        #
        print("Rank : ", rank, " Path : ", image[0], " Score : ",image[2])
        rank += 0
    # Tag the images

    selected_structure_first_50 = sorted_images_and_hashes[:52] 
    selected_structure_second_50 = sorted_images_and_hashes[52:103]
    selected_structure_third_50 = sorted_images_and_hashes[103:154]
    
    tier4 = sorted_images_and_hashes[150:200] 
    tier5 = sorted_images_and_hashes[200:250]
    tier6 = sorted_images_and_hashes[250:300]
    tier7 = sorted_images_and_hashes[300:350] 
    tier8 = sorted_images_and_hashes[350:400]
    tier9 = sorted_images_and_hashes[400:450]



    tier10 = sorted_images_and_hashes[450:500] 
    tier11 = sorted_images_and_hashes[500:550]
    tier12 = sorted_images_and_hashes[550:750]
    tier13 = sorted_images_and_hashes[750:950]
    tier14 = sorted_images_and_hashes[950:1150]
    tier15 = sorted_images_and_hashes[1150:1350]

    #tag_image(file_hash,tag_id,user)

    
    plot_name1 = tag_name + "_tier1_hs"
    plot_name2 = tag_name + "_tier2_hs"
    plot_name3  = tag_name + "_tier3_hs"
    plot_name4 = tag_name + "_tier4_hs"
    plot_name5 = tag_name + "_tier5_hs"
    plot_name6  = tag_name + "_tier6_hs"
    plot_name7 = tag_name + "_tier7_hs"
    plot_name8  = tag_name + "_tier8_hs"
    plot_name9  = tag_name + "_tier9_hs"

    plot_name10 = tag_name + "_tier10_hs"
    plot_name11  = tag_name + "_tier11_hs"
    plot_name12  = tag_name + "_tier12_hs"


    plot_name13 = tag_name + "_tier13_hs"
    plot_name14  = tag_name + "_tier14_hs"
    plot_name15  = tag_name + "_tier15_hs"

    plot_images_with_scores_hasheless(selected_structure_first_50,plot_name1)
    plot_images_with_scores_hasheless(selected_structure_second_50,plot_name2)
    plot_images_with_scores_hasheless(selected_structure_third_50,plot_name3)

    plot_images_with_scores_hasheless(tier4,plot_name4)
    plot_images_with_scores_hasheless(tier5,plot_name5)
    plot_images_with_scores_hasheless(tier6,plot_name6)
    plot_images_with_scores_hasheless(tier7,plot_name7)
    plot_images_with_scores_hasheless(tier8,plot_name8)
    plot_images_with_scores_hasheless(tier9,plot_name9)

    plot_images_with_scores_hasheless(tier10,plot_name10)
    plot_images_with_scores_hasheless(tier11,plot_name11)
    plot_images_with_scores_hasheless(tier12,plot_name12)

    
    plot_images_with_scores_hasheless(tier13,plot_name13)
    plot_images_with_scores_hasheless(tier14,plot_name14)
    plot_images_with_scores_hasheless(tier15,plot_name15)



import random
def plot_samples_hashless_binning(loaded_model,dataset_name, number_of_samples,tag_name):

    images_paths = get_file_paths(dataset_name,number_of_samples)
    
    # Process the images
    sorted_images_and_hashes = process_and_sort_dataset(images_paths, loaded_model) 

    rank = 1
    for image in sorted_images_and_hashes:
        #
        print("Rank : ", rank, " Path : ", image[0], " Score : ",image[2])
        rank += 0
    # Tag the images


    offset = int(number_of_samples/16)
    offset_addition = offset

    selected_structure_first_50 = sorted_images_and_hashes[:offset] 
    selected_structure_first_50 = random.sample(selected_structure_first_50, min(len(selected_structure_first_50),52))


    selected_structure_second_50 = sorted_images_and_hashes[offset:offset+offset_addition]
    offset = offset + offset_addition
    print(f"offset is {offset}")
    selected_structure_second_50 = random.sample(selected_structure_second_50, min(len(selected_structure_second_50),52))



    selected_structure_third_50 = sorted_images_and_hashes[offset:offset+offset_addition]
    offset = offset + offset_addition
    print(f"offset is {offset}")
    selected_structure_third_50 = random.sample(selected_structure_third_50, min(len(selected_structure_third_50),52))


    tier4 = sorted_images_and_hashes[offset:offset+offset_addition]
    offset = offset + offset_addition
    print(f"offset is {offset}")
    tier4 = random.sample(tier4,min(len(tier4),52))


    tier5 = sorted_images_and_hashes[offset:offset+offset_addition]
    offset = offset + offset_addition
    print(f"offset is {offset}")
    tier5 = random.sample(tier5, min(len(tier5),52))

    tier6 = sorted_images_and_hashes[offset:offset+offset_addition]
    offset = offset + offset_addition
    print(f"offset is {offset}")
    tier6 = random.sample(tier6, min(len(tier6),52))

    tier7 = sorted_images_and_hashes[offset:offset+offset_addition]
    offset = offset + offset_addition
    print(f"offset is {offset}")
    tier7 = random.sample(tier7, min(len(tier7),52))

    tier8 = sorted_images_and_hashes[offset:offset+offset_addition]
    offset = offset + offset_addition
    print(f"offset is {offset}")
    tier8 = random.sample(tier8, min(len(tier8),52))

    tier9 = sorted_images_and_hashes[offset:offset+offset_addition]
    offset = offset + offset_addition
    print(f"offset is {offset}")
    tier9 = random.sample(tier9, min(len(tier9),52))

    tier10 = sorted_images_and_hashes[offset:offset+offset_addition]
    offset = offset + offset_addition
    print(f"offset is {offset}")
    tier10 = random.sample(tier10, min(len(tier10),52))

    tier11 = sorted_images_and_hashes[offset:offset+offset_addition]
    offset = offset + offset_addition
    print(f"offset is {offset}")
    tier11 = random.sample(tier11, min(len(tier11),52))

    tier12 = sorted_images_and_hashes[offset:offset+offset_addition]
    offset = offset + offset_addition
    print(f"offset is {offset}")
    tier12 = random.sample(tier12, min(len(tier12),52))

    tier13 = sorted_images_and_hashes[offset:offset+offset_addition]
    offset = offset + offset_addition
    print(f"offset is {offset}")
    tier13 = random.sample(tier13, min(len(tier13),52))

    tier14 = sorted_images_and_hashes[offset:offset+offset_addition]
    offset = offset + offset_addition
    print(f"offset is {offset}")
    tier14 = random.sample(tier14, min(len(tier14),52))

    tier15 = sorted_images_and_hashes[offset:offset+offset_addition]
    offset = offset + offset_addition
    print(f"offset is {offset}")
    tier15 = random.sample(tier15, min(len(tier15),52))

    tier16 = sorted_images_and_hashes[offset:offset+offset_addition]
    tier16 = random.sample(tier16, min(len(tier16),52))


    #tag_image(file_hash,tag_id,user)

    
    plot_name1 = tag_name + "_tier1_hs"
    plot_name2 = tag_name + "_tier2_hs"
    plot_name3  = tag_name + "_tier3_hs"
    plot_name4 = tag_name + "_tier4_hs"
    plot_name5 = tag_name + "_tier5_hs"
    plot_name6  = tag_name + "_tier6_hs"
    plot_name7 = tag_name + "_tier7_hs"
    plot_name8  = tag_name + "_tier8_hs"
    plot_name9  = tag_name + "_tier9_hs"

    plot_name10 = tag_name + "_tier10_hs"
    plot_name11  = tag_name + "_tier11_hs"
    plot_name12  = tag_name + "_tier12_hs"


    plot_name13 = tag_name + "_tier13_hs"
    plot_name14  = tag_name + "_tier14_hs"
    plot_name15  = tag_name + "_tier15_hs"

    plot_name16  = tag_name + "_tier16_hs"

    plot_images_with_scores_hasheless(selected_structure_first_50,plot_name1)
    plot_images_with_scores_hasheless(selected_structure_second_50,plot_name2)
    plot_images_with_scores_hasheless(selected_structure_third_50,plot_name3)

    plot_images_with_scores_hasheless(tier4,plot_name4)
    plot_images_with_scores_hasheless(tier5,plot_name5)
    plot_images_with_scores_hasheless(tier6,plot_name6)
    plot_images_with_scores_hasheless(tier7,plot_name7)
    plot_images_with_scores_hasheless(tier8,plot_name8)
    plot_images_with_scores_hasheless(tier9,plot_name9)

    plot_images_with_scores_hasheless(tier10,plot_name10)
    plot_images_with_scores_hasheless(tier11,plot_name11)
    plot_images_with_scores_hasheless(tier12,plot_name12)

    
    plot_images_with_scores_hasheless(tier13,plot_name13)
    plot_images_with_scores_hasheless(tier14,plot_name14)
    plot_images_with_scores_hasheless(tier15,plot_name15)


    plot_images_with_scores_hasheless(tier16,plot_name16)


def plot_samples_hashless_from_target_dataset(loaded_model, tag_id ,tag_name):

    images_paths = get_tag_jobs(tag_id)
    
    # Process the images
    sorted_images_and_hashes = process_and_sort_dataset(images_paths, loaded_model) 

    rank = 1
    for image in sorted_images_and_hashes:
        #
        print("Rank : ", rank, " Path : ", image[0], " Score : ",image[2])
        rank += 0
    # Tag the images

    
    selected_structure_first_50 = sorted_images_and_hashes[:52] 
    # selected_structure_second_50 = sorted_images_and_hashes[52:103]
    # selected_structure_third_50 = sorted_images_and_hashes[103:154]
    
    # tier4 = sorted_images_and_hashes[150:200] 
    # tier5 = sorted_images_and_hashes[200:250]
    # tier6 = sorted_images_and_hashes[250:300]
    # tier7 = sorted_images_and_hashes[300:350] 
    # tier8 = sorted_images_and_hashes[350:400]
    # tier9 = sorted_images_and_hashes[400:450]



    # tier10 = sorted_images_and_hashes[450:500] 
    # tier11 = sorted_images_and_hashes[500:550]
    # tier12 = sorted_images_and_hashes[550:750]
    # tier13 = sorted_images_and_hashes[750:950]
    # tier14 = sorted_images_and_hashes[950:1150]
    # tier15 = sorted_images_and_hashes[1150:1350]

    #tag_image(file_hash,tag_id,user)

    
    plot_name1 = tag_name + "_tier1_hs"
    # plot_name2 = tag_name + "_tier2_hs"
    # plot_name3  = tag_name + "_tier3_hs"
    # plot_name4 = tag_name + "_tier4_hs"
    # plot_name5 = tag_name + "_tier5_hs"
    # plot_name6  = tag_name + "_tier6_hs"
    # plot_name7 = tag_name + "_tier7_hs"
    # plot_name8  = tag_name + "_tier8_hs"
    # plot_name9  = tag_name + "_tier9_hs"

    # plot_name10 = tag_name + "_tier10_hs"
    # plot_name11  = tag_name + "_tier11_hs"
    # plot_name12  = tag_name + "_tier12_hs"


    # plot_name13 = tag_name + "_tier13_hs"
    # plot_name14  = tag_name + "_tier14_hs"
    # plot_name15  = tag_name + "_tier15_hs"

    plot_images_with_scores_hasheless(selected_structure_first_50,plot_name1)
    # plot_images_with_scores_hasheless(selected_structure_second_50,plot_name2)
    # plot_images_with_scores_hasheless(selected_structure_third_50,plot_name3)

    # plot_images_with_scores_hasheless(tier4,plot_name4)
    # plot_images_with_scores_hasheless(tier5,plot_name5)
    # plot_images_with_scores_hasheless(tier6,plot_name6)
    # plot_images_with_scores_hasheless(tier7,plot_name7)
    # plot_images_with_scores_hasheless(tier8,plot_name8)
    # plot_images_with_scores_hasheless(tier9,plot_name9)

    # plot_images_with_scores_hasheless(tier10,plot_name10)
    # plot_images_with_scores_hasheless(tier11,plot_name11)
    # plot_images_with_scores_hasheless(tier12,plot_name12)

    
    # plot_images_with_scores_hasheless(tier13,plot_name13)
    # plot_images_with_scores_hasheless(tier14,plot_name14)
    # plot_images_with_scores_hasheless(tier15,plot_name15)


def plot_samples_graph(loaded_model,dataset_name, number_of_samples,tag_name):

    images_paths = get_file_paths(dataset_name,number_of_samples)
    
    # Process the images
    sorted_images_and_hashes = process_and_sort_dataset(images_paths, loaded_model) 

    rank = 1
    ranks = []  # List to store ranks
    scores = []  # List to store scores

    for image in sorted_images_and_hashes:
        #
        
        ranks.append(rank)
        scores.append(image["score"])
        rank += 1
    # Tag the images

    # Plotting the graph
    plt.figure(figsize=(8, 6))
    plt.plot(ranks, scores, marker='o')
    plt.xlabel('Rank')
    plt.ylabel('Score')
    plt.title(f'Sample Graph: Rank vs Score for {tag_name}')
    plt.grid(True)
    plt.savefig("output/rank.png")

    # Save the figure to a file
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # upload the graph report
    minio_path="environmental/output/my_tests"
    minio_path= minio_path + "/ranking_distribution_"+ tag_name + '_' +date_now+".png"
    cmd.upload_data(minio_client, 'datasets', minio_path, buf)
    # Remove the temporary file
    os.remove("output/rank.png")
    # Clear the current figure
    plt.clf()


from scipy.interpolate import interp1d

def plot_samples_graph_interpolation(loaded_model,dataset_name, number_of_samples,tag_name, model_type):

    images_paths = get_file_paths(dataset_name,number_of_samples)
    
    # Process the images
    sorted_images_and_hashes = process_and_sort_dataset(images_paths, loaded_model) 

    rank = 1
    ranks = []  # List to store ranks
    scores = []  # List to store scores

    for image in sorted_images_and_hashes:
        #
        print("Rank : ", rank, " Path : ", image[0], " Score : ",image[2])
        ranks.append(rank)
        scores.append(image["score"])
        rank += 1
    # Tag the images

    
    xs = ranks
    ys = scores
    # Generate additional points for higher granularity (64 segments)
    x_dense = np.linspace(min(xs), max(xs), 64)
    y_dense = interp1d(xs, ys, kind='linear')(x_dense)

    # Linear interpolation function with higher granularity
    interp_func_dense = interp1d(x_dense, y_dense, kind='linear')

    # Plot the original function and the piecewise linear approximation with segments
    plt.plot(x_dense, y_dense, label='Piecewise Linear Approximation (64 segments)', linewidth=2, linestyle='--')
    plt.plot(xs, ys,  label='Real data points', markersize=3,linestyle='--')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Real data VS Linear Approximation: {tag_name} using {model_type}')
    plt.legend()
    plt.grid(True)

    # # Plotting the graph
    # plt.figure(figsize=(8, 6))
    # plt.plot(ranks, scores, marker='o')
    # plt.xlabel('Rank')
    # plt.ylabel('Score')
    # plt.title(f'Sample Graph: Rank vs Score for {tag_name}')
    # plt.grid(True)
    plt.savefig("output/rank.png")

    # Save the figure to a file
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # upload the graph report
    minio_path="environmental/output/my_tests"
    minio_path= minio_path + "/ranking_distribution_"+ tag_name + '_' +date_now+".png"
    cmd.upload_data(minio_client, 'datasets', minio_path, buf)
    # Remove the temporary file
    os.remove("output/rank.png")
    # Clear the current figure
    plt.clf()


def piecewise_linear(score, start_x, start_y, end_x, end_y):
    return (start_y * (end_x - score) + end_y * (score - start_x)) / (end_x - start_x)

def plot_samples_graph_interpolation_plus_mapping(loaded_model,dataset_name, number_of_samples,tag_name, model_type):

    images_paths = get_file_paths(dataset_name,number_of_samples)
    
    # Process the images
    sorted_images_and_hashes = process_and_sort_dataset(images_paths, loaded_model) 
    
    rank = 1
    ranks = []  # List to store ranks
    scores = []  # List to store scores

    for image in sorted_images_and_hashes:
        #
        print("Rank : ", rank, " Path : ", image[0], " Score : ",image[2])
        ranks.append(rank)
        scores.append(image[2])
        rank += 1
    # Tag the images

    xs = ranks
    ys = scores

    max_score = sorted_images_and_hashes[0][2]
    min_score = sorted_images_and_hashes[number_of_samples-1][2]
    
   

    # Categorize scores into bins
    num_bins = 1024
    bins = np.linspace(min_score, max_score, num_bins+1)
    bin_indices = np.digitize(scores, bins)

    # mapping_functions = []
    # for bin_idx in range(1, num_bins+1):
    #     bin_start = bins[bin_idx - 1]
    #     bin_end = bins[bin_idx]
    #     # Map scores in each bin from +x to -x to 1 to -1
    #     mapping_function = lambda score: piecewise_linear(score, bin_start, 1, bin_end, -1)
    #     mapping_functions.append(mapping_function)

    # # Apply mapping functions to scores in each bin
    # mapped_scores = [mapping_functions[bin_idx - 1](score) for bin_idx, score in zip(bin_indices, scores)]


    # Adjust bin indices to ensure they don't exceed the number of bins
    bin_indices = np.clip(bin_indices, 1, num_bins)

    # Define mapping functions for each bin
    mapping_functions = []
    for bin_idx in range(1, num_bins + 1):
        bin_start = min_score
        bin_end = max_score
        # Map scores in each bin from +x to -x to 1 to -1
        mapping_function = lambda score: piecewise_linear(score, bin_start, -1 , max_score, +1)
        
        mapping_functions.append(mapping_function)

    # Debugging prints
    print("Length of mapping_functions:", len(mapping_functions))
    print("Length of bin_indices:", len(bin_indices))

    # Apply mapping functions to scores in each bin
    mapped_scores = []
    for bin_idx, score in zip(bin_indices, scores):
        if bin_idx >= 1 and bin_idx <= num_bins:
            mapping_function = mapping_functions[bin_idx - 1]
            mapped_score = mapping_function(score)
            mapped_scores.append(mapped_score)
            print(f'the bin {bin_idx} values is {mapped_score}')
        else:
            print(f"Invalid bin index: {bin_idx}")

    # Print mapped_scores for inspection
    print("Length of mapped_scores:", len(mapped_scores))
    print(f'max score is {max_score} and min score is {min_score}')
    # Generate additional points for higher granularity (64 segments)

    x_dense = np.linspace(min(xs), max(xs), 64)
    y_dense = interp1d(xs, ys, kind='linear')(x_dense)

    # Linear interpolation function with higher granularity
    interp_func_dense = interp1d(x_dense, y_dense, kind='linear')

    # Plot the original function and the piecewise linear approximation with segments
    plt.plot(x_dense, y_dense, label='Piecewise Linear Approximation (64 segments)', linewidth=2, linestyle='--')
    plt.plot(xs, ys,  label='Real data points', markersize=3,linestyle='--')
    plt.plot(np.arange(len(mapped_scores)), mapped_scores,  label='pricewise linear(1024 segs, limited to -+ 1)', markersize=3,linestyle='--')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Mapping data for: {tag_name} using {model_type}')
    plt.legend()
    plt.grid(True)

    # # Plotting the graph
    # plt.figure(figsize=(8, 6))
    # plt.plot(ranks, scores, marker='o')
    # plt.xlabel('Rank')
    # plt.ylabel('Score')
    # plt.title(f'Sample Graph: Rank vs Score for {tag_name}')
    # plt.grid(True)
    plt.savefig("output/rank.png")

    # Save the figure to a file
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # upload the graph report
    minio_path="environmental/output/my_tests"
    minio_path= minio_path + "/ranking_distribution_"+ tag_name + '_' +date_now+".png"
    cmd.upload_data(minio_client, 'datasets', minio_path, buf)
    # Remove the temporary file
    os.remove("output/rank.png")
    # Clear the current figure
    plt.clf()

    return sorted_images_and_hashes, mapped_scores




def get_file_paths(dataset,num_samples):
        
        response = requests.get(f'{API_URL}/queue/image-generation/list-by-dataset?dataset={dataset}&size={num_samples}')
        
        jobs = json.loads(response.content)
        
        file_paths=[job['file_path'] for job in jobs]
        #image_hashes=[job['image_hash'] for job in jobs]
        
        return file_paths









def plot_samples_hashless_combination(loaded_model_list,dataset_name, number_of_samples,tag_name):

    images_paths = get_file_paths(dataset_name,number_of_samples)
    
    # Process the images
    sorted_images_and_hashes = process_and_sort_dataset_form_multiplemodels(images_paths, loaded_model_list) 

    rank = 1
    for image in sorted_images_and_hashes:
        #
        print("Rank : ", rank, " Path : ", image[0], " Score : ",image[2])
        rank += 0
    # Tag the images

    selected_structure_first_50 = sorted_images_and_hashes[:52] 
    selected_structure_second_50 = sorted_images_and_hashes[52:103]
    selected_structure_third_50 = sorted_images_and_hashes[103:154]
    
    tier4 = sorted_images_and_hashes[150:200] 
    tier5 = sorted_images_and_hashes[200:250]
    tier6 = sorted_images_and_hashes[250:300]
    tier7 = sorted_images_and_hashes[300:350] 
    tier8 = sorted_images_and_hashes[350:400]
    tier9 = sorted_images_and_hashes[400:450]



    tier10 = sorted_images_and_hashes[450:500] 
    tier11 = sorted_images_and_hashes[500:550]
    tier12 = sorted_images_and_hashes[550:750]
    tier13 = sorted_images_and_hashes[750:950]
    tier14 = sorted_images_and_hashes[950:1150]
    tier15 = sorted_images_and_hashes[1150:1350]

    #tag_image(file_hash,tag_id,user)

    
    plot_name1 = tag_name + "_tier1_hs"
    plot_name2 = tag_name + "_tier2_hs"
    plot_name3  = tag_name + "_tier3_hs"
    plot_name4 = tag_name + "_tier4_hs"
    plot_name5 = tag_name + "_tier5_hs"
    plot_name6  = tag_name + "_tier6_hs"
    plot_name7 = tag_name + "_tier7_hs"
    plot_name8  = tag_name + "_tier8_hs"
    plot_name9  = tag_name + "_tier9_hs"

    plot_name10 = tag_name + "_tier10_hs"
    plot_name11  = tag_name + "_tier11_hs"
    plot_name12  = tag_name + "_tier12_hs"


    plot_name13 = tag_name + "_tier13_hs"
    plot_name14  = tag_name + "_tier14_hs"
    plot_name15  = tag_name + "_tier15_hs"

    plot_images_with_scores_hasheless(selected_structure_first_50,plot_name1)
    plot_images_with_scores_hasheless(selected_structure_second_50,plot_name2)
    plot_images_with_scores_hasheless(selected_structure_third_50,plot_name3)

    plot_images_with_scores_hasheless(tier4,plot_name4)
    plot_images_with_scores_hasheless(tier5,plot_name5)
    plot_images_with_scores_hasheless(tier6,plot_name6)
    plot_images_with_scores_hasheless(tier7,plot_name7)
    plot_images_with_scores_hasheless(tier8,plot_name8)
    plot_images_with_scores_hasheless(tier9,plot_name9)

    plot_images_with_scores_hasheless(tier10,plot_name10)
    plot_images_with_scores_hasheless(tier11,plot_name11)
    plot_images_with_scores_hasheless(tier12,plot_name12)

    
    plot_images_with_scores_hasheless(tier13,plot_name13)
    plot_images_with_scores_hasheless(tier14,plot_name14)
    plot_images_with_scores_hasheless(tier15,plot_name15)


def get_file_paths_and_hashes_uuid(dataset,num_samples):
        print('Loading image file paths')
        response = requests.get(f'{API_URL}/image/list-image-metadata-by-dataset-v1?dataset={dataset}&limit={num_samples}')
        
        jobs_full = json.loads(response.content)
        
        jobs = jobs_full.get('response', [])
        #print(jobs)
        file_paths=[job['image_path'] for job in jobs]
        hashes=[job['image_hash'] for job in jobs]
        uuid =[job['uuid'] for job in jobs] 
        #image_hashes=[job['image_hash'] for job in jobs]

        for i in  range(len(file_paths)):
            print("Path : ", file_paths[i], " Hash : ", hashes[i], " UUID : ",uuid[i])
        
        return file_paths, hashes,uuid


def process_and_sort_dataset_with_hashes_uui_dict(images_paths, hashes,uuid, model):
    # Initialize an empty list to hold the structure for each image
    structure = []

    # Process each image path
    for i in range(len(images_paths)):
        # Extract embedding and image tensor from the image path
        print(images_paths[i])
        image, embedding = get_clip_and_image_from_path(images_paths[i])
        
        # Compute the score by passing the image tensor through the model
        # Ensure the tensor is in the correct shape, device, etc.
        score = model.classify(embedding).cpu()
        
        image_dict = {
            'path': images_paths[i],
            'embedding': embedding,
            'score': score.item(),
            'image_tensor': image,
            'hash': hashes[i],
            'uuid': uuid[i]
        }

        # Append the path, embedding, and score as a tuple to the structure list
        structure.append(image_dict)  # Assuming score is a tensor, use .item() to get the value

    # The lambda function specifies that the sorting is based on the third element of each tuple (index 2)
    sorted_structure = sorted(structure, key=lambda x: x['score'], reverse=True)

    return sorted_structure


def get_segment_points(min_score, max_score, num_segments):
    segment_size = (max_score - min_score) / num_segments
    return [min_score + segment_size * i for i in range(num_segments + 1)]

def map_scores(x, min_score, max_score, num_segments, target_min=0, target_max=1):
    segment_points = get_segment_points(min_score, max_score, num_segments)
    slope_list = [(target_max - target_min) / (segment_points[1] - segment_points[0]) for segment_points in zip(segment_points[:-1], segment_points[1:])]

    for i, slope in enumerate(slope_list):
        if x < segment_points[i + 1]:
            intercept = target_min - slope * segment_points[i]
            y = slope * x + intercept
            return y


def plot_samples_graph_interpolation_plus_mapping_v2(loaded_model,dataset_name, number_of_samples,tag_name, model_type):



    # get the paths and hashes
    images_paths_ood, images_hashes_ood, uuid_ood = get_file_paths_and_hashes_uuid(dataset_name,number_of_samples)

    # Process the images
    sorted_images_and_hashes = process_and_sort_dataset_with_hashes_uui_dict(images_paths_ood, images_hashes_ood,uuid_ood, loaded_model) 

    rank = 1
    ranks = []  # List to store ranks
    scores = []  # List to store scores

    for image in sorted_images_and_hashes:
        #
        print("Rank : ", rank, " Path : ", image["path"], " Score : ",image["score"])
        ranks.append(rank)
        scores.append(image["score"])
        rank += 1
    # Tag the images

    xs = ranks
    ys = scores

    max_score = sorted_images_and_hashes[0]["score"]
    min_score = sorted_images_and_hashes[number_of_samples-1]["score"]
    
   

    # Categorize scores into bins
    num_bins = 256
    bins = np.linspace(min_score, max_score, num_bins+1)
    bin_indices = np.digitize(scores, bins)


    # Adjust bin indices to ensure they don't exceed the number of bins
    bin_indices = np.clip(bin_indices, 1, num_bins)

    # Define mapping functions for each bin
    mapping_functions = []
    for bin_idx in range(1, num_bins + 1):
        bin_start = min_score
        bin_end = max_score
        # Map scores in each bin from +x to -x to 1 to -1
        mapping_function = lambda score: piecewise_linear(score, bin_start, 0 , max_score, 1)
        
        mapping_functions.append(mapping_function)

    # Debugging prints
    print("Length of mapping_functions:", len(mapping_functions))
    print("Length of bin_indices:", len(bin_indices))

    # Apply mapping functions to scores in each bin
    mapped_scores = []
    for bin_idx, score in zip(bin_indices, scores):
        if bin_idx >= 1 and bin_idx <= num_bins:
            mapping_function = mapping_functions[bin_idx - 1]
            mapped_score = mapping_function(score)
            mapped_scores.append(mapped_score)
            print(f'the bin {bin_idx} values is {mapped_score}')
        else:
            print(f"Invalid bin index: {bin_idx}")



    
    # Print mapped_scores for inspection
    print("Length of mapped_scores:", len(mapped_scores))
    print(f'max score is {max_score} and min score is {min_score}')
    # Generate additional points for higher granularity (64 segments)

    x_dense = np.linspace(min(xs), max(xs), 64)
    y_dense = interp1d(xs, ys, kind='linear')(x_dense)

    # Linear interpolation function with higher granularity
    interp_func_dense = interp1d(x_dense, y_dense, kind='linear')

    # # Plot the original function and the piecewise linear approximation with segments
    # fig, ax1 = plt.subplots()
    # #plt.plot(x_dense, y_dense, label='Piecewise Linear Approximation (64 segments)', linewidth=2, linestyle='--')
    # ax1.set_xlabel('Rank')
    # ax1.set_ylabel('Energy')
    # ax1.plot(xs, ys,  label='Real data points', markersize=3,linestyle='--')
   

    # ax2 = ax1.twinx() 
    # ax1.set_xlabel('Rank')
    # ax1.set_ylabel('Energy approximation')
    # ax2.plot(np.arange(len(mapped_scores)), mapped_scores,  label=f'pricewise linear({num_bins} segs, limited to -+ 1)', markersize=3,linestyle='--')
    # fig.tight_layout() 


   # Plot the original function and the piecewise linear approximation with segments
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Rank')
    ax1.set_ylabel('Energy')
    ax1.plot(xs, ys, label='Real data points',  markersize=3, color='blue')

    # Create a second y-axis for the second plot
    ax2 = ax1.twinx()
    ax2.set_ylabel('Energy approximation')
    ax2.plot(np.arange(len(mapped_scores)), mapped_scores, label='Piecewise Linear',  markersize=3, linestyle='--',color='red')

    # Set the y-axis limits for the second plot dynamically based on the real data
    ax2.set_ylim(min(1, 0))
    #print(f'max score is {max_score} and min score is {min_score}')
    #fig.tight_layout()
    plt.title(f'Sample Graph: Rank vs Score for {tag_name}')
    
    plt.legend(loc='best')
    plt.grid(True)



    # plt.title(f'Mapping data for: {tag_name} using {model_type}')
    # plt.legend()
    # plt.grid(True)

    # # Plotting the graph
    # plt.figure(figsize=(8, 6))
    # plt.plot(ranks, scores, marker='o')
    # plt.xlabel('Rank')
    # plt.ylabel('Score')
    # plt.title(f'Sample Graph: Rank vs Score for {tag_name}')
    # plt.grid(True)
    plt.savefig("output/rank.png")

    # Save the figure to a file
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # upload the graph report
    minio_path="environmental/output/my_tests"
    minio_path= minio_path + "/ranking_distribution_"+ tag_name + '_' +date_now+".png"
    cmd.upload_data(minio_client, 'datasets', minio_path, buf)
    # Remove the temporary file
    os.remove("output/rank.png")
    # Clear the current figure
    plt.clf()

    return sorted_images_and_hashes, mapped_scores







def plot_samples_graph_interpolation_plus_mapping_combined(loaded_model, loaded_model_2, dataset_name, number_of_samples,tag_name, model_type,tag_name_1,tag_name_2):



    # get the paths and hashes
    images_paths_ood, images_hashes_ood, uuid_ood = get_file_paths_and_hashes_uuid(dataset_name,number_of_samples)

    # Process the images
    sorted_images_and_hashes = process_and_sort_dataset_with_hashes_uui_dict(images_paths_ood, images_hashes_ood,uuid_ood, loaded_model) 





    rank = 1
    ranks = []  # List to store ranks
    scores = []  # List to store scores

    for image in sorted_images_and_hashes:
        #
        print("Rank : ", rank, " Path : ", image["path"], " Score : ",image["score"])
        ranks.append(rank)
        scores.append(image["score"])
        rank += 1
    # Tag the images

    xs = ranks
    ys = scores

    max_score = sorted_images_and_hashes[0]["score"]
    min_score = sorted_images_and_hashes[number_of_samples-1]["score"]
    
   

    # Categorize scores into bins
    num_bins = 1024
    bins = np.linspace(min_score, max_score, num_bins+1)
    bin_indices = np.digitize(scores, bins)


    # Adjust bin indices to ensure they don't exceed the number of bins
    bin_indices = np.clip(bin_indices, 1, num_bins)

    # Define mapping functions for each bin
    mapping_functions = []
    for bin_idx in range(1, num_bins + 1):
        bin_start = min_score
        bin_end = max_score
        # Map scores in each bin from +x to -x to 1 to -1
        mapping_function = lambda score: piecewise_linear(score, bin_start, 0 , max_score, 1)
        
        mapping_functions.append(mapping_function)

    # Debugging prints
    print("Length of mapping_functions:", len(mapping_functions))
    print("Length of bin_indices:", len(bin_indices))

    # Apply mapping functions to scores in each bin
    mapped_scores = []
    for bin_idx, score in zip(bin_indices, scores):
        if bin_idx >= 1 and bin_idx <= num_bins:
            mapping_function = mapping_functions[bin_idx - 1]
            mapped_score = mapping_function(score)
            mapped_scores.append(mapped_score)
            print(f'the bin {bin_idx} values is {mapped_score}')
        else:
            print(f"Invalid bin index: {bin_idx}")



    ###################### Element 2

    # Process the images
    sorted_images_and_hashes_2 = process_and_sort_dataset_with_hashes_uui_dict(images_paths_ood, images_hashes_ood,uuid_ood, loaded_model_2) 


    rank_2 = 1
    ranks_2 = []  # List to store ranks
    scores_2 = []  # List to store scores
    print(f"the length of hash 2 is {len(sorted_images_and_hashes_2)}")

    for image in sorted_images_and_hashes_2:
        #
        print("Rank 2: ", rank_2, " Path 2: ", image["path"], " Score 2: ",image["score"])
        ranks_2.append(rank_2)
        scores_2.append(image["score"])
        rank_2 += 1
    # Tag the images

    xs_2 = ranks_2
    ys_2 = scores_2

    max_score_2 = sorted_images_and_hashes_2[0]["score"]
    min_score_2 = sorted_images_and_hashes_2[number_of_samples-1]["score"]
    


    # Categorize scores into bins
    num_bins_2 = 1024
    bins_2 = np.linspace(min_score_2, max_score_2, num_bins_2+1)
    bin_indices_2 = np.digitize(scores_2, bins_2)


    # Adjust bin indices to ensure they don't exceed the number of bins
    bin_indices_2 = np.clip(bin_indices_2, 1, num_bins_2)



    # # Define mapping functions for each bin
    # mapping_functions = []
    # for bin_idx in range(1, num_bins + 1):
    #     bin_start = min_score
    #     bin_end = max_score
    #     # Map scores in each bin from +x to -x to 1 to -1
    #     mapping_function = lambda score: piecewise_linear(score, bin_start, -1 , max_score, +1)
        
    #     mapping_functions.append(mapping_function)




    # Define mapping functions for each bin
    
    mapping_functions_2 = []
    for bin_idx_2 in range(1, num_bins_2 + 1):
        bin_start_2 = min_score_2
        bin_end_2 = max_score_2
        # Map scores in each bin from +x to -x to 1 to -1
        mapping_function_2 = lambda score: piecewise_linear(score, bin_start_2, -1 , max_score_2, +1)
        
        mapping_functions_2.append(mapping_function_2)

    # Debugging prints
    print("Length of mapping_functions:", len(mapping_functions_2))
    print("Length of bin_indices:", len(bin_indices_2))




    # Apply mapping functions to scores in each bin
    mapped_scores_2 = []
    for bin_idx_2, score in zip(bin_indices_2, scores):
        if bin_idx_2 >= 1 and bin_idx_2 <= num_bins_2:
            mapping_function_2 = mapping_functions_2[bin_idx_2 - 1]
            mapped_score_2 = mapping_function_2(score)
            mapped_scores_2.append(mapped_score_2)
            print(f'the bin {bin_idx_2} values is {mapped_score_2}')
        else:
            print(f"Invalid bin index: {bin_idx_2}")


    # Print mapped_scores for inspection
    print("Length of mapped_scores:", len(mapped_scores))
    print(f'max score is {max_score} and min score is {min_score}')
    # Generate additional points for higher granularity (64 segments)

    x_dense = np.linspace(min(xs), max(xs), 64)
    y_dense = interp1d(xs, ys, kind='linear')(x_dense)

    # Linear interpolation function with higher granularity
    interp_func_dense = interp1d(x_dense, y_dense, kind='linear')

    # Plot the original function and the piecewise linear approximation with segments
    #plt.plot(x_dense, y_dense, label='Piecewise Linear Approximation (64 segments)', linewidth=2, linestyle='--')
    plt.plot(xs, ys,  label=f'Real data points for {tag_name_1}', markersize=3,linestyle='--')
    plt.plot(xs_2, ys_2,  label=f'Real data points for {tag_name_2}', markersize=3,linestyle='--')
    plt.plot(np.arange(len(mapped_scores)), mapped_scores,  label=f'pricewise linear for {tag_name_1}', markersize=3,linestyle='--')
    plt.plot(np.arange(len(mapped_scores_2)), mapped_scores_2,  label=f'pricewise linear for {tag_name_2} ', markersize=3,linestyle='--')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Mapping data for: {tag_name_1} &nd {tag_name_2} using {model_type}')
    plt.legend()
    plt.grid(True)

    # # Plotting the graph
    # plt.figure(figsize=(8, 6))
    # plt.plot(ranks, scores, marker='o')
    # plt.xlabel('Rank')
    # plt.ylabel('Score')
    # plt.title(f'Sample Graph: Rank vs Score for {tag_name}')
    # plt.grid(True)
    plt.savefig("output/rank.png")

    # Save the figure to a file
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # upload the graph report
    minio_path="environmental/output/my_tests"
    minio_path= minio_path + "/ranking_distribution_"+ tag_name + '_' +date_now+".png"
    cmd.upload_data(minio_client, 'datasets', minio_path, buf)
    # Remove the temporary file
    os.remove("output/rank.png")
    # Clear the current figure
    plt.clf()

    combined_data = []
    sorted_images_and_hashes_updated_1 = []
    sorted_images_and_hashes_updated_2 = []

    for i in range (0,len(sorted_images_and_hashes)):
        sorted_images_and_hashes_updated_1.append({"path": sorted_images_and_hashes[i]["path"], 
                                                   "embedding": sorted_images_and_hashes[i]["embedding"],
                                                    "score": mapped_scores[i],
                                                      "image_tensor": sorted_images_and_hashes[i]["image_tensor"]})
                                                       

    for i in range (0,len(sorted_images_and_hashes_2)):
        sorted_images_and_hashes_updated_2.append({"path": sorted_images_and_hashes_2[i]["path"], 
                                                    "embedding": sorted_images_and_hashes_2[i]["embedding"], 
                                                    "score": mapped_scores_2[i],
                                                    "image_tensor":  sorted_images_and_hashes_2[i]["image_tensor"]})


    for item_one in sorted_images_and_hashes_updated_1:
        print(f'item : {item_one}')
        path = item_one['path']
        score_one = item_one['score']

        for item_two in sorted_images_and_hashes_updated_2:
            if item_two['path'] == path:
                combined_score = score_one + item_two['score']
                print(f"path 1: {path} is path 2: {item_two['path']} old score is {score_one} new score is {item_two['score']} and combined is {combined_score}")
                # sorted_images[i]["embedding"], new_scores[i], sorted_images[i]["image_tensor"])

                # append((sorted_images[i]["path"], sorted_images[i]["embedding"], new_scores[i], sorted_images[i]["image_tensor"]))
                combined_element = {'path': item_one['path'], 'score': combined_score, "embedding": item_one["embedding"], "image_tensor": item_one["image_tensor"] }
                combined_data.append(combined_element)
                break  # Stop searching in list_two once we find the matching path

   
    result = sorted(combined_data, key=lambda x: x['score'], reverse=True)            

    return  result  #sorted_images_and_hashes, sorted_images_and_hashes_2, mapped_scores, mapped_scores_2






######### let's go




# #EBM
args = parse_args()
original_model=EBM_Single_Class(minio_access_key=args.minio_access_key,
                            minio_secret_key=args.minio_secret_key,
                            dataset= args.dataset,
                            class_name= "topic-aquatic" ,
                            model = None,
                            save_name = args.save_name,
                            class_id =  get_tag_id_by_name(args.class_name),
                            training_batch_size=args.training_batch_size,
                            num_samples= args.num_samples,
                            epochs= args.epochs,
                            learning_rate= args.learning_rate)

#original_model = EBM_Single_Class(train_loader = None,val_loader = None, adv_loader = None,img_shape=(1280,))
# Load the last occult trained model
 # "defect-color-over-saturated" #"defect-color-mildly-over-saturated" # "defect-color-over-saturated"  #"topic-forest" #"concept-occult" #"concept-cybernetic"  #"defect-color-too-dark" #"content-has-waifu" #"concept-occult" #"topic-aquatic" #"topic-aquatic" #"topic-desert"
#  "topic-desert" topic-med "topic-medieval"  "concept-cybernetic"
tag_name_x =  "concept-cybernetic"
original_model.load_model_from_minio(minio_client, dataset_name = "environmental", tag_name =tag_name_x, model_type = "energy-based-model")



# ELM
from training_worker.classifiers.models.elm_regression import ELMRegression
# elm_model = ELMRegression()
#def load_model(self, minio_client, model_dataset, tag_name, model_type, scoring_model, not_include, device=None):

elm_model, _ = load_model_elm(device = original_model.device, minio_client = minio_client, model_dataset = "environmental",scoring_model = 'score' ,tag_name = tag_name_x, model_type = "elm-regression-clip-h", not_include= 'batatatatatata')


# 4 Iso 
# 18 Forest
# 35 cyber
#plot_samples_hashless_from_target_dataset(original_model, 35 ,tag_name = tag_name_x)
# plot_samples_hashless_from_target_dataset(elm_model, 35 ,tag_name = tag_name_x)

#plot_samples_hashless(loaded_model = original_model, dataset_name = "environmental", number_of_samples = 30000,tag_name =tag_name_x)
#plot_samples_hashless(loaded_model = elm_model, dataset_name = "environmental", number_of_samples = 30000,tag_name =tag_name_x)


# next elm
#plot_samples_hashless_binning(loaded_model = original_model, dataset_name = "environmental", number_of_samples = 40000,tag_name =tag_name_x)
#plot_samples_hashless_binning(loaded_model = elm_model, dataset_name = "environmental", number_of_samples = 32000,tag_name =tag_name_x)

# Graphs
#plot_samples_graph(loaded_model = original_model, dataset_name = "environmental", number_of_samples = 40000,tag_name =tag_name_x)
#plot_samples_graph(loaded_model = elm_model, dataset_name = "environmental", number_of_samples = 40000,tag_name =tag_name_x)


# graph interpol
#plot_samples_graph_interpolation(loaded_model = original_model, dataset_name = "environmental", number_of_samples = 40000,tag_name =tag_name_x, model_type = "EBM Model" )
sorted_images , new_scores = plot_samples_graph_interpolation_plus_mapping_v2(loaded_model = original_model, dataset_name = "environmental", number_of_samples = 40000 ,tag_name =tag_name_x, model_type = "EBM Model" )

sorted_images_x = []


      
        # image_dict = {
        #     'path': images_paths[i],
        #     'embedding': embedding,
        #     'score': score.item(),
        #     'image_tensor': image,
        #     'hash': hashes[i],
        #     'uuid': uuid[i]
        # }


#structure.append((image_path, embedding, score.item(),image)) 
tag_name = tag_name_x
for i in range (0,len(sorted_images)):
    sorted_images_x.append((sorted_images[i]["path"], sorted_images[i]["embedding"], new_scores[i], sorted_images[i]["image_tensor"]))
    #sorted_images_x  sorted_images[i][2] = new_scores[i]


sorted_images_and_hashes = sorted_images_x

# rank = 1
# for image in sorted_images_and_hashes:
#     #
#     print("Rank : ", rank, " Path : ", image["path"], " Score : ",image["score"])
#     rank += 0
# # Tag the images

selected_structure_first_50 = sorted_images_and_hashes[:52] 
selected_structure_second_50 = sorted_images_and_hashes[52:103]
selected_structure_third_50 = sorted_images_and_hashes[103:154]

tier4 = sorted_images_and_hashes[150:200] 
tier5 = sorted_images_and_hashes[200:250]
tier6 = sorted_images_and_hashes[250:300]
tier7 = sorted_images_and_hashes[300:350] 
tier8 = sorted_images_and_hashes[350:400]
tier9 = sorted_images_and_hashes[400:450]



tier10 = sorted_images_and_hashes[450:500] 
tier11 = sorted_images_and_hashes[500:550]
tier12 = sorted_images_and_hashes[550:750]
tier13 = sorted_images_and_hashes[750:950]
tier14 = sorted_images_and_hashes[950:1150]
tier15 = sorted_images_and_hashes[1150:1350]

#tag_image(file_hash,tag_id,user)


plot_name1 = tag_name + "_tier1_hs"
plot_name2 = tag_name + "_tier2_hs"
plot_name3  = tag_name + "_tier3_hs"
plot_name4 = tag_name + "_tier4_hs"
plot_name5 = tag_name + "_tier5_hs"
plot_name6  = tag_name + "_tier6_hs"
plot_name7 = tag_name + "_tier7_hs"
plot_name8  = tag_name + "_tier8_hs"
plot_name9  = tag_name + "_tier9_hs"

plot_name10 = tag_name + "_tier10_hs"
plot_name11  = tag_name + "_tier11_hs"
plot_name12  = tag_name + "_tier12_hs"


plot_name13 = tag_name + "_tier13_hs"
plot_name14  = tag_name + "_tier14_hs"
plot_name15  = tag_name + "_tier15_hs"

plot_images_with_scores_hasheless(selected_structure_first_50,plot_name1)
# plot_images_with_scores_hasheless(selected_structure_second_50,plot_name2)
# plot_images_with_scores_hasheless(selected_structure_third_50,plot_name3)

# plot_images_with_scores_hasheless(tier4,plot_name4)
# plot_images_with_scores_hasheless(tier5,plot_name5)
# plot_images_with_scores_hasheless(tier6,plot_name6)
# plot_images_with_scores_hasheless(tier7,plot_name7)
# plot_images_with_scores_hasheless(tier8,plot_name8)
# plot_images_with_scores_hasheless(tier9,plot_name9)

# plot_images_with_scores_hasheless(tier10,plot_name10)
# plot_images_with_scores_hasheless(tier11,plot_name11)
# plot_images_with_scores_hasheless(tier12,plot_name12)


# plot_images_with_scores_hasheless(tier13,plot_name13)
# plot_images_with_scores_hasheless(tier14,plot_name14)
# plot_images_with_scores_hasheless(tier15,plot_name15)



############################ comb with - + 1 ########################








############################ Train ########################


# tag_name_x_2 = "perspective-2d-side-view" # "topic-medieval" # "content-has-character" #"perspective-isometric"  # "perspective-3d"  #"concept-cybernetic" #"concept-nature"
# defect_test=EBM_Single_Class(minio_access_key="D6ybtPLyUrca5IdZfCIM",
#                             minio_secret_key= "2LZ6pqIGOiZGcjPTR6DZPlElWBkRTkaLkyLIBt4V",
#                             dataset= "environmental",
#                             class_name= tag_name_x_2,
#                             model = None,
#                             save_name = "bla",
#                             class_id =  get_tag_id_by_name(tag_name_x_2),
#                             training_batch_size=64,
#                             num_samples= 32000,
#                             epochs= 20,
#                             learning_rate= 0.001)


# defect_test.train_v3()


# defect_test.load_model_from_minio(minio_client , dataset_name = "environmental", tag_name =tag_name_x_2, model_type = "energy-based-model")
# plot_samples_hashless(loaded_model = defect_test, dataset_name = "environmental", number_of_samples = 30000,tag_name =tag_name_x_2)





########## comb

# model_list = []

# model_1_name = "content-has-character" # "topic-forest" #"topic-medieval" # "perspective-isometric" # "perspective-3d" "perspective-isometric" #"topic-medieval"  #  "topic-forest" "topic-desert" "topic-aquatic" "concept-cybernetic" "concept-nature" 
# model_1=EBM_Single_Class(minio_access_key=args.minio_access_key,
#                             minio_secret_key=args.minio_secret_key,
#                             dataset= args.dataset,
#                             class_name= model_1_name,
#                             model = None,
#                             save_name = args.save_name,
#                             class_id =  get_tag_id_by_name(args.class_name),
#                             training_batch_size=args.training_batch_size,
#                             num_samples= args.num_samples,
#                             epochs= args.epochs,
#                             learning_rate= args.learning_rate)

# #original_model = EBM_Single_Class(train_loader = None,val_loader = None, adv_loader = None,img_shape=(1280,))
# # Load the last occult trained model


# model_1.load_model_from_minio(minio_client, dataset_name = "environmental", tag_name =model_1_name, model_type = "energy-based-model")

# model_list.append(model_1)



# model_2_name =  "concept-cybernetic" # "concept-cybernetic"  #"topic-medieval"  # "topic-desert"  # "topic-desert"   #  "topic-forest"  # "perspective-3d" #"perspective-isometric" 
# model_2=EBM_Single_Class(minio_access_key=args.minio_access_key,
#                             minio_secret_key=args.minio_secret_key,
#                             dataset= args.dataset,
#                             class_name= model_2_name ,
#                             model = None,
#                             save_name = args.save_name,
#                             class_id =  get_tag_id_by_name(args.class_name),
#                             training_batch_size=args.training_batch_size,
#                             num_samples= args.num_samples,
#                             epochs= args.epochs,
#                             learning_rate= args.learning_rate)

# #original_model = EBM_Single_Class(train_loader = None,val_loader = None, adv_loader = None,img_shape=(1280,))
# # Load the last occult trained model


# model_2.load_model_from_minio(minio_client, dataset_name = "environmental", tag_name = model_2_name, model_type = "energy-based-model")

# model_list.append(model_2)

# tag_name_combined = f"{model_1_name}-and-{model_2_name}"

# plot_samples_hashless_combination(loaded_model_list = model_list, dataset_name = "environmental", number_of_samples = 30000,tag_name =tag_name_combined)



# tag_name = tag_name_combined
# sorted_images_and_hashes = plot_samples_graph_interpolation_plus_mapping_combined(loaded_model = model_1,
#                                                         loaded_model_2 = model_2,
#                                                           dataset_name = "environmental",
#                                                             number_of_samples = 10000 ,
#                                                             tag_name =tag_name_x,
#                                                               model_type = "EBM Model",
#                                                                tag_name_1=model_1_name,
#                                                                 tag_name_2= model_2_name)
# # rank = 1
# # for image in sorted_images_and_hashes:
# #     #
# #     print("Rank : ", rank, " Path : ", image["path"], " Score : ",image["score"])
# #     rank += 0
# # # Tag the images

# selected_structure_first_50 = sorted_images_and_hashes[:52] 
# selected_structure_second_50 = sorted_images_and_hashes[52:103]
# selected_structure_third_50 = sorted_images_and_hashes[103:154]

# tier4 = sorted_images_and_hashes[150:200] 
# tier5 = sorted_images_and_hashes[200:250]
# tier6 = sorted_images_and_hashes[250:300]
# tier7 = sorted_images_and_hashes[300:350] 
# tier8 = sorted_images_and_hashes[350:400]
# tier9 = sorted_images_and_hashes[400:450]



# tier10 = sorted_images_and_hashes[450:500] 
# tier11 = sorted_images_and_hashes[500:550]
# tier12 = sorted_images_and_hashes[550:750]
# tier13 = sorted_images_and_hashes[750:950]
# tier14 = sorted_images_and_hashes[950:1150]
# tier15 = sorted_images_and_hashes[1150:1350]

# #tag_image(file_hash,tag_id,user)


# plot_name1 = tag_name + "_tier1_hs"
# plot_name2 = tag_name + "_tier2_hs"
# plot_name3  = tag_name + "_tier3_hs"
# plot_name4 = tag_name + "_tier4_hs"
# plot_name5 = tag_name + "_tier5_hs"
# plot_name6  = tag_name + "_tier6_hs"
# plot_name7 = tag_name + "_tier7_hs"
# plot_name8  = tag_name + "_tier8_hs"
# plot_name9  = tag_name + "_tier9_hs"

# plot_name10 = tag_name + "_tier10_hs"
# plot_name11  = tag_name + "_tier11_hs"
# plot_name12  = tag_name + "_tier12_hs"


# plot_name13 = tag_name + "_tier13_hs"
# plot_name14  = tag_name + "_tier14_hs"
# plot_name15  = tag_name + "_tier15_hs"

# plot_images_with_scores_hasheless_v2(selected_structure_first_50,plot_name1)
# plot_images_with_scores_hasheless_v2(selected_structure_second_50,plot_name2)
# plot_images_with_scores_hasheless_v2(selected_structure_third_50,plot_name3)

# plot_images_with_scores_hasheless_v2(tier4,plot_name4)
# plot_images_with_scores_hasheless_v2(tier5,plot_name5)
# plot_images_with_scores_hasheless_v2(tier6,plot_name6)
# plot_images_with_scores_hasheless_v2(tier7,plot_name7)
# plot_images_with_scores_hasheless_v2(tier8,plot_name8)
# plot_images_with_scores_hasheless_v2(tier9,plot_name9)

# plot_images_with_scores_hasheless_v2(tier10,plot_name10)
# plot_images_with_scores_hasheless_v2(tier11,plot_name11)
# plot_images_with_scores_hasheless_v2(tier12,plot_name12)


# plot_images_with_scores_hasheless_v2(tier13,plot_name13)
# plot_images_with_scores_hasheless_v2(tier14,plot_name14)
# plot_images_with_scores_hasheless_v2(tier15,plot_name15)









####################################################


# class_names = get_unique_tag_names()
# all_tags = get_unique_tag_ids()
# print("all tags : ", all_tags )
# print("all tags length : ", len(all_tags) )
# target_data , ood_data = get_all_classes_paths(class_ids = all_tags,target_id=22)


# target_scores_EBM = []
# ood_scores_EBM = []

# target_scores_ELM = []
# ood_scores_ELM = []

# for target in target_data:
#     vector = get_clip_from_path(target)
#     ebm_score = (original_model.classify(vector).item())
#     elm_score = (elm_model.classify(vector)).item()
#     target_scores_EBM.append(ebm_score)
#     target_scores_ELM.append(elm_score)
#     print(f'The score for EBM is {ebm_score} , and the score of ELM is {elm_score}')


# for ood_image in ood_data:
#     vector = get_clip_from_path(ood_image)
#     ebm_score = (original_model.classify(vector).item())
#     elm_score = (elm_model.classify(vector)).item()
#     ood_scores_EBM.append(ebm_score)
#     ood_scores_ELM.append(elm_score)
#     print(f'The score for EBM is {ebm_score} , and the score of ELM is {elm_score}')



# print(f'Average EBM score is {statistics.mean(target_scores_EBM)} , and average ELM score is {statistics.mean(target_scores_ELM)}')
# print(f'Standard diviation EBM: {statistics.stdev(target_scores_EBM)} , ELM {statistics.stdev(target_scores_ELM)}')


# print(f'Average OOD EBM score is {statistics.mean(ood_scores_EBM)} , and average OOD ELM score is {statistics.mean(ood_scores_ELM)}')
# print(f'Standard diviation OOD EBM: {statistics.stdev(ood_scores_EBM)} , ELM {statistics.stdev(ood_scores_ELM)}')


















# new_sorted_images = process_and_sort_dataset(ood_data, original_model)



# selected_structure_first_50 = new_sorted_images[:52]
# selected_structure_second_50 = new_sorted_images[52:103]
# selected_structure_third_50 = new_sorted_images[103:154]
# model_name = "EBM-Aquatic"

# plot_name1 = model_name + "_tier1"
# plot_name2 = model_name + "_tier2"
# plot_name3  = model_name + "_tier3"

# plot_images_with_scores_hasheless(selected_structure_first_50,plot_name1)
# plot_images_with_scores_hasheless(selected_structure_second_50,plot_name2)
# plot_images_with_scores_hasheless(selected_structure_third_50,plot_name3)