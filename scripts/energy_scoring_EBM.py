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

# ------------------------------------------------- Parameters BIS -------------------------------------------------
base_directory = "./"
sys.path.insert(0, base_directory)

from utility.path import separate_bucket_and_file_path
from data_loader.utils import get_object
from utility.http import request

API_URL = "http://192.168.3.1:8111"
minio_client = cmd.get_minio_client("D6ybtPLyUrca5IdZfCIM", "2LZ6pqIGOiZGcjPTR6DZPlElWBkRTkaLkyLIBt4V",None)

date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%d-%m-%Y %H:%M:%S')


# ------------------------------------------------- Import function from EBM  -------------------------------------------------
from training_worker.classifiers.models import ebm_single_class


#/image/list-image-metadata-by-dataset-v1?dataset=environmental&limit=20&offset=0&order=desc&time_unit=minutes

def get_all_non_tagged_jobs(class_ids,dataset_name):
    all_data = {}  # Dictionary to store data for all class IDs
    
    for class_id in class_ids:
        response = requests.get(f'{API_URL}/image/list-image-metadata-by-dataset-v1?dataset={dataset_name}')
        
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
    

    return all_data




def get_file_paths_and_hashes_uuid(dataset,num_samples):
        print('Loading image file paths')
        response = requests.get(f'{API_URL}/image/list-image-metadata-by-dataset-v1?dataset={dataset}&limit={num_samples}')
        
        jobs = json.loads(response.content)
        #print(jobs)
        file_paths=[job['image_path'] for job in jobs]
        hashes=[job['image_hash'] for job in jobs]
        uuid =[job['job_uuid'] for job in jobs] 
        #image_hashes=[job['image_hash'] for job in jobs]

        for i in  range(len(file_paths)):
            print("Path : ", file_paths[i], " Hash : ", hashes[i], " UUID : ",uuid[i])




# /classifier-scores/set-image-classifier-score
# {
#   "uuid": "string",
#   "classifier_id": 0,
#   "image_hash": "string",
#   "tag_id": 0,
#   "score": 0
# }

def score_images_based_on_energy():
    # load data

    # load model

    # score each image
    return True



get_file_paths_and_hashes_uuid("environmental",100)