from PIL import Image
import numpy as np
import requests
from io import BytesIO
import math
import matplotlib.pyplot as plt

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

import statistics

API_URL = "http://192.168.3.1:8111"
date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%d-%m-%Y %H:%M:%S')

minio_client = cmd.get_minio_client("D6ybtPLyUrca5IdZfCIM",
            "2LZ6pqIGOiZGcjPTR6DZPlElWBkRTkaLkyLIBt4V",
            None)

from utility.path import separate_bucket_and_file_path
from data_loader.utils import get_object
from utility.http import request
import json

def luminance_standard(image_path):
    # response = requests.get(image_path)
    # image = Image.open(BytesIO(response.content))
    image = get_image(image_path)
    image = np.asarray(image) / 255.0  # Normalize pixel values to [0, 1]
    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
    luminance = 0.2126 * R + 0.7152 * G + 0.0722 * B
    return np.mean(luminance)

def luminance_perceived_v1(image_path):
    # response = requests.get(image_path)
    # image = Image.open(BytesIO(response.content))
    image = get_image(image_path)
    image = np.asarray(image) / 255.0  # Normalize pixel values to [0, 1]
    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
    luminance = 0.299 * R + 0.587 * G + 0.114 * B
    return  np.mean(luminance)

def luminance_perceived_v2(image_path,power_curve_factor):
    # response = requests.get(image_path)
    # image = Image.open(BytesIO(response.content))
    image = get_image(image_path)
    image = np.asarray(image) / 255.0  # Normalize pixel values to [0, 1]
    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
    luminance = np.sqrt(0.299 * R**power_curve_factor + 0.587 * G**power_curve_factor + 0.114 * B**power_curve_factor)
    return  np.mean(luminance)

def average_luminance(image_path, power_curve_factor, weight_for_standard_luminance = 1, weight_for_perceived_luminance_v1 = 1, weight_for_perceived_luminance_v2 = 1 ):

    return ( (luminance_standard(image_path) * weight_for_standard_luminance) + 
            (luminance_perceived_v1(image_path)*weight_for_perceived_luminance_v1) +
             (luminance_perceived_v2(image_path,power_curve_factor)*weight_for_perceived_luminance_v2)
             )/ ( weight_for_standard_luminance + weight_for_perceived_luminance_v1 + weight_for_perceived_luminance_v2 ) 



def bin_images_by_brightness(image_urls, power_curve_factor=2, num_bins=10, weight_for_standard_luminance=1, weight_for_perceived_luminance_v1=1, weight_for_perceived_luminance_v2=1):
    # Calculate average luminance for each image
    luminance_values = []
    for url in image_urls:
        luminance = average_luminance(url, power_curve_factor, weight_for_standard_luminance, weight_for_perceived_luminance_v1, weight_for_perceived_luminance_v2)
        luminance_values.append((url, luminance))
    
    # Sort images by their luminance values
    luminance_values.sort(key=lambda x: x[1])
    
    # Divide images into bins
    bin_size = len(luminance_values) // num_bins
    bins = []
    for i in range(num_bins):
        bin_start = i * bin_size
        bin_end = (i + 1) * bin_size if i != num_bins - 1 else len(luminance_values)
        bins.append(luminance_values[bin_start:bin_end])
    
    return bins



# Images utilities

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




# Try it for dictionary
def plot_images_with_scores_hasheless_v2(bin,name):
    minio_client = cmd.get_minio_client("D6ybtPLyUrca5IdZfCIM",
            "2LZ6pqIGOiZGcjPTR6DZPlElWBkRTkaLkyLIBt4V",
            None)
    # Number of images
    num_images = len(bin)
    
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
    for element in bin:
        image = get_image(element[0])
        score = element[1]
        

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


images_paths, _  = get_all_tag_jobs(class_ids = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], target_id =15)

bins = bin_images_by_brightness(images_paths,
                                 power_curve_factor=1.8,
                                   num_bins=10,
                                     weight_for_standard_luminance=1,
                                       weight_for_perceived_luminance_v1=1,
                                         weight_for_perceived_luminance_v2=1)


plot_images_with_scores_hasheless_v2(bins[0],"test")