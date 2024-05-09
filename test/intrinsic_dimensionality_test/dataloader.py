import os
import sys
import requests
import json
import torch
import msgpack
from tqdm import tqdm
import argparse
import numpy as np
base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from test.intrinsic_dimensionality_test.utils import get_object
from test.intrinsic_dimensionality_test.utils import separate_bucket_and_file_path

API_URL="http://192.168.3.1:8111"

def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--minio-access-key', type=str, help='Minio access key')
        parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
        parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")

        return parser.parse_args()

class KandinskyDatasetLoader:
    def __init__(self,
                 minio_client, 
                 dataset= "environmental"):
        
        # get minio client
        self.minio_client = minio_client
        self.dataset= dataset

        # get device
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = torch.device(device)

    def load_kandinsky_jobs(self):
        print(f"Fetching kandinsky jobs for the {self.dataset} dataset")
        response = requests.get(f'{API_URL}/queue/image-generation/list-completed-by-dataset-and-task-type?dataset={self.dataset}&task_type=img2img_generation_kandinsky')
        jobs = json.loads(response.content)
        return jobs
    
    def decode_ndarray(self, packed_obj):
        if '__ndarray__' in packed_obj:
            return np.array(packed_obj['__ndarray__'])
        return packed_obj
    
    def load_clip_vector_data(self, limit=-1):
        jobs= self.load_kandinsky_jobs()
        jobs= jobs[:limit]
        feature_vectors=[]
        
        print("Loading input clip vectors and sigma scores for each job")
        for job in tqdm(jobs):
            try:
                file_path= job['file_path']
                _, input_file_path = separate_bucket_and_file_path(file_path)
                file_path = os.path.splitext(input_file_path)[0]
                input_clip_path = file_path + "_embedding.msgpack"
                clip_data = get_object(self.minio_client, input_clip_path)
                input_clip_vector = msgpack.unpackb(clip_data, object_hook=self.decode_ndarray, raw=False)["image_embedding"]
                feature_vectors.append(input_clip_vector)
            except Exception as e:
                print("An error occured", e)
        
        return feature_vectors
    

    def load_latents(self, limit=-1):
        jobs= self.load_kandinsky_jobs()
        jobs= jobs[:limit]
        latents= []

        for job in tqdm(jobs):

            try:
                file_path= job['file_path']
                _, input_file_path = separate_bucket_and_file_path(file_path)
                file_path = os.path.splitext(input_file_path)[0]
                input_vae_path = file_path + "_vae_vae_latent.msgpack"
                features_data = get_object(self.minio_client, input_vae_path)
                features = msgpack.unpackb(features_data)["latent_vector"]
                
                latents.append(features)
            except Exception as e:
                print(f"Error processing clip at path {input_vae_path}: {e}")
        print(len(latents))
        
        return latents
    