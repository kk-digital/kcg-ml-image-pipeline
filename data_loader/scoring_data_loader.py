import torch
import tqdm
import requests
import msgpack
import sys
import os
import json

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from data_loader.utils import get_object
from utility.path import separate_bucket_and_file_path
from utility.http import generation_request

DATA_MINIO_DIRECTORY="data/latent-generator"
API_URL = "http://192.168.3.1:8111"

class ScoringDataLoader:


    def __init__(self, minio_client, device:str = "cuda", dataset:str = "environmental", num_samples:int = 10000) -> None:
        self.minio_client = minio_client
        self.device = device
        self.dataset = dataset
        self.num_samples = num_samples

        self.clip_distribution = self.get_clip_distribution()
        self.samples = self.load_samples_from_minio()
        self.X_train, self.y_train = self.get_train_data()


    def get_clip_distribution(self):
        data = get_object(self.minio_client, "{}/output/stats/clip_stats.msgpack".format(self.dataset))
        data_dict = msgpack.unpackb(data)

        mean_vector = torch.tensor(data_dict["mean"]).to(device=self.device)
        std_vector = torch.tensor(data_dict["std"]).to(device=self.device)
        max_vector = torch.tensor(data_dict["max"]).to(device=self.device)
        min_vector = torch.tensor(data_dict["min"]).to(device=self.device)

        return mean_vector, std_vector, max_vector, min_vector
    
    def get_file_paths(self):
        print('Loading image file paths')

        jobs = generation_request.http_get_list_by_dataset(dataset=self.dataset, model_type="elm-v1", min_clip_sigma_score=0, num_samples=self.num_samples)

        file_paths=[job['file_path'] for job in jobs]

        return file_paths
    
    def load_samples_from_minio(self):
        file_paths= self.get_file_paths()

        print(len(file_paths))

        latents=[]
        missing=0
        for path in tqdm(file_paths):
            try:
                clip_path= path.replace('.jpg', '_clip_kandinsky.msgpack')
                bucket, features_vector_path= separate_bucket_and_file_path(clip_path) 
                features_data = get_object(self.minio_client, features_vector_path)
                features_vector = msgpack.unpackb(features_data)["clip-feature-vector"]
                features_vector= torch.tensor(features_vector).to(device=self.device, dtype=torch.float32)
                
                latents.append(features_vector)
            except:
                missing+=1
        
        print(missing)

        return latents

    def get_train_data(self):
        inputs=[]
        outputs=[]

        # get self training data
        self_training_path = DATA_MINIO_DIRECTORY + f"/self_training/"
        self_training_files = self.minio_client.list_objects('datasets', prefix=self_training_path, recursive=True)
        self_training_files = [file.object_name for file in self_training_files]

        for file in self_training_files:
            print(file)

            # get data
            data = self.minio_client.get_object('datasets', file)
            # Read the content of the msgpack file
            content = data.read()

            # Deserialize the content using msgpack
            self_training_data = msgpack.loads(content)
            
            # append the self training data to list of data
            self_training_inputs, self_training_outputs= self.load_self_training_data(self_training_data)
            inputs.extend(self_training_inputs)
            outputs.extend(self_training_outputs)
        return inputs, outputs
    
    def load_self_training_data(self, data):
        inputs=[]
        outputs=[]
        for d in data:
            inputs.append(d['input_clip'][0])
            outputs.append(d['output_clip_score'])
        
        return inputs, outputs