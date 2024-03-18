import argparse
from datetime import datetime
import io
import json
import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.optim as optim
import msgpack
from PIL import Image
import time
import random
from tqdm import tqdm

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from kandinsky.model_paths import PRIOR_MODEL_PATH
from transformers import CLIPImageProcessor
from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from training_worker.scoring.models.scoring_fc import ScoringFCNetwork
from kandinsky.models.kandisky import KandinskyPipeline
from utility.path import separate_bucket_and_file_path
from utility.minio import cmd
from data_loader.utils import get_object
from torch.nn.functional import cosine_similarity 
from training_worker.scoring.models.scoring_xgboost import ScoringXgboostModel
from training_worker.scoring.models.scoring_treeconnect import ScoringTreeConnectNetwork


DATA_MINIO_DIRECTORY="data/latent-generator"
API_URL = "http://192.168.3.1:8111"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")
    parser.add_argument('--model-type', type=str, help='model type, fc or xgboost', default="fc")
    parser.add_argument('--kandinsky-batch-size', type=int, default=5)
    parser.add_argument('--training-batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--construct-dataset', action='store_true', default=False)
    parser.add_argument('--num-samples', type=int, default=10000)

    return parser.parse_args()

class ABRankingFcTrainingPipeline:
    def __init__(self,
                    minio_access_key,
                    minio_secret_key,
                    dataset,
                    model_type,
                    kandinsky_batch_size=5,
                    training_batch_size=64,
                    num_samples=10000,
                    learning_rate=0.001,
                    epochs=10):
        
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
        self.training_batch_size= training_batch_size
        self.kandinsky_batch_size= kandinsky_batch_size
        self.num_samples= num_samples
        self.learning_rate= learning_rate
        self.epochs= epochs
        self.model_type= model_type

        if(self.model_type=="fc"):
            self.model= ScoringFCNetwork(minio_client=self.minio_client, dataset=dataset)
        elif(self.model_type=="xgboost"):
            self.model= ScoringXgboostModel(minio_client=self.minio_client, dataset=dataset)
        elif(self.model_type=="treeconnect"):
            self.model= ScoringTreeConnectNetwork(minio_client=self.minio_client, dataset=dataset)

        # load kandinsky clip
        self.image_processor= CLIPImageProcessor.from_pretrained(PRIOR_MODEL_PATH, subfolder="image_processor", local_files_only=True)

        # load kandinsky's autoencoder
        self.image_generator= KandinskyPipeline(device= self.device, strength=0.75, decoder_guidance_scale=12,
                                                decoder_steps=50)
        self.image_generator.load_models(task_type="img2img")

        self.image_encoder= self.image_generator.image_encoder

        # load scoring model
        self.scoring_model= self.load_scoring_model()
        self.mean= float(self.scoring_model.mean)
        self.std= float(self.scoring_model.standard_deviation)

        self.clip_mean , self.clip_std, self.clip_max, self.clip_min= self.get_clip_distribution()
        print(self.clip_mean, self.clip_std)

    def get_clip_distribution(self):
        data = get_object(self.minio_client, "environmental/output/stats/clip_stats.msgpack")
        data_dict = msgpack.unpackb(data)

        mean_vector = torch.tensor(data_dict["mean"]).to(device=self.device)
        std_vector = torch.tensor(data_dict["std"]).to(device=self.device)
        max_vector = torch.tensor(data_dict["max"]).to(device=self.device)
        min_vector = torch.tensor(data_dict["min"]).to(device=self.device)

        return mean_vector, std_vector, max_vector, min_vector

    # load elm or linear scoring models
    def load_scoring_model(self):
        input_path=f"{self.dataset}/models/ranking/"

        scoring_model = ABRankingELMModel(1280)
        file_name=f"score-elm-v1-clip-h.safetensors"

        model_files=cmd.get_list_of_objects_with_prefix(self.minio_client, 'datasets', input_path)
        most_recent_model = None

        for model_file in model_files:
            if model_file.endswith(file_name):
                most_recent_model = model_file

        if most_recent_model:
            model_file_data =cmd.get_file_from_minio(self.minio_client, 'datasets', most_recent_model)
        else:
            print("No .safetensors files found in the list.")
            return

        print(most_recent_model)

        # Create a BytesIO object and write the downloaded content into it
        byte_buffer = io.BytesIO()
        for data in model_file_data.stream(amt=8192):
            byte_buffer.write(data)
        # Reset the buffer's position to the beginning
        byte_buffer.seek(0)

        scoring_model.load_safetensors(byte_buffer)
        scoring_model.model=scoring_model.model.to(torch.device(self.device))

        return scoring_model
    
    def sample_random_latents(self):
        sampled_embeddings = torch.normal(mean=self.clip_mean.repeat(self.num_samples, 1),
                                      std=self.clip_std.repeat(self.num_samples, 1))
    
        latents=[]
        for embed in sampled_embeddings:
            latents.append(embed.unsqueeze(0))

        return latents
    
    def get_image_features(self, image):
        # Preprocess image
        if isinstance(image, Image.Image) or isinstance(image, list):
            image = self.image_processor(image, return_tensors="pt")['pixel_values']
        
         # Compute CLIP features
        if isinstance(image, torch.Tensor) or isinstance(image, list):
            with torch.no_grad():
                features = self.image_encoder(pixel_values= image.half().to(self.device)).image_embeds
        else:
            raise ValueError(
                f"`image` can only contains elements to be of type `PIL.Image.Image` or `torch.Tensor`  but is {type(image)}"
            )
        
        return features

    def construct_dataset(self):
        # generate latents
        latents= self.load_samples_from_minio()
        training_data=[]
        init_image_batch = [Image.open("./test/test_inpainting/white_512x512.jpg") for i in range(self.kandinsky_batch_size)]

        for i in range(0, len(latents), self.kandinsky_batch_size):

            try:
                # Prepare the batch
                latent_batch = latents[i:i + self.kandinsky_batch_size]
                latent_batch_tensor = torch.stack(latent_batch).to(self.device).half()  # Ensure correct device and dtype
                
                # Process batch through image generator and feature extraction
                # Adjust generate_img2img and get_image_features to accept and return batches
                output_images, _ = self.image_generator.generate_img2img_in_batches(init_imgs=init_image_batch,
                                                                    image_embeds=latent_batch_tensor.squeeze(1),
                                                                    batch_size= self.kandinsky_batch_size)
                clip_vectors = self.get_image_features(output_images).float()  # Assuming this returns a batch of vectors

                # Iterate through the batch for scoring (since scoring model processes one item at a time)
                for j, (latent, clip_vector) in enumerate(zip(latent_batch_tensor, clip_vectors)):
                    input_clip_score = self.scoring_model.predict_clip(latent.float()).item()  # Convert back if necessary
                    image_score = self.scoring_model.predict_clip(clip_vector.unsqueeze(0)).item()
                    input_clip_score= (input_clip_score - self.mean) / self.std
                    image_score= (image_score - self.mean) / self.std
                    cosine_sim =cosine_similarity(clip_vector.unsqueeze(0), latent).item()

                    data = {
                        'input_clip': latent.detach().cpu().numpy().tolist(),
                        'output_clip': clip_vector.unsqueeze(0).detach().cpu().numpy().tolist(),
                        'input_clip_score': input_clip_score,
                        'output_clip_score': image_score,
                        'cosine_sim': cosine_sim
                    }

                    training_data.append(data)
            except:
                print("an error occured")

        self.store_training_data(training_data)

    def get_file_paths(self):
        print('Loading image file paths')
        response = requests.get(f'{API_URL}/queue/image-generation/list-by-dataset?dataset={self.dataset}&model_type=elm-v1&min_clip_sigma_score=0&size={self.num_samples}')
        
        jobs = json.loads(response.content)

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

    def train(self):
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
        
        # training and saving the model
        if self.model_type in ["fc", "treeconnect"]:
            loss=self.model.train(inputs, outputs, num_epochs= self.epochs, batch_size=self.training_batch_size, learning_rate=self.learning_rate)
        elif self.model_type=="xgboost":
            loss=self.model.train(inputs, outputs)
        self.model.save_model()
    
    def load_self_training_data(self, data):
        inputs=[]
        outputs=[]
        for d in data:
            inputs.append(d['input_clip'][0])
            outputs.append(d['output_clip_score'])
        
        return inputs, outputs

    # store self training data
    def store_training_data(self, training_data):
        batch_size = 10000
        dataset_path = DATA_MINIO_DIRECTORY + "/self_training/"
        dataset_files = self.minio_client.list_objects('datasets', prefix=dataset_path, recursive=True)
        dataset_files = [file.object_name for file in dataset_files]

        batch = []  # Accumulate training data points until the batch size is reached

        if(len(dataset_files)==0):
            index=1
        else:
            last_file_path=dataset_files[len(dataset_files)-1]
            # Read the content of the last unfinished file
            if last_file_path.endswith("_incomplete.msgpack"):
                data = self.minio_client.get_object('datasets', last_file_path)
                content = data.read()
                batch = msgpack.loads(content)
                index = len(dataset_files)
                self.minio_client.remove_object('datasets', last_file_path)
            else:
                index= len(dataset_files) + 1

        for data in training_data:
            batch.append(data)

            if len(batch) == batch_size:
                self.store_batch_in_msgpack_file(batch, index)
                index += 1
                batch = []  # Reset the batch for the next file

        # If there are remaining data points not reaching the batch size, store them
        if batch:
            self.store_batch_in_msgpack_file(batch, index, incomplete=True)

    # function for storing self training data in a msgpack file
    def store_batch_in_msgpack_file(self, batch, index, incomplete=False):
        if incomplete:
            file_path=f"{str(index).zfill(4)}_incomplete.msgpack"
        else:
            file_path=f"{str(index).zfill(4)}.msgpack"
        packed_data = msgpack.packb(batch, use_single_float=True)

        local_file_path = f"output/temporary_file.msgpack"
        with open(local_file_path, 'wb') as local_file:
            local_file.write(packed_data)

        with open(local_file_path, 'rb') as file:
            content = file.read()

        buffer = io.BytesIO(content)
        buffer.seek(0)

        minio_path = DATA_MINIO_DIRECTORY + f"/self_training/{file_path}"
        cmd.upload_data(self.minio_client, 'datasets', minio_path, buffer)

        os.remove(local_file_path)


def main():
    args = parse_args()

    training_pipeline=ABRankingFcTrainingPipeline(minio_access_key=args.minio_access_key,
                                minio_secret_key=args.minio_secret_key,
                                dataset= args.dataset,
                                model_type=args.model_type,
                                kandinsky_batch_size=args.kandinsky_batch_size,
                                training_batch_size=args.training_batch_size,
                                num_samples= args.num_samples,
                                epochs= args.epochs,
                                learning_rate= args.learning_rate)
    
    global DATA_MINIO_DIRECTORY
    DATA_MINIO_DIRECTORY= f"{args.dataset}/" + DATA_MINIO_DIRECTORY

    if args.construct_dataset:
        training_pipeline.construct_dataset()
    
    # do self training
    training_pipeline.train()

if __name__ == "__main__":
    main()

            