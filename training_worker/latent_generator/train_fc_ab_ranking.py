import argparse
from datetime import datetime
import io
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import msgpack
from PIL import Image
import time
import random

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from kandinsky.model_paths import PRIOR_MODEL_PATH
from transformers import CLIPImageProcessor
from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from training_worker.ab_ranking.model.ab_ranking_linear import ABRankingModel
from training_worker.ab_ranking.model.ab_ranking_fc import ABRankingFCNetwork
from kandinsky.models.kandisky import KandinskyPipeline
from utility.minio import cmd
from data_loader.utils import get_object


DATA_MINIO_DIRECTORY="data/latent-generator"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")
    parser.add_argument('--model-type', type=str, help='model type, linear or elm', default="linear")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--construct-dataset', action='store_true', default=False)

    return parser.parse_args()

class ABRankingFcTrainingPipeline:
    def __init__(self,
                    minio_access_key,
                    minio_secret_key,
                    dataset,
                    model_type,
                    batch_size=32):
        
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
        self.model_type= model_type
        self.batch_size= batch_size

        self.model= ABRankingFCNetwork(minio_client=self.minio_client)

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

        if(self.model_type=="elm"):
            scoring_model = ABRankingELMModel(1280)
            file_name=f"score-elm-v1-kandinsky-clip.safetensors"
        else:
            scoring_model= ABRankingModel(1280)
            file_name=f"score-linear-kandinsky-clip.safetensors"

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
    

    def sample_random_latents(self, num_samples=100000):
        # Calculate the range (max - min) for each dimension
        range = self.clip_max - self.clip_min
    
        # Generate random values in the range [0, 1], then scale and shift them to the [min, max] range
        sampled_embeddings = torch.randn(num_samples, *self.clip_mean.shape, device=self.clip_mean.device) * range + self.clip_min
            
        latents=[]
        for embed in sampled_embeddings:
            latents.append(embed)
        
        return latents
    
    def get_image_features(self, image):
        # Preprocess image
        if isinstance(image, Image.Image, list[Image.Image]):
            image = self.image_processor(image, return_tensors="pt")['pixel_values']
        
         # Compute CLIP features
        if isinstance(image, torch.Tensor, list[torch.Tensor]):
            with torch.no_grad():
                features = self.image_encoder(pixel_values= image.half().to(self.device)).image_embeds
        else:
            raise ValueError(
                f"`image` can only contains elements to be of type `PIL.Image.Image` or `torch.Tensor`  but is {type(image)}"
            )
        
        return features

    def construct_dataset(self, num_samples=10000):
        # generate latents
        latents= self.sample_random_latents(num_samples=num_samples)
        training_data=[]
        init_image_batch = [Image.open("./test/test_inpainting/white_512x512.jpg") for i in range(self.batch_size)]

        for i in range(0, len(latents), self.batch_size):
            # Prepare the batch
            latent_batch = latents[i:i + self.batch_size]
            latent_batch_tensor = torch.stack(latent_batch).to(self.device).half()  # Ensure correct device and dtype
            
            print(latent_batch_tensor.shape)
            # Process batch through image generator and feature extraction
            # Adjust generate_img2img and get_image_features to accept and return batches
            output_images, _ = self.image_generator.generate_img2img_in_batches(init_imgs=init_image_batch,
                                                                image_embeds=latent_batch_tensor.squeeze(0),
                                                                batch_size= self.batch_size)
            clip_vectors = self.get_image_features(output_images).float()  # Assuming this returns a batch of vectors
            
            # Iterate through the batch for scoring (since scoring model processes one item at a time)
            for j, (latent, clip_vector) in enumerate(zip(latent_batch_tensor, clip_vectors)):
                input_clip_score = self.scoring_model.predict_clip(latent.float()).item()  # Convert back if necessary
                image_score = self.scoring_model.predict_clip(clip_vector).item()

                data = {
                    'input_clip': latent.detach().cpu().numpy().tolist(),
                    'input_clip_score': input_clip_score,
                    'output_clip_score': image_score,
                }

                training_data.append(data)

        self.store_training_data(training_data)

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
        loss=self.model.train(inputs, outputs)
        self.model.save_model()
    
    def load_self_training_data(self, data):
        inputs=[]
        outputs=[]
        for d in data:
            inputs.append(d['input_clip'])
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
            file_path=f"/{str(index).zfill(4)}_incomplete.msgpack"
        else:
            file_path=f"/{str(index).zfill(4)}.msgpack"
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
                                batch_size=args.batch_size)
    
    if args.construct_dataset:
        training_pipeline.construct_dataset()
    
    # do self training
    training_pipeline.train()

if __name__ == "__main__":
    main()

            