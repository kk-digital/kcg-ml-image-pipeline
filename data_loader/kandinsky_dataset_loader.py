import io
import os
import sys
import requests
import json
import torch
import msgpack
from tqdm import tqdm
import argparse

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from data_loader.utils import get_object
from kandinsky_worker.dataloaders.image_embedding import ImageEmbedding
from utility.minio import cmd
from utility.path import separate_bucket_and_file_path
from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from training_worker.classifiers.models.elm_regression import ELMRegression
from training_worker.ab_ranking.model import constants

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
                 dataset):
        
        # get minio client
        self.minio_client = minio_client
        self.dataset= dataset

        # get device
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = torch.device(device)

        # get the ranking model
        self.ranking_model= self.load_scoring_model()
        self.score_mean= float(self.ranking_model.mean)
        self.score_std= float(self.ranking_model.standard_deviation)
        
    
    def load_scoring_model(self):
        prefix=f"{self.dataset}/models/ranking/"
        suffix=f"score-elm-v1-clip-h.safetensors"

        ranking_model = ABRankingELMModel(1280, device=self.device)

        model_files=cmd.get_list_of_objects_with_prefix(self.minio_client, 'datasets', prefix)
        most_recent_model = None

        for model_file in model_files:
            if model_file.endswith(suffix):
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

        ranking_model.load_safetensors(byte_buffer)

        return ranking_model
    
    def get_classifier_model(self, tag_name, input_type):
        input_path = f"{self.dataset}/models/classifiers/{tag_name}/"
        file_suffix = f"elm-regression-{input_type}.safetensors"

        # Use the MinIO client's list_objects method directly with recursive=True
        model_files = [obj.object_name for obj in self.minio_client.list_objects('datasets', prefix=input_path, recursive=True) if obj.object_name.endswith(file_suffix)]
        
        if not model_files:
            print(f"No .safetensors models found for tag: {tag_name}")
            return None

        # Assuming there's only one model per tag or choosing the first one
        model_files.sort(reverse=True)
        model_file = model_files[0]
        print(f"Loading model: {model_file}")

        return self.load_model_with_filename(self.minio_client, model_file, tag_name)

    def load_model_with_filename(self, minio_client, model_file, model_info=None):
        model_data = minio_client.get_object('datasets', model_file)
        
        clip_model = ELMRegression(device=self.device)
        
        # Create a BytesIO object from the model data
        byte_buffer = io.BytesIO(model_data.data)
        clip_model.load_safetensors(byte_buffer)

        print(f"Model loaded for tag: {model_info}")
        
        return clip_model


    def load_kandinsky_jobs(self):
        print(f"Fetching kandinsky jobs for the {self.dataset} dataset")
        response = requests.get(f'{API_URL}/queue/image-generation/list-completed-by-dataset-and-task-type?dataset={self.dataset}&task_type=img2img_generation_kandinsky')
            
        jobs = json.loads(response.content)

        return jobs
    
    def load_clip_vector_data(self, limit=-1):
        jobs= self.load_kandinsky_jobs()
        jobs= jobs[:limit]
        feature_vectors=[]
        scores=[]
        
        print("Loading input clip vectors and sigma scores for each job")
        for job in tqdm(jobs):
            try:
                file_path= job['file_path']
                bucket_name, input_file_path = separate_bucket_and_file_path(file_path)
                file_path = os.path.splitext(input_file_path)[0]

                input_clip_path = file_path + "_embedding.msgpack"
                print("input_file_path = " + input_clip_path)
                clip_data = get_object(self.minio_client, input_clip_path)
                embedding_dict = ImageEmbedding.from_msgpack_bytes(clip_data)
                input_clip_vector= embedding_dict.image_embedding
                input_clip_vector= input_clip_vector[0].cpu().numpy().tolist()

                output_clip_path = file_path + "_clip_kandinsky.msgpack"
                features_data = get_object(self.minio_client, output_clip_path)
                features_vector = msgpack.unpackb(features_data)["clip-feature-vector"]
                output_clip_vector= torch.tensor(features_vector).to(device=self.device)

                output_clip_score = self.ranking_model.predict_clip(output_clip_vector).item()
                image_clip_sigma_score = (output_clip_score - self.score_mean) / self.score_std 

                feature_vectors.append(input_clip_vector)
                scores.append(image_clip_sigma_score)

            except:
                print("An error occured")
        
        return feature_vectors, scores
    

    def load_latents(self, limit=-1):
        jobs= self.load_kandinsky_jobs()
        jobs= jobs[:limit]
        latents= []

        for job in tqdm(jobs):

            try:
                file_path= job['file_path']
                bucket_name, input_file_path = separate_bucket_and_file_path(file_path)
                file_path = os.path.splitext(input_file_path)[0]
                input_vae_path = file_path + "_vae_vae_latent.msgpack"

                features_data = get_object(self.minio_client, input_vae_path)
                print("------------------->")
                features = msgpack.unpackb(features_data)["latent_vector"]
                print(features)
                features = torch.tensor(features).to(device=self.device)
                latents.append(features)
            except Exception as e:
                print(f"Error processing clip at path {input_vae_path}: {e}")

        return latents, 
    
    def load_classifier_scores(self, tag_name, batch_size, input_type=constants.KANDINSKY_CLIP):
        classifier= self.get_classifier_model(tag_name, input_type)
        jobs= self.load_kandinsky_jobs()
        feature_vectors=[]
        scores=[]

        input_batch=[]
        output_batch=[]

        print("Loading input clip vectors and sigma scores for each job")
        for job in tqdm(jobs):
            try:
                file_path= job['file_path']
                bucket_name, input_file_path = separate_bucket_and_file_path(file_path)
                file_path = os.path.splitext(input_file_path)[0]

                input_clip_path = file_path + "_embedding.msgpack"
                clip_data = get_object(self.minio_client, input_clip_path)
                embedding_dict = ImageEmbedding.from_msgpack_bytes(clip_data)
                input_clip_vector= embedding_dict.image_embedding
                input_clip_vector= input_clip_vector[0].cpu().numpy().tolist()

                output_clip_path = file_path + "_clip_kandinsky.msgpack"
                features_data = get_object(self.minio_client, output_clip_path)
                features_vector = msgpack.unpackb(features_data)["clip-feature-vector"]
                output_clip_vector= torch.tensor(features_vector).to(device=self.device)

                input_batch.append(input_clip_vector)
                output_batch.append(output_clip_vector)
                if len(input_batch) == batch_size:
                    output_features= torch.stack(output_batch, dim=0).squeeze(1).to(device=self.device)
                    # add vector length if input type includes it
                    if input_type== constants.KANDINSKY_CLIP_WITH_LENGTH:
                        vector_length= torch.norm(output_features, dim=1).unsqueeze(1)
                        output_features= torch.cat([output_features , vector_length], dim=1)

                    output_clip_scores = classifier.classify(output_features).squeeze(1).tolist()
                    scores.extend(output_clip_scores)
                    feature_vectors.extend(input_batch)
                    input_batch=[]
                    output_batch=[]

            except Exception as e:
                print(f"An error occured {e}")

        return feature_vectors, scores

def main():
    args= parse_args()

    # get minio client
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key)

    dataloader= KandinskyDatasetLoader(minio_client=minio_client,
                                       dataset=args.dataset)
    
    dataset= dataloader.load_clip_vector_data()

    print(f"Number of datapoints loaded: {len(dataset)}")
    print(f"{dataset[0]}")


if __name__ == "__main__":
    main()

        