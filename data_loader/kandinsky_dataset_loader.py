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


    def load_kandinsky_jobs(self):
        print(f"Fetching kandinsky jobs for the {self.dataset} dataset")
        response = requests.get(f'{API_URL}/queue/image-generation/list-completed-by-dataset-and-task-type?dataset={self.dataset}&task_type=img2img_generation_kandinsky')
            
        jobs = json.loads(response.content)

        return jobs
    
    def load_clip_vector_data(self):
        jobs= self.load_kandinsky_jobs()
        feature_vectors=[]
        scores=[]

        index=0
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

                output_clip_score = self.ranking_model.predict_clip(output_clip_vector).item()
                image_clip_sigma_score = (output_clip_score - self.score_mean) / self.score_std 

                feature_vectors.append(input_clip_vector)
                scores.append(image_clip_sigma_score)

            except:
                print("An error occured")
            
            index+=1

            if index>50:
                break
        
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

        