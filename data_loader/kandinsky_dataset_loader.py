import os
import sys
import requests
import json
from tqdm import tqdm
import argparse

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from data_loader.utils import get_object
from kandinsky_worker.dataloaders.image_embedding import ImageEmbedding
from utility.minio import cmd
from utility.path import separate_bucket_and_file_path

API_URL="http://192.168.3.1:8111"

def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--minio-access-key', type=str, help='Minio access key')
        parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
        parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")

        return parser.parse_args()

class KandinskyDatasetLoader:
    def __init__(self,
                 minio_access_key,
                 minio_secret_key, 
                 dataset):
        
        # get minio client
        self.minio_client = cmd.get_minio_client(minio_access_key=minio_access_key,
                                            minio_secret_key=minio_secret_key)
        self.dataset= dataset

    def load_kandinsky_jobs(self):
        print(f"Fetching kandinsky jobs for the {self.dataset} dataset")
        response = requests.get(f'{API_URL}/queue/image-generation/list-completed-by-dataset-and-task-type?dataset={self.dataset}&task_type=img2img_generation_kandinsky')
            
        jobs = json.loads(response.content)

        return jobs
    
    def load_clip_vector_data(self):
        jobs= self.load_kandinsky_jobs()
        clip_vectors=[]
        
        print("Loading input clip vectors and sigma scores for each job")
        for job in tqdm(jobs):
            try:
                file_path= job['file_path']
                bucket_name, input_file_path = separate_bucket_and_file_path(file_path)
                file_path = os.path.splitext(input_file_path)[0]

                input_clip_path = file_path + "_embedding.msgpack"
                clip_data = get_object(self.minio_client, input_clip_path)
                embedding_dict = ImageEmbedding.from_msgpack_bytes(clip_data)
                image_embedding= embedding_dict.image_embedding
                
                clip_vectors.append({
                    "input_clip": image_embedding,
                    "score": job["clip_sigma_score"]
                })

            except:
                print("An error occured")
        
        return clip_vectors

def main():
    args= parse_args()

    dataloader= KandinskyDatasetLoader(minio_access_key=args.minio_access_key,
                                       minio_secret_key=args.minio_secret_key,
                                       dataset=args.dataset)
    
    dataset= dataloader.load_clip_vector_data()

    print(f"Number of datapoints loaded: {dataset}")


if __name__ == "__main__":
    main()

        