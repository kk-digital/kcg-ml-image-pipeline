import argparse
from io import BytesIO
import json
import os
import sys
import numpy as np
import requests
import msgpack
from tqdm import tqdm

base_directory = "./"
sys.path.insert(0, base_directory)
from utility.path import separate_bucket_and_file_path
from kandinsky.models.clip_image_encoder.clip_image_encoder import KandinskyCLIPImageEncoder
from utility.minio import cmd
from data_loader.utils import get_object

API_URL = "http://192.168.3.1:8111"

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--minio-access-key", type=str,
                        help="The minio access key to use so worker can upload files to minio server")
    parser.add_argument("--minio-secret-key", type=str,
                        help="The minio secret key to use so worker can upload files to minio server")

    return parser.parse_args()

def get_job_list(dataset):
    response = requests.get(f'{API_URL}/image_by_rank/image-list-sorted?dataset={dataset}&limit=4000000&model_type=elm-v1')
        
    jobs = json.loads(response.content)

    return jobs['response']

def main():
    args = parse_args()

    minio_client= cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                       minio_secret_key=args.minio_secret_key)
    
    datasets=["character", "environmental", "icons", "mech", "propaganda-poster", "waifu"]
    
    for dataset in datasets:
        print(f"fetching job data for {dataset}...........")
        jobs_list= get_job_list(dataset=dataset)
        clip_vectors=[]

        print(f"fetching image clip files for {dataset}...........")
        for job in tqdm(jobs_list):
            try:
                image_path= job['image_path']
                bucket_name, input_file_path = separate_bucket_and_file_path(image_path)
                file_path = os.path.splitext(input_file_path)[0]

                clip_path = file_path + "_clip_kandinsky.msgpack"
                clip_data = get_object(minio_client, clip_path)
                clip_vector = msgpack.unpackb(clip_data)['clip-feature-vector']
                clip_vectors.append(clip_vector)
            except:
                print("an error occured")

        # Convert list of vectors into a numpy array for easier computation
        clip_vectors_np = np.array(clip_vectors)

        # Calculate mean and std for each feature
        mean_vector = np.mean(clip_vectors_np, axis=0)
        std_vector = np.std(clip_vectors_np, axis=0)

        # Calculate max and min vectors
        max_vector = np.max(clip_vectors_np, axis=0)
        min_vector = np.min(clip_vectors_np, axis=0)

        stats = {
            "mean": mean_vector.tolist(),
            "std": std_vector.tolist(),
            "max": max_vector.tolist(),
            "min": min_vector.tolist(),
        }

        print(stats)
        stats_msgpack = msgpack.packb(stats)

        data = BytesIO()
        data.write(stats_msgpack)
        data.seek(0)

        # Storing stats in MinIO
        bucket_name = "datasets"
        output_path = f"{dataset}/output/stats/clip_stats.msgpack"
        cmd.upload_data(minio_client, bucket_name, output_path, data)

if __name__ == '__main__':
    main()
