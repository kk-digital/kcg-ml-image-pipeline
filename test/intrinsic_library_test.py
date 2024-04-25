import argparse
import os
import sys
import torch
from intrinsics_dimension import mle_id, twonn_numpy, twonn_pytorch
base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from data_loader.kandinsky_dataset_loader import KandinskyDatasetLoader
from utility.minio import cmd
from utility.http import request
from utility.path import separate_bucket_and_file_path
from data_loader.utils import get_object
import time
import msgpack
import requests
import json

API_URL="http://192.168.3.1:8111"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--minio-addr', type=str, help='Minio address')
    parser.add_argument('--data-type', type=str, default="clip", help='Data types to obtain intrinsic dimensions, for exmaple, clip and vae')
    parser.add_argument('--num-vectors', type=int, default=1024, help="Number of vectors to get intrinsic dimensions")
    return parser.parse_args()

def load_kandinsky_jobs(self, dataset):
    print(f"Fetching kandinsky jobs for the {dataset} dataset")
    response = requests.get(f'{API_URL}/queue/image-generation/list-completed-by-dataset-and-task-type?dataset={dataset}&task_type=img2img_generation_kandinsky')
        
    jobs = json.loads(response.content)

    return jobs


def load_vectors(minio_client, dataset_names, vector_type="clip",  limit=1024):
    num_loaded_vectors = 0

    feature_vectors = []

    for dataset_name in dataset_names:

        if num_loaded_vectors >= limit:
            break

        # jobs = request.http_get_completed_job_by_dataset(dataset=dataset_name, limit=limit)
        jobs = load_kandinsky_jobs(dataset_name)
        for job in jobs:

            if num_loaded_vectors >= limit:
                break
            
            # path = job.get("task_output_file_dict", {}).get("output_file_path")
            path = job['file_path']
            if path:
                if vector_type == "clip":
                    path = path.replace(".jpg", "_embedding.msgpack")
                    bucket, features_vector_path = separate_bucket_and_file_path(path)
                    try:
                        loaded_feature_vector = get_object(minio_client, features_vector_path)
                        feature_vector_dict = msgpack.unpackb(loaded_feature_vector)

                        feature_vectors.append(feature_vector_dict["image_embedding"])

                        num_loaded_vectors += 1
                    except Exception as e:
                        print("Error in loading feature vector: {}, {}".format(features_vector_path, e))

                elif vector_type == "vae":
                    path = path.replace(".jpg", "_vae_latent.msgpack")
                    bucket, features_vector_path = separate_bucket_and_file_path(path)
                    try:
                        feature_vector = get_object(minio_client, features_vector_path)
                        print("----->", msgpack.unpackb(feature_vector))
                        feature_vector = msgpack.unpackb(feature_vector)["latent_vector"]
                        feature_vectors.append(feature_vector)

                        num_loaded_vectors += 1
                    except Exception as e:
                        print("Error in loading feature vector: {}, {}".format(path, e))

    return feature_vectors
        
def main():
    args = parse_args()

    # get minio client
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)
    dataset_names = request.http_get_dataset_names()
    all_feature_vectors = []

    list_clip_vector_num = [args.num_vectors]

    for dataset_name in dataset_names:
        try:
            dataloader = KandinskyDatasetLoader(minio_client=minio_client, dataset=dataset_name)
        except Exception as e:
            print("Error in initializing kankinsky dataset loader:{}".format(e))
            continue

        if args.data_type == "clip":
            feature_vectors, _= dataloader.load_clip_vector_data(limit=max(list_clip_vector_num))
        elif args.data_type == "vae":
            feature_vectors = dataloader.load_latents(limit=max(list_clip_vector_num))
        else:
            print("No support data type {}".format(args.data_type))
            return None
        all_feature_vectors.extend(feature_vectors)
        print(len(all_feature_vectors), max(list_clip_vector_num))
        if len(all_feature_vectors) >= max(list_clip_vector_num):
            break
    result = []

    print("length", len(all_feature_vectors))

    for clip_vector_num in list_clip_vector_num:
        data = torch.tensor(all_feature_vectors[:clip_vector_num])

        # wrangle the latent vector [1, 4, 64, 64]
        if args.data_type == "vae":
            data = data.reshape((data.size(0), -1))

        d1 = mle_id(data, k=2)
        d2 = twonn_numpy(data.numpy(), return_xy=False)
        d3 = twonn_pytorch(data, return_xy=False)

        result.append({
            "number of clip vector": data.size(0),
            "dimension of clip vector": data.size(1),
            "dimension": data.size(1),
            "mle_id": d1,
            "twonn_numpy": d2,
            "twonn_pytorch": d3,
        })
    print(result)

if __name__ == "__main__":
    main()