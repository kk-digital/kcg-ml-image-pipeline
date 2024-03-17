
import argparse
import os
import sys
import msgpack
import hashlib
import json

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())
from utility.http import request
from utility.minio import cmd

DATA_MINIO_DIRECTORY="data/latent-generator"
API_URL = "http://192.168.3.1:8111"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")

    return parser.parse_args()


def get_self_training_data(minio_client):
    self_training_path = DATA_MINIO_DIRECTORY + "/self_training/"
    self_training_files = minio_client.list_objects('datasets', prefix=self_training_path, recursive=True)
    
    unique_data = []
    duplicates = []

    for file in self_training_files:
        file_path = file.object_name
        print(f"Processing {file_path}")

        # Get data from MinIO
        data = minio_client.get_object('datasets', file_path)
        # Read and deserialize the msgpack file content
        content = msgpack.unpackb(data.read(), raw=False)

        # Check if this content is already in unique_data
        if any(content == existing_content for existing_content in unique_data):
            print(f"Duplicate found: {file_path}")
            duplicates.append(file_path)
        else:
            unique_data.append(content)

def main():
    args = parse_args()

    # get minio client
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key)
    
    global DATA_MINIO_DIRECTORY

    if args.dataset != "all":
        DATA_MINIO_DIRECTORY= f"{args.dataset}/data/latent-generator"
        get_self_training_data(minio_client)
    
    else:
        # if all, train models for all existing datasets
        # get dataset name list
        dataset_names = request.http_get_dataset_names()
        print("dataset names=", dataset_names)
        for dataset in dataset_names:
            DATA_MINIO_DIRECTORY= f"{dataset}/data/latent-generator"
            get_self_training_data(minio_client)

if __name__ == "__main__":
    main()
            
            