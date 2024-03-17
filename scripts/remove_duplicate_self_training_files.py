
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

def standardize_and_hash(data):
    """Standardize the data format for comparison and compute a hash."""
    # Sorting the dictionaries by keys and the list of dictionaries to ensure consistent ordering
    standardized_data = json.dumps(data, sort_keys=True)
    return hashlib.md5(standardized_data.encode('utf-8')).hexdigest()

def get_self_training_data(minio_client):
    self_training_path = DATA_MINIO_DIRECTORY + "/self_training/"
    self_training_files = minio_client.list_objects('datasets', prefix=self_training_path, recursive=True)
    
    hash_to_file = {}
    duplicates = []

    for file in self_training_files:
        file_path = file.object_name
        print(f"Processing {file_path}")

        # Get data from MinIO
        data = minio_client.get_object('datasets', file_path)
        # Read and deserialize the msgpack file content
        content = msgpack.unpackb(data.read(), raw=False)

        # Generate a hash for the standardized content
        content_hash = standardize_and_hash(content)

        # Check if this content has been seen before
        if content_hash in hash_to_file:
            # Duplicate content found
            print(f"Duplicate found: {file_path}")
            duplicates.append(file_path)
        else:
            # New unique content
            hash_to_file[content_hash] = file_path

    # Handling duplicates
    for duplicate_path in duplicates:
        print(f"Removing duplicate: {duplicate_path}")

    print("Processed files. Duplicates identified.")

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
            
            