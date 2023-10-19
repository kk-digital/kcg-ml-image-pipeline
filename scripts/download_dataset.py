
import argparse
import os
import sys
from xmlrpc.client import ResponseError
from minio import Minio
import requests

base_directory = "./"
sys.path.insert(0, base_directory)

# MinIO server informatio,
MINIO_ADDRESS = "123.176.98.90:9000"
access_key = "GXvqLWtthELCaROPITOG"
secret_key = "DmlKgey5u0DnMHP30Vg7rkLT0NNbNIGaM8IwPckD"

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--minio-adress", type=str, default=MINIO_ADDRESS,
                        help="IP adress of the MinIO server")
    parser.add_argument("--access-key", type=str, default=access_key,
                        help="Access key of the MinIO server")
    parser.add_argument("--secret-key", type=str, default=secret_key,
                        help="Secret key of the MinIO server")
    
    parser.add_argument("--bucket-name", type=str, default="datasets",
                        help="Name of bucket")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="The name of the dataset to be downloaded")
    parser.add_argument("--file-type", type=str, default="all",
                        help="Type of files that will be downloaded (images or embeddings)")
    parser.add_argument("--output-path", type=str, default="./output",
                        help="path where the diles will be stored")
    
    return parser.parse_args()

def connect_to_minio(minio_addr, access_key, secret_key):
    # Initialize the MinIO client
    client = Minio(minio_addr, access_key, secret_key, secure=False)

    #Check server status
    try:
        response = requests.get("http://" + minio_addr + "/minio/health/live", timeout=5)
        if response.status_code == 200:
            print("Connected to MinIO server.")
        else:
            return None
    except requests.RequestException as e:
        return None
    
    return client

def download_all_files_in_dataset(client, bucket_name, dataset_name, output_path, file_type):
    try:
        # List objects in the bucket
        objects = client.list_objects(bucket_name, dataset_name, recursive=True)

        for obj in objects:
            object_name = obj.object_name

            # filter file type
            if file_type == "images" and not object_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            elif file_type == "embeddings" and not object_name.lower().endswith("_embedding.msgpack"):
                continue

            # Construct the local file path with the dataset name as a subdirectory
            local_file_path = os.path.join(output_path, object_name)

            # Download the object
            client.fget_object(bucket_name, object_name, local_file_path)
            print(f"Downloaded {object_name} to {local_file_path}")
        
        print("Download completed.")
    except ResponseError as e:
            print(f"Error: {e}")
    

def main():
    # parsing arguments
    args = parse_args()
    bucket_name= args.bucket_name
    dataset_name = args.dataset_name
    output_path = args.output_path
    file_type = args.file_type
    minio_address = args.minio_adress
    access_key = args.access_key
    secret_key = args.secret_key


    client = connect_to_minio(minio_address, access_key, secret_key)
    if client is not None:
        download_all_files_in_dataset(client,bucket_name, dataset_name, output_path, file_type)
    else:
        print("Failed to connect to MinIO server:")

    

if __name__ == '__main__':
    main()