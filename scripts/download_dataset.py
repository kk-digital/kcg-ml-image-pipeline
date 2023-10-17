
import argparse
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
secure = False 

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--bucket-name", type=str,
                        help="Name of bucket")
    parser.add_argument("--dataset-name", type=str,
                        help="The name of the dataset to be downloaded")
    parser.add_argument("--output-path", type=str,
                        help="The path where the downloaded files will be stored")
    return parser.parse_args()

def connect_to_minio():
    # Initialize the MinIO client
    client = Minio(MINIO_ADDRESS, access_key, secret_key, secure=secure)

    #Check server status
    try:
        response = requests.get("http://" + MINIO_ADDRESS + "/minio/health/live", timeout=5)
        if response.status_code == 200:
            print("Connected to MinIO server.")
        else:
            return None
    except requests.RequestException as e:
        return None
    
    return client
    
def download_all_files_in_dataset(client, bucket_name, directory_name, local_directory):
        try:
            objects = client.list_objects(bucket_name, prefix=directory_name, recursive=True)
            for obj in objects:
                object_name = obj.object_name
                # get the local path for the downloaded file
                local_path = local_directory + object_name[len(directory_name):]

                client.fget_object(bucket_name, object_name, local_path)
                print(f"Downloaded {object_name} to {local_path}")
        except ResponseError as e:
            print(f"Error: {e}")

def main():
    # parsing arguments
    args = parse_args()
    bucket_name= args.bucket_name
    dataset_name = args.dataset_name
    output_path = args.output_path


    client = connect_to_minio()
    if client is not None:
        download_all_files_in_dataset(client,bucket_name, dataset_name, output_path)
    else:
        print("Failed to connect to MinIO server:")

    

if __name__ == '__main__':
    main()