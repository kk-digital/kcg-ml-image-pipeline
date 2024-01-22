import os
import sys
from minio import Minio
from utility.minio import cmd
from tqdm import tqdm

minio_ip_addr = '192.168.3.5:9000'
access_key = 'v048BpXpWrsVIHUfdAix'
secret_key = '4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu'
BUCKET_NAME = 'datasets'

def list_datasets(minio_client, bucket_name):
    """
    List all directories in the bucket, assuming each directory is a dataset.
    """
    objects = minio_client.list_objects(bucket_name, recursive=False)
    for obj in objects:
        if obj.is_dir:
            yield obj.object_name.rstrip('/')

def rename_files_in_dataset(minio_client, bucket_name, dataset_name):
    """
    Rename all _latent.msgpack files to _vae_latent.msgpack in a specific dataset.
    """
    prefix = f'{dataset_name}/'
    objects = minio_client.list_objects(bucket_name, prefix=prefix, recursive=True)
    for obj in tqdm(objects, desc=f"Renaming in {dataset_name}", unit="file"):
        if obj.object_name.endswith('_latent.msgpack'):
            new_name = obj.object_name.replace('_latent.msgpack', '_vae_latent.msgpack')
            minio_client.copy_object(bucket_name, new_name, f"{bucket_name}/{obj.object_name}")
            minio_client.remove_object(bucket_name, obj.object_name)
            print(f"Renamed {obj.object_name} to {new_name}")

def main():
    minio_client = cmd.connect_to_minio_client(minio_ip_addr, access_key, secret_key)
    for dataset in list_datasets(minio_client, BUCKET_NAME):
        print(f"Processing dataset: {dataset}")
        rename_files_in_dataset(minio_client, BUCKET_NAME, dataset)

if __name__ == "__main__":
    main()
