import os
import sys
from minio import Minio
from utility.minio import cmd
from tqdm import tqdm

minio_ip_addr = '192.168.3.5:9000'
access_key = 'v048BpXpWrsVIHUfdAix'
secret_key = '4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu'
BUCKET_NAME = 'datasets'

def rename_files(minio_client, bucket_name):
    """
    Rename all files ending with '_latent.msgpack' to '_vae_latent.msgpack'.
    """
    objects = minio_client.list_objects(bucket_name, recursive=True)
    
    # Collect all objects that need to be renamed
    to_rename = [obj for obj in objects if obj.object_name.endswith('_latent.msgpack')]

    # Process each object with tqdm for progress indication
    for obj in tqdm(to_rename, desc="Renaming files", unit="file"):
        new_name = obj.object_name.replace('_latent.msgpack', '_vae_latent.msgpack')
        minio_client.copy_object(bucket_name, new_name, f"{bucket_name}/{obj.object_name}")
        minio_client.remove_object(bucket_name, obj.object_name)
        print(f"Renamed {obj.object_name} to {new_name}")

def main():
    minio_client = cmd.connect_to_minio_client(minio_ip_addr, access_key, secret_key)
    rename_files(minio_client, BUCKET_NAME)

if __name__ == "__main__":
    main()
