import os
import json
from pymongo import MongoClient
from minio import Minio
from collections import OrderedDict

def connect_to_minio_client(minio_ip_addr, access_key, secret_key):
    print("Connecting to minio client...")
    client = Minio(minio_ip_addr, access_key, secret_key, secure=False)
    print("Successfully connected to minio client...")
    return client

def list_datasets(minio_client, bucket_name):
    datasets = set()
    print(f"Listing top-level directories in the MinIO bucket '{bucket_name}' to find datasets...")

    # Listing only the top-level directories in the bucket
    objects = minio_client.list_objects(bucket_name, recursive=False)
    for obj in objects:
        if obj.is_dir:
            # Top-level directories are separated by a slash and end with a slash
            dataset_name = obj.object_name.strip('/').split('/')[0]
            datasets.add(dataset_name)

    datasets_list = list(datasets)
    print(f"Found datasets: {datasets_list}")
    return datasets_list

def migrate_json_to_mongodb(minio_client, mongo_collection, datasets):
    bucket_name = 'datasets'

    for dataset in datasets:
        folder_name = f'{dataset}/data/ranking/aggregate'
        print(f"Processing dataset '{dataset}' located at '{folder_name}' in bucket '{bucket_name}'...")

        objects = minio_client.list_objects(bucket_name, prefix=folder_name, recursive=True)
        for obj in objects:
            if obj.is_dir:
                continue

            print(f"Found object '{obj.object_name}' in dataset '{dataset}'...")
            response = minio_client.get_object(bucket_name, obj.object_name)
            data = response.read()
            original_data = json.loads(data.decode('utf-8'))

            json_filename = obj.object_name.split('/')[-1]

            ordered_data = OrderedDict([
                ("file_name", json_filename),
                ("dataset", dataset),
                ("ranking_image_pair", original_data),
                ("selected_residual", {}) 
            ])

            mongo_collection.insert_one(ordered_data)
            print(f"Migrated '{json_filename}' to MongoDB.")

    print("Migration completed.")

def main():
    minio_ip_addr = '192.168.3.5:9000'
    access_key = 'v048BpXpWrsVIHUfdAix'
    secret_key = '4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu'

    # MinIO bucket name
    bucket_name = 'datasets'

    # Connect to MinIO
    minio_client = connect_to_minio_client(minio_ip_addr, access_key, secret_key)

    # List datasets
    datasets = list_datasets(minio_client, bucket_name)

    mongo_client = MongoClient('mongodb://localhost:27017/')
    db = mongo_client['orchestration-job-db']
    image_pair_ranking_collection = db['image_pair_ranking']

    migrate_json_to_mongodb(minio_client, image_pair_ranking_collection, datasets)

    mongo_client.close()

if __name__ == "__main__":
    main()
