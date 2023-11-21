import json
import argparse
from minio import Minio
from dotenv import dotenv_values
import io

# Load configuration from .env file
config = dotenv_values("./orchestration/api/.env")

# Initialize MinIO client using utility function
minio_client = Minio(
    config["MINIO_ADDRESS"],
    access_key=config["MINIO_ACCESS_KEY"],
    secret_key=config["MINIO_SECRET_KEY"],
    secure=False  # Update this according to your MinIO configuration (True if using HTTPS)
)

def load_job_from_minio(minio_client, bucket_name, dataset, folder_name, image_name):
    json_path = f"{dataset}/job/{folder_name}/{image_name}.json"
    try:
        # Get the JSON object content
        response = minio_client.get_object(bucket_name, json_path)
        job_data = response.read().decode('utf-8')
        job = json.loads(job_data)
        # Print in pretty JSON format
        pretty_job = json.dumps(job, ensure_ascii=False, indent=4)
        print(pretty_job)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in file: {json_path}. Error: {e}")
    except Exception as e:
        print(f"Error loading file {json_path}: {str(e)}")

def load_all_jobs_from_minio(minio_client, bucket_name, dataset):
    job_count = 0
    folder_number = 1  # Start with the first folder

    while True:
        folder_name = str(folder_number).zfill(4)
        prefix = f"{dataset}/job/{folder_name}/"
        objects = minio_client.list_objects(bucket_name, prefix=prefix, recursive=True)
        objects_list = list(objects)

        if not objects_list:
            # If no objects are found in the current folder, stop the loop
            break

        for obj in objects_list:
            if obj.object_name.endswith('.json'):
                image_name = obj.object_name.split('/')[-1].split('.')[0]
                load_job_from_minio(minio_client, bucket_name, dataset, folder_name, image_name)
                job_count += 1

        folder_number += 1  # Increment to the next folder

    print(f"Total jobs loaded for dataset '{dataset}': {job_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load jobs from MinIO for a given dataset.')
    parser.add_argument('dataset', type=str, help='The name of the dataset to load jobs from.')
    args = parser.parse_args()

    bucket_name = "datasets"  # This should be your actual bucket name on MinIO
    load_all_jobs_from_minio(minio_client, bucket_name, args.dataset)
