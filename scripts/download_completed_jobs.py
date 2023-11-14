import os
import sys
import json
import msgpack
import io
import pymongo
from bson import json_util
from dotenv import dotenv_values
from minio import Minio
import argparse
import time

# Load configuration from .env file
config = dotenv_values("./orchestration/api/.env")

def connect_to_db():
    mongodb_client = pymongo.MongoClient(config["DB_URL"])
    mongodb_db = mongodb_client["orchestration-job-db"]
    print('Connected to database')
    return mongodb_db

# Initialize MinIO client using utility function
minio_client = Minio(
    config["MINIO_ADDRESS"],
    access_key=config["MINIO_ACCESS_KEY"],
    secret_key=config["MINIO_SECRET_KEY"],
    secure=False  # Update this according to your MinIO configuration (True if using HTTPS)
)

def save_job_to_minio(minio_client, bucket_name, job, dataset):
    job.pop('_id', None)  # Remove the '_id' field if it exists
    
    # Extract the image name with the file extension and keep it as a string
    image_name = job["task_input_dict"]["file_path"].split('/')[-1].split('.')[0]
    
    # Extract the numeric part of the image name for folder calculation
    image_number_for_folder = int(image_name)
    
    # Calculate folder number by floor dividing the image number by 1000 and adding 1
    folder_number = image_number_for_folder // 1000 + 1
    
    # Format the folder number with leading zeros
    folder_name = str(folder_number).zfill(4)
    
    # Serialize the job to JSON and msgpack formats
    json_data = json.dumps(job, default=json_util.default)
    msgpack_data = msgpack.packb(job, default=json_util.default)

    # Define file paths using the specified directory structure
    json_path = f"{dataset}/job/{folder_name}/{image_name}.json"
    msgpack_path = f"{dataset}/job/{folder_name}/{image_name}.msgpack"

    # Save JSON to MinIO
    minio_client.put_object(bucket_name, json_path, io.BytesIO(json_data.encode('utf-8')), len(json_data))

    # Save msgpack to MinIO
    minio_client.put_object(bucket_name, msgpack_path, io.BytesIO(msgpack_data), len(msgpack_data))

    print(f"Job saved to MinIO: {json_path} and {msgpack_path}")



def process_jobs_for_dataset(minio_client, bucket_name, dataset):
    db = connect_to_db()
    jobs_collection = db['completed-jobs']
    
    job_index = 0  # Initialize a counter for the job index
    start_time = time.time()  # Record the start time for the entire operation

    for job in jobs_collection.find({"task_input_dict.dataset": dataset}):

        save_job_to_minio(minio_client, bucket_name, job, dataset)  # Removed the job_index argument

        job_index += 1  # Increment the job index for each job

    end_time = time.time()  # Record the end time for the entire operation
    total_time = end_time - start_time  # Calculate the total time taken

    print(f"Total jobs downloaded and saved to MinIO for dataset '{dataset}': {job_index}")
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Average time per job: {total_time / job_index:.2f} seconds" if job_index else "No jobs processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and save jobs for a given dataset.')
    parser.add_argument('dataset', type=str, help='The name of the dataset to process.')
    args = parser.parse_args()

    bucket_name = "datasets"  # This should be your actual bucket name on MinIO
    process_jobs_for_dataset(minio_client, bucket_name, args.dataset)
