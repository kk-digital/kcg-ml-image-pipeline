import os
import json
import msgpack
import io
import pymongo
from bson import json_util
from minio import Minio
import argparse
import time
from dotenv import dotenv_values

# Load configuration from .env file
# This configuration includes MinIO connection details
config = dotenv_values("./orchestration/api/.env")

# Initialize MinIO client using settings from the .env file
minio_client = Minio(
    config["MINIO_ADDRESS"],
    access_key=config["MINIO_ACCESS_KEY"],
    secret_key=config["MINIO_SECRET_KEY"],
    secure=False  # Set to True if MinIO server uses HTTPS
)

class JobSaver:
    def __init__(self, minio_client, db_url, bucket_name):
        """Initializes the JobSaver class with MinIO client, database URL, and bucket name."""
        self.minio_client = minio_client
        self.db_url = db_url
        self.bucket_name = bucket_name
        self.mongodb_db = self.connect_to_db()

    def connect_to_db(self):
        """Establishes connection to the MongoDB database."""
        mongodb_client = pymongo.MongoClient(self.db_url)
        return mongodb_client["orchestration-job-db"]

    def get_last_uploaded_file(self, dataset):
        """Finds the last uploaded file in a dataset to avoid re-uploading existing files."""
        last_file = None
        objects = self.minio_client.list_objects(self.bucket_name, prefix=f"{dataset}/job/", recursive=True)
        for obj in objects:
            if obj.object_name.endswith('.json'):
                last_file = obj.object_name
        return last_file

    def save_job_to_minio(self, job, dataset):
        """Saves a job to MinIO after formatting and serialization."""
        # Remove MongoDB's '_id' field to avoid serialization issues
        job.pop('_id', None)
    
        # Extract the image name from the file path and strip the file extension
        # This assumes the file path ends with a filename
        image_name = job["task_input_dict"]["file_path"].split('/')[-1].split('.')[0]
    
        # Calculate the folder number based on the image number
        # Assuming images are stored in folders named as per thousand image files (0001, 0002, etc.)
        image_number_for_folder = int(image_name)
        folder_number = image_number_for_folder // 1000 + 1
    
        # Format the folder number with leading zeros (e.g., "0001")
        folder_name = str(folder_number).zfill(4)
    
        # Serialize the job to JSON format for human-readable storage
        json_data = json.dumps(job, default=json_util.default)
    
        # Serialize the job to msgpack format for efficient binary storage
        msgpack_data = msgpack.packb(job, default=json_util.default)
    
        # Construct the file paths for JSON and msgpack files within the MinIO bucket
        json_path = f"{dataset}/job/{folder_name}/{image_name}.json"
        msgpack_path = f"{dataset}/job/{folder_name}/{image_name}.msgpack"
    
        # Upload the JSON data to MinIO
        self.minio_client.put_object(self.bucket_name, json_path, io.BytesIO(json_data.encode('utf-8')), len(json_data))
    
        # Upload the msgpack data to MinIO
        self.minio_client.put_object(self.bucket_name, msgpack_path, io.BytesIO(msgpack_data), len(msgpack_data))
    
        # Print a confirmation message with the paths of the uploaded files
        print(f"Job saved to MinIO: {json_path} and {msgpack_path}")
    
    def process_jobs_for_dataset(self, dataset):
        """Processes and saves all jobs for a given dataset to MinIO."""
        # Connect to the 'completed-jobs' collection in the MongoDB database
        jobs_collection = self.mongodb_db['completed-jobs']
    
        # Retrieve the last uploaded file from MinIO to determine where to resume uploading
        last_file = self.get_last_uploaded_file(dataset)
    
        # Extract the image number from the last uploaded file's name, if it exists
        last_uploaded_image_number = int(last_file.split('/')[-1].split('.')[0]) if last_file else None
    
        # Initialize a counter to track the number of jobs processed
        job_index = 0
    
        # Record the start time for processing
        start_time = time.time()
    
        # Iterate over each job in the MongoDB collection
        for job in jobs_collection.find({"task_input_dict.dataset": dataset}):
            # Extract the image number from each job's file path
            image_number = int(job["task_input_dict"]["file_path"].split('/')[-1].split('.')[0])
    
            # Skip jobs that have already been uploaded
            if last_uploaded_image_number and image_number <= last_uploaded_image_number:
                continue
    
            # Save the job to MinIO
            self.save_job_to_minio(job, dataset)
    
            # Increment the job counter
            job_index += 1
    
        # Record the end time after processing all jobs
        end_time = time.time()
    
        # Calculate the total time taken to process the jobs
        total_time = end_time - start_time
    
        # Print the total number of jobs processed and the time taken
        print(f"Total jobs downloaded and saved to MinIO for dataset '{dataset}': {job_index}")
        print(f"Total time taken: {total_time:.2f} seconds")


def main():
    """Main function to handle command-line arguments for dataset processing."""
    parser = argparse.ArgumentParser(description='Save jobs to MinIO for a given dataset.')
    parser.add_argument('dataset', type=str, help='The name of the dataset to process.')
    args = parser.parse_args()

    job_saver = JobSaver(minio_client, config["DB_URL"], "datasets")
    job_saver.process_jobs_for_dataset(args.dataset)

if __name__ == "__main__":
    main()
