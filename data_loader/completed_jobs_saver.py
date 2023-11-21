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
config = dotenv_values("./orchestration/api/.env")

# Initialize MinIO client
minio_client = Minio(
    config["MINIO_ADDRESS"],
    access_key=config["MINIO_ACCESS_KEY"],
    secret_key=config["MINIO_SECRET_KEY"],
    secure=False
)

class JobSaver:
    def __init__(self, minio_client, db_url, bucket_name):
        self.minio_client = minio_client
        self.db_url = db_url
        self.bucket_name = bucket_name
        self.mongodb_db = self.connect_to_db()

    def connect_to_db(self):
        mongodb_client = pymongo.MongoClient(self.db_url)
        return mongodb_client["orchestration-job-db"]

    def get_last_uploaded_file(self, dataset):
        last_file = None
        objects = self.minio_client.list_objects(self.bucket_name, prefix=f"{dataset}/job/", recursive=True)
        for obj in objects:
            if obj.object_name.endswith('.json'):
                last_file = obj.object_name
        return last_file

    def save_job_to_minio(self, job, dataset):
        job.pop('_id', None)
        image_name = job["task_input_dict"]["file_path"].split('/')[-1].split('.')[0]
        image_number_for_folder = int(image_name)
        folder_number = image_number_for_folder // 1000 + 1
        folder_name = str(folder_number).zfill(4)
        json_data = json.dumps(job, default=json_util.default)
        msgpack_data = msgpack.packb(job, default=json_util.default)
        json_path = f"{dataset}/job/{folder_name}/{image_name}.json"
        msgpack_path = f"{dataset}/job/{folder_name}/{image_name}.msgpack"
        self.minio_client.put_object(self.bucket_name, json_path, io.BytesIO(json_data.encode('utf-8')), len(json_data))
        self.minio_client.put_object(self.bucket_name, msgpack_path, io.BytesIO(msgpack_data), len(msgpack_data))
        print(f"Job saved to MinIO: {json_path} and {msgpack_path}")

    def process_jobs_for_dataset(self, dataset):
        jobs_collection = self.mongodb_db['completed-jobs']
        last_file = self.get_last_uploaded_file(dataset)
        last_uploaded_image_number = int(last_file.split('/')[-1].split('.')[0]) if last_file else None
        job_index = 0
        start_time = time.time()
        for job in jobs_collection.find({"task_input_dict.dataset": dataset}):
            image_number = int(job["task_input_dict"]["file_path"].split('/')[-1].split('.')[0])
            if last_uploaded_image_number and image_number <= last_uploaded_image_number:
                continue
            self.save_job_to_minio(job, dataset)
            job_index += 1
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total jobs downloaded and saved to MinIO for dataset '{dataset}': {job_index}")
        print(f"Total time taken: {total_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Save jobs to MinIO for a given dataset.')
    parser.add_argument('dataset', type=str, help='The name of the dataset to process.')
    args = parser.parse_args()

    job_saver = JobSaver(minio_client, config["DB_URL"], "datasets")
    job_saver.process_jobs_for_dataset(args.dataset)

if __name__ == "__main__":
    main()
