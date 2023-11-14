import os
import json
import msgpack
import pymongo
from bson import json_util  
from dotenv import dotenv_values
import argparse

# Load configuration from .env file
config = dotenv_values("./orchestration/api/.env")

def connect_to_db():
    mongodb_client = pymongo.MongoClient(config["DB_URL"])
    mongodb_db = mongodb_client["orchestration-job-db"]  
    print('Connected to database')
    return mongodb_db

def save_job_as_json_and_msgpack(job, dataset):
    # Extract the image name without the file extension
    image_name = job["task_input_dict"]["file_path"].split('/')[-1].split('.')[0]
    
    # Use the image name for directory and file names
    directory = f"./{dataset}/{image_name}"
    os.makedirs(directory, exist_ok=True)
    
    json_path = f"{directory}/{image_name}.json"
    msgpack_path = f"{directory}/{image_name}.msgpack"

    # Save as JSON with custom encoder
    with open(json_path, "w") as file:
        json.dump(job, file, default=json_util.default, indent=4)

    # Save as Message Pack
    with open(msgpack_path, "wb") as file:
        packed = msgpack.packb(job, default=json_util.default)
        file.write(packed)
    
    return image_name  # Return the image name to use in the calling function

def process_jobs_for_dataset(dataset):
    db = connect_to_db()
    jobs_collection = db['completed-jobs']
    
    job_count = 0  # Initialize a counter for the number of downloaded jobs
    for job in jobs_collection.find({"task_input_dict.dataset": dataset}):
        image_name = save_job_as_json_and_msgpack(job, dataset)  # Receive the image name here
        print(f"Processed job. Files saved in './{dataset}/{image_name}'")
        job_count += 1  # Increment the counter for each processed job
    
    print(f"Total jobs downloaded and saved for dataset '{dataset}': {job_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and save jobs for a given dataset.')
    parser.add_argument('dataset', type=str, help='The name of the dataset to process.')
    args = parser.parse_args()

    process_jobs_for_dataset(args.dataset)