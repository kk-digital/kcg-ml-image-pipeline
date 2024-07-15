import time
import datetime
import random
from pymongo import MongoClient

# MongoDB connection details
MONGO_URI = "mongodb://192.168.3.1:32017/"
DATABASE_NAME = "orchestration-job-db"
COMPLETED_JOBS_COLLECTION = "completed-jobs"
DATASETS_COLLECTION = "datasets"
ALL_IMAGES_COLLECTION = "all-images"
BUCKET_ID = 0  # Hardcoded bucket ID

# Function to generate UUID
def generate_uuid(task_creation_time):
    dt = datetime.datetime.strptime(task_creation_time, "%Y-%m-%dT%H:%M:%S.%f")
    unix_time = int(time.mktime(dt.timetuple()))
    unix_time_32bit = unix_time & 0xFFFFFFFF
    random_32bit = random.randint(0, 0xFFFFFFFF)
    uuid = (random_32bit & 0xFFFFFFFF) | (unix_time_32bit << 32)
    return uuid

# Function to convert datetime to int32 Unix time
def datetime_to_unix_int32(dt_str):
    dt = datetime.datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%f")
    unix_time = int(time.mktime(dt.timetuple()))
    return unix_time & 0xFFFFFFFF

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
completed_jobs_collection = db[COMPLETED_JOBS_COLLECTION]
datasets_collection = db[DATASETS_COLLECTION]
all_images_collection = db[ALL_IMAGES_COLLECTION]

# Fetch datasets to create a mapping of dataset_name to dataset_id
dataset_mapping = {}
for dataset in datasets_collection.find():
    dataset_mapping[dataset["dataset_name"]] = dataset["dataset_id"]

# Process each document in completed_jobs_collection
for job in completed_jobs_collection.find():
    task_creation_time = job.get("task_creation_time")
    if not task_creation_time:
        continue  # Skip if task_creation_time is not available

    # Generate UUID
    uuid = generate_uuid(task_creation_time)

    # Get dataset_id from dataset_mapping
    dataset_name = job.get("task_input_dict", {}).get("dataset")
    dataset_id = dataset_mapping.get(dataset_name, None)
    if dataset_id is None:
        continue  # Skip if dataset_id is not found

    # Convert task_creation_time to int32 Unix time
    date_int32 = datetime_to_unix_int32(task_creation_time)
    
    # Format the new document with UTC creation_time
    utc_creation_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Format the new document
    new_document = {
        "uuid": uuid,
        "index": -1,  # Not used but included as per requirement
        "bucket_id": BUCKET_ID,
        "dataset_id": dataset_id,
        "image_hash": job.get("task_output_file_dict", {}).get("output_file_hash"),
        "image_path": job.get("task_output_file_dict", {}).get("output_file_path"),
        "date": date_int32,
        "creation_time": utc_creation_time
    }

    # Insert the new document into all_images_collection
    all_images_collection.insert_one(new_document)

print("Data imported successfully.")
client.close()
