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
BATCH_SIZE = 5000  # Number of documents to process in each batch

# Function to generate UUID
def generate_uuid(task_creation_time):
    formats = ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"]
    for fmt in formats:
        try:
            dt = datetime.datetime.strptime(task_creation_time, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"time data '{task_creation_time}' does not match any known format")
    
    unix_time = int(time.mktime(dt.timetuple()))
    unix_time_32bit = unix_time & 0xFFFFFFFF
    random_32bit = random.randint(0, 0xFFFFFFFF)
    uuid = (random_32bit & 0xFFFFFFFF) | (unix_time_32bit << 32)
    return uuid

# Function to convert datetime to int32 Unix time
def datetime_to_unix_int32(dt_str):
    formats = ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"]
    for fmt in formats:
        try:
            dt = datetime.datetime.strptime(dt_str, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"time data '{dt_str}' does not match any known format")
    
    unix_time = int(time.mktime(dt.timetuple()))
    return unix_time & 0xFFFFFFFF

# Connect to MongoDB
print("Connecting to MongoDB...")
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
completed_jobs_collection = db[COMPLETED_JOBS_COLLECTION]
datasets_collection = db[DATASETS_COLLECTION]
all_images_collection = db[ALL_IMAGES_COLLECTION]

# Fetch datasets to create a mapping of dataset_name to dataset_id
print("Fetching dataset mappings...")
dataset_mapping = {}
for dataset in datasets_collection.find():
    dataset_mapping[dataset["dataset_name"]] = dataset["dataset_id"]
print(f"Dataset mappings fetched: {dataset_mapping}")

# Process each document in completed_jobs_collection in batches
print("Processing completed jobs...")
try:
    cursor = completed_jobs_collection.find(no_cursor_timeout=True)
    batch = []
    for job in cursor:
        task_creation_time = job.get("task_creation_time")
        if not task_creation_time:
            print("Skipping job due to missing task_creation_time")
            continue  # Skip if task_creation_time is not available

        image_path = job.get("task_output_file_dict", {}).get("output_file_path")
        if not image_path:
            print("Skipping job due to missing image_path")
            continue  # Skip if image_path is not available

        # Check if the document already exists in all_images_collection based on image_path
        if all_images_collection.find_one({"image_path": image_path}):
            print(f"Skipping job with image_path {image_path} as it has already been processed")
            continue

        try:
            # Generate UUID
            uuid = generate_uuid(task_creation_time)
            print(f"Generated UUID: {uuid}")

            # Get dataset_id from dataset_mapping
            dataset_name = job.get("task_input_dict", {}).get("dataset")
            dataset_id = dataset_mapping.get(dataset_name, None)
            if dataset_id is None:
                print(f"Skipping job due to missing dataset_id for dataset_name: {dataset_name}")
                continue  # Skip if dataset_id is not found

            # Convert task_creation_time to int32 Unix time
            date_int32 = datetime_to_unix_int32(task_creation_time)
            print(f"Converted task_creation_time to int32 Unix time: {date_int32}")


            # Format the new document
            new_document = {
                "uuid": uuid,
                "index": -1,  # Not used but included as per requirement
                "bucket_id": BUCKET_ID,
                "dataset_id": dataset_id,
                "image_hash": job.get("task_output_file_dict", {}).get("output_file_hash"),
                "image_path": image_path,
                "date": date_int32,
            }

            batch.append(new_document)

            if len(batch) >= BATCH_SIZE:
                all_images_collection.insert_many(batch)
                print(f"Inserted {len(batch)} documents")
                batch.clear()

        except Exception as e:
            print(f"Error processing job: {e}")

    if batch:
        all_images_collection.insert_many(batch)
        print(f"Inserted {len(batch)} documents")
except Exception as e:
    print(f"Error processing jobs: {e}")
finally:
    cursor.close()

print("Data imported successfully.")
client.close()
