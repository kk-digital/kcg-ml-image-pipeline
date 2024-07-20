import time
import datetime
import random
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor, as_completed

# MongoDB connection details
MONGO_URI = "mongodb://192.168.3.1:32017/"
DATABASE_NAME = "orchestration-job-db"
COMPLETED_JOBS_COLLECTION = "extracts"
DATASETS_COLLECTION = "datasets"
ALL_IMAGES_COLLECTION = "all-images"
BUCKET_ID = 1  # Hardcoded bucket ID
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

# Fetch datasets to create a mapping of (dataset_name, bucket_id) to dataset_id
print("Fetching dataset mappings...")
dataset_mapping = {}
for dataset in datasets_collection.find():
    key = (dataset["dataset_name"], BUCKET_ID)
    dataset_mapping[key] = dataset["dataset_id"]
print(f"Dataset mappings fetched: {dataset_mapping}")

# Function to process a batch of jobs
def process_batch(batch):
    if batch:
        all_images_collection.insert_many(batch)
        print(f"Inserted {len(batch)} documents")
        batch.clear()

# Process a single job
def process_job(job, dataset_mapping, existing_image_paths, bucket_id):
    dataset_name = job.get("dataset")
    if not dataset_name:
        print("Skipping job due to missing dataset_name")
        return None  # Skip if dataset_name is not available

    dataset_id = dataset_mapping.get((dataset_name, bucket_id), None)
    if dataset_id is None:
        print(f"Skipping job due to missing dataset_id for dataset_name: {dataset_name} and bucket_id: {bucket_id}")
        return None  # Skip if dataset_id is not found

    task_creation_time = job.get("upload_date")
    if not task_creation_time:
        print("Skipping job due to missing task_creation_time")
        return None  # Skip if task_creation_time is not available

    image_path = job.get("file_path")
    if not image_path:
        print("Skipping job due to missing image_path")
        return None  # Skip if image_path is not available

    if image_path in existing_image_paths:
        print(f"Skipping job with image_path {image_path} as it has already been processed")
        return None  # Skip if image_path already exists

    try:
        # Generate UUID
        uuid = generate_uuid(task_creation_time)
        print(f"Generated UUID: {uuid}")

        # Convert task_creation_time to int32 Unix time
        date_int32 = datetime_to_unix_int32(task_creation_time)
        print(f"Converted task_creation_time to int32 Unix time: {date_int32}")

        # Format the new document
        new_document = {
            "uuid": uuid,
            "index": -1,  # Not used but included as per requirement
            "bucket_id": bucket_id,
            "dataset_id": dataset_id,
            "image_hash": job.get("image_hash"),
            "image_path": image_path,
            "date": date_int32,
        }
        return new_document

    except Exception as e:
        print(f"Error processing job: {e}")
        return None

# Process each document in completed_jobs_collection in batches
print("Processing completed jobs...")
try:
    total_processed = 0
    cursor = completed_jobs_collection.find(no_cursor_timeout=True).batch_size(BATCH_SIZE)
    batch = []

    # Gather all image paths first to check for duplicates
    existing_image_paths = set()
    for doc in all_images_collection.find({}, {"image_path": 1}):
        existing_image_paths.add(doc["image_path"])

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_job = {executor.submit(process_job, job, dataset_mapping, existing_image_paths, BUCKET_ID): job for job in cursor}

        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                new_document = future.result()
                if new_document:
                    batch.append(new_document)
                    existing_image_paths.add(new_document["image_path"])
                    total_processed += 1

                if len(batch) >= BATCH_SIZE:
                    process_batch(batch)

            except Exception as e:
                print(f"Error processing job: {e}")

    if batch:
        process_batch(batch)
    
    print(f"Total processed jobs: {total_processed}")

except Exception as e:
    print(f"Error processing jobs: {e}")
finally:
    cursor.close()

print("Data imported successfully.")
client.close()
