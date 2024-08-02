import time
import datetime
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor, as_completed

# MongoDB connection details
MONGO_URI = "mongodb://192.168.3.1:32017/"
DATABASE_NAME = "orchestration-job-db"
COMPLETED_JOBS_COLLECTION = "completed-jobs"
EXTRACTS_COLLECTION = "extracts"
EXTERNAL_IMAGES_COLLECTION = "external_images"
ALL_IMAGES_COLLECTION = "all-images"
BATCH_SIZE = 5000  # Number of documents to process in each batch

# Connect to MongoDB
print("Connecting to MongoDB...")
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
completed_jobs_collection = db[COMPLETED_JOBS_COLLECTION]
extracts_collection = db[EXTRACTS_COLLECTION]
external_images_collection = db[EXTERNAL_IMAGES_COLLECTION]
all_images_collection = db[ALL_IMAGES_COLLECTION]

# Determine the target collection for the job based on the hash
def determine_target_collection(image_hash):
    if completed_jobs_collection.find_one({"task_output_file_dict.output_file_hash": image_hash}):
        return completed_jobs_collection, "task_output_file_dict.output_file_hash"
    elif extracts_collection.find_one({"image_hash": image_hash}):
        return extracts_collection, "image_hash"
    elif external_images_collection.find_one({"image_hash": image_hash}):
        return external_images_collection, "image_hash"
    else:
        return None, None

# Check if the job should be skipped
def should_skip_job(job, target_collection):
    if target_collection == completed_jobs_collection and "task_type" in job and "clip" in job["task_type"]:
        return True
    return False

# Process a single job
def process_job(job):
    image_hash = job.get("image_hash")
    if not image_hash:
        print("Skipping job due to missing image_hash")
        return None  # Skip if required fields are not available

    uuid = job.get("uuid")
    if not uuid:
        print("Skipping job due to missing uuid")
        return None  # Skip if uuid is not available

    target_collection, field_path = determine_target_collection(image_hash)
    if not target_collection:
        print("Skipping job due to undefined target collection")
        return None

    if should_skip_job(job, target_collection):
        print(f"Skipping job with uuid {uuid} due to task_type containing 'clip'")
        return None

    job["image_uuid"] = uuid  # Migrate the existing uuid as image_uuid

    return job, target_collection

# Process each document in all_images_collection in batches
print("Processing all images...")
try:
    total_processed = 0
    cursor = all_images_collection.find(no_cursor_timeout=True).batch_size(BATCH_SIZE)
    completed_batch = []
    extracts_batch = []
    external_images_batch = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_job = {executor.submit(process_job, job): job for job in cursor}

        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
                if result:
                    new_document, target_collection = result
                    if target_collection == completed_jobs_collection:
                        completed_batch.append(new_document)
                        if len(completed_batch) >= BATCH_SIZE:
                            completed_jobs_collection.insert_many(completed_batch)
                            print(f"Inserted {len(completed_batch)} documents into {completed_jobs_collection.name}")
                            completed_batch.clear()
                    elif target_collection == extracts_collection:
                        extracts_batch.append(new_document)
                        if len(extracts_batch) >= BATCH_SIZE:
                            extracts_collection.insert_many(extracts_batch)
                            print(f"Inserted {len(extracts_batch)} documents into {extracts_collection.name}")
                            extracts_batch.clear()
                    elif target_collection == external_images_collection:
                        external_images_batch.append(new_document)
                        if len(external_images_batch) >= BATCH_SIZE:
                            external_images_collection.insert_many(external_images_batch)
                            print(f"Inserted {len(external_images_batch)} documents into {external_images_collection.name}")
                            external_images_batch.clear()
                    total_processed += 1

            except Exception as e:
                print(f"Error processing job: {e}")

    # Insert any remaining documents in the batches
    if completed_batch:
        completed_jobs_collection.insert_many(completed_batch)
        print(f"Inserted {len(completed_batch)} documents into {completed_jobs_collection.name}")
    if extracts_batch:
        extracts_collection.insert_many(extracts_batch)
        print(f"Inserted {len(extracts_batch)} documents into {extracts_collection.name}")
    if external_images_batch:
        external_images_collection.insert_many(external_images_batch)
        print(f"Inserted {len(external_images_batch)} documents into {external_images_collection.name}")

    print(f"Total processed jobs: {total_processed}")

except Exception as e:
    print(f"Error processing jobs: {e}")
finally:
    cursor.close()

print("Data migrated successfully.")
client.close()
