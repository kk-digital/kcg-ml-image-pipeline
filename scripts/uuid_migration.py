import time
import datetime
from pymongo import MongoClient

# MongoDB connection details
MONGO_URI = "mongodb://192.168.3.1:32017/"
DATABASE_NAME = "orchestration-job-db"
COMPLETED_JOBS_COLLECTION = "completed-jobs"
EXTRACTS_COLLECTION = "extracts"
EXTERNAL_IMAGES_COLLECTION = "external_images"
ALL_IMAGES_COLLECTION = "all-images"

# Hardcoded image_hash
image_hash = "c7bb1a8ac9337733a2b5a095fbefbf6209bf840fec74aecc72e89d71ca7e1442"

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
    if completed_jobs_collection.find_one({"task_output_file_dict.output_file_hash": image_hash}) is not None:
        return completed_jobs_collection, "task_output_file_dict.output_file_hash"
    elif extracts_collection.find_one({"image_hash": image_hash}) is not None:
        return extracts_collection, "image_hash"
    elif external_images_collection.find_one({"image_hash": image_hash}) is not None:
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
    if target_collection is None:
        print("Skipping job due to undefined target collection")
        return None

    if should_skip_job(job, target_collection):
        print(f"Skipping job with uuid {uuid} due to task_type containing 'clip'")
        return None

    job["image_uuid"] = uuid  # Migrate the existing uuid as image_uuid

    return job, target_collection

# Process the document with the specific image_hash
print("Processing the specified image hash...")

try:
    job = all_images_collection.find_one({"image_hash": image_hash})
    if job is None:
        print(f"No job found with image_hash: {image_hash}")
    else:
        processed_job, target_collection = process_job(job)

        if processed_job is not None:
            update_result = target_collection.update_one(
                {"_id": job["_id"]},
                {"$set": {"image_uuid": processed_job["image_uuid"]}}
            )
            if update_result.modified_count > 0:
                print(f"Updated document with image_uuid {processed_job['image_uuid']} in {target_collection.name}")
            else:
                print(f"No documents were updated in {target_collection.name}")

except Exception as e:
    print(f"Error processing job: {e}")

finally:
    client.close()

print("Data migrated successfully.")
