import os
import json
from pymongo import MongoClient, UpdateOne
from datetime import datetime
from io import BytesIO

# MongoDB connection setup
MONGO_URI = "mongodb://192.168.3.1:32017/"
DATABASE_NAME = "orchestration-job-db"
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DATABASE_NAME]
ranking_datapoints_collection = db['ranking_datapoints']
completed_jobs_collection = db['completed-jobs']
extracts_collection = db['extracts']
external_images_collection = db['external_images']

BATCH_SIZE = 100  # Adjust the batch size as needed

def determine_image_source(image_hash):
    if completed_jobs_collection.find_one({"task_output_file_dict.output_file_hash": image_hash}):
        return "generated_image"
    elif extracts_collection.find_one({"image_hash": image_hash}):
        return "extract_image"
    elif external_images_collection.find_one({"image_hash": image_hash}):
        return "external_image"
    else:
        return None

def update_image_source(doc):
    if 'image_source' in doc:
        doc.pop('image_source')
    
    if 'image_1_metadata' in doc and doc['image_1_metadata'].get('file_hash'):
        image_source_1 = determine_image_source(doc['image_1_metadata']['file_hash'])
        doc['image_1_metadata']['image_source'] = image_source_1
    
    if 'image_2_metadata' in doc and doc['image_2_metadata'].get('file_hash'):
        image_source_2 = determine_image_source(doc['image_2_metadata']['file_hash'])
        doc['image_2_metadata']['image_source'] = image_source_2
    
    return doc

def update_mongodb():
    cursor = ranking_datapoints_collection.find(no_cursor_timeout=True).batch_size(BATCH_SIZE)
    processed_count = 0

    try:
        while True:
            batch = []
            try:
                for _ in range(BATCH_SIZE):
                    batch.append(next(cursor))
            except StopIteration:
                pass

            if not batch:
                break

            bulk_updates = []
            for doc in batch:
                updated_doc = update_image_source(doc)
                bulk_updates.append(
                    UpdateOne({"_id": updated_doc["_id"]}, {"$set": updated_doc})
                )

            if bulk_updates:
                ranking_datapoints_collection.bulk_write(bulk_updates)
                processed_count += len(bulk_updates)
                print(f"Updated {len(bulk_updates)} documents in MongoDB.")

    finally:
        cursor.close()
        print(f"Total documents processed: {processed_count}")

if __name__ == "__main__":
    update_mongodb()
