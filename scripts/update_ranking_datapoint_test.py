import os
import json
from pymongo import MongoClient
from minio import Minio
from bson import ObjectId
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

# MinIO connection setup
minio_client = Minio(
    '192.168.3.5:9000',
    access_key='v048BpXpWrsVIHUfdAix',
    secret_key='4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu',
    secure=False  # Set to True if using HTTPS
)

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
    if '_id' in doc:
        print(f"Updating document ID: {doc['_id']}")
    
    if 'image_1_metadata' in doc and doc['image_1_metadata'].get('file_hash'):
        image_source_1 = determine_image_source(doc['image_1_metadata']['file_hash'])
        doc['image_1_metadata']['image_source'] = image_source_1
        print(f"Updated image_1_metadata with image_source: {image_source_1}")
    
    if 'image_2_metadata' in doc and doc['image_2_metadata'].get('file_hash'):
        image_source_2 = determine_image_source(doc['image_2_metadata']['file_hash'])
        doc['image_2_metadata']['image_source'] = image_source_2
        print(f"Updated image_2_metadata with image_source: {image_source_2}")
    
    return doc

def update_minio_objects():
    # List of rank_model_id values to process
    rank_model_ids = range(11)  # 0 to 10 inclusive

    for rank_model_id in rank_model_ids:
        formatted_rank_model_id = f"{rank_model_id:05d}"
        path = f"ranks/{formatted_rank_model_id}/data/ranking/aggregate"
        
        objects = minio_client.list_objects("datasets", prefix=path, recursive=True)
        for obj in objects:
            full_path = obj.object_name

            try:
                # Read the file from MinIO
                response = minio_client.get_object("datasets", full_path)
                file_data = response.read()
                response.close()
                response.release_conn()

                # Parse the JSON data
                json_data = json.loads(file_data)

                # Update the JSON data
                json_data = update_image_source(json_data)

                # Serialize the updated JSON data
                updated_json_data = json.dumps(json_data, indent=4).encode('utf-8')
                data = BytesIO(updated_json_data)

                # Upload the updated data back to MinIO
                minio_client.put_object("datasets", full_path, data, len(updated_json_data), content_type='application/json')
                print(f"Uploaded successfully to MinIO: {full_path}")
            except Exception as e:
                print(f"Error processing file {full_path}: {str(e)}")

if __name__ == "__main__":
    update_minio_objects()
