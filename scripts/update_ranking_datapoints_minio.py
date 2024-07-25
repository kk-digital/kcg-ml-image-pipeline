import os
import json
from pymongo import MongoClient
from minio import Minio
from datetime import datetime
from io import BytesIO

# MongoDB connection setup
MONGO_URI = "mongodb://192.168.3.1:32017/"
DATABASE_NAME = "orchestration-job-db"
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DATABASE_NAME]
ranking_datapoints_collection = db['ranking_datapoints']

# MinIO connection setup
minio_client = Minio(
    '192.168.3.5:9000',
    access_key='v048BpXpWrsVIHUfdAix',
    secret_key='4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu',
    secure=False  # Set to True if using HTTPS
)

def migrate_to_minio():
    cursor = ranking_datapoints_collection.find(no_cursor_timeout=True)
    processed_count = 0

    try:
        for doc in cursor:
            print(f"Processing document with ID: {doc['_id']}")

            # Prepare data for MinIO upload (excluding the '_id' field)
            minio_data = doc.copy()
            minio_data.pop("_id")

            formatted_rank_model_id = f"{doc['rank_model_id']:05d}"
            path = f"datasets/ranks/{formatted_rank_model_id}/data/ranking/aggregate"
            file_name = doc['file_name']
            full_path = f"{path}/{file_name}"
            
            json_data = json.dumps(minio_data, indent=4).encode('utf-8')
            data = BytesIO(json_data)

            # Upload data to MinIO
            try:
                minio_client.put_object("datasets", full_path, data, len(json_data), content_type='application/json')
                print(f"Uploaded successfully to MinIO: {full_path}")
            except Exception as e:
                print(f"Error uploading to MinIO: {str(e)}")

            processed_count += 1
            print(f"Processed document {processed_count} for MinIO upload.")

    finally:
        cursor.close()
        print(f"Total documents processed for MinIO upload: {processed_count}")

if __name__ == "__main__":
    migrate_to_minio()
