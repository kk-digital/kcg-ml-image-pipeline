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

def migrate_to_minio(rank_model_id: int):
    cursor = ranking_datapoints_collection.find({"rank_model_id": rank_model_id}, no_cursor_timeout=True)
    processed_count = 0

    try:
        for doc in cursor:
            print(f"Processing document with ID: {doc['_id']}")
            print(f"File Name: {doc['file_name']}")
            print(f"Rank Model ID: {doc['rank_model_id']}")

            # Prepare data for MinIO upload (excluding the '_id' field)
            minio_data = doc.copy()
            minio_data.pop("_id")
            minio_data.pop("file_name")

            # Convert minio_data to JSON and then to bytes
            json_data = json.dumps(minio_data, indent=4)
            data = json_data.encode('utf-8')
            data_length = len(data)

            formatted_rank_model_id = f"{doc['rank_model_id']:05d}"
            path = f"datasets/ranks/{formatted_rank_model_id}/data/ranking/aggregate"
            full_path = f"{path}/{doc['file_name']}"

            # Upload data to MinIO
            try:
                print(f"Uploading to MinIO: {full_path}")
                minio_client.put_object("datasets", full_path, BytesIO(data), data_length, content_type='application/json')
                print(f"Uploaded successfully to MinIO: {full_path}")
            except Exception as e:
                print(f"Error uploading to MinIO: {str(e)}")

            processed_count += 1
            print(f"Processed document {processed_count} for MinIO upload.")

    finally:
        cursor.close()
        print(f"Total documents processed for MinIO upload: {processed_count}")

if __name__ == "__main__":
    rank_model_ids = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]
    for rank_model_id in rank_model_ids:
        print(f"Processing rank_model_id: {rank_model_id}")
        migrate_to_minio(rank_model_id)
