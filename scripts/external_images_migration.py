import os
import uuid
from datetime import datetime
from pymongo import MongoClient
from minio import Minio
from minio.error import S3Error
from PIL import Image
import io

# MinIO and MongoDB configurations
MINIO_ENDPOINT = '192.168.3.5:9000'
MINIO_ACCESS_KEY = 'v048BpXpWrsVIHUfdAix'
MINIO_SECRET_KEY = '4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu'
MINIO_BUCKET_NAME = 'external'

MONGODB_URI = 'mongodb://192.168.3.1:32017/'
MONGODB_DB_NAME = 'orchestration-job-db'
MONGODB_COLLECTION_NAME = 'external_images'

# Initialize MinIO client
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=True  # Set to False if not using HTTPS
)

# Initialize MongoDB client
mongo_client = MongoClient(MONGODB_URI)
mongodb = mongo_client[MONGODB_DB_NAME]
collection = mongodb[MONGODB_COLLECTION_NAME]

def get_image_metadata(object_path):
    response = minio_client.get_object(MINIO_BUCKET_NAME, object_path)
    image = Image.open(io.BytesIO(response.read()))
    width, height = image.size
    image_format = image.format

    return {
        "width": width,
        "height": height,
        "image_format": image_format
    }

def migrate_images():
    try:
        objects = minio_client.list_objects(MINIO_BUCKET_NAME, recursive=True)
        for obj in objects:
            if obj.is_dir:
                continue
            
            file_path = obj.object_name
            dataset = file_path.split('/')[1]  # Extract dataset name
            image_hash = uuid.uuid4().hex
            image_metadata = get_image_metadata(file_path)
            
            # Prepare document for MongoDB
            document = {
                "upload_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                "image_hash": image_hash,
                "image_resolution": {
                    "width": image_metadata["width"],
                    "height": image_metadata["height"]
                },
                "image_format": image_metadata["image_format"],
                "file_path": file_path,
                "source_image_dict": {},
                "task_attributes_dict": {},
                "dataset": dataset,
                "uuid": str(uuid.uuid4())
            }
            
            # Insert document into MongoDB
            collection.insert_one(document)
            print(f"Inserted document for {file_path}")

    except S3Error as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    migrate_images()
