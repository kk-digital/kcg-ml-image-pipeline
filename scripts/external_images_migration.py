import os
import uuid
from datetime import datetime
from pymongo import MongoClient
from minio import Minio
from minio.error import S3Error
from PIL import Image
import io
import hashlib
from tqdm import tqdm

# MinIO and MongoDB configurations
MINIO_ENDPOINT = '192.168.3.5:9000'
MINIO_ACCESS_KEY = 'v048BpXpWrsVIHUfdAix'
MINIO_SECRET_KEY = '4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu'
MINIO_BUCKET_NAME = 'external'

MONGODB_URI = 'mongodb://192.168.3.1:32017/'
MONGODB_DB_NAME = 'orchestration-job-db'
MONGODB_COLLECTION_NAME = 'external_images'
# Dataset to migrate
DATASET_TO_MIGRATE = 'metroid-fusion-dataset'

# Initialize MinIO client
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False  # Set to False if not using HTTPS
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

def generate_image_hash(image_data):
    return hashlib.sha256(image_data).hexdigest()

def migrate_images():
    try:
        objects = minio_client.list_objects("external", prefix=f"{DATASET_TO_MIGRATE}", recursive=True)
        object_list = list(objects)
        
        for obj in tqdm(object_list, desc="Migrating images"):
            if obj.is_dir:
                continue
            
            file_path = obj.object_name
            image_response = minio_client.get_object("external", file_path)
            image_data = image_response.read()
            image_hash = generate_image_hash(image_data)
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
                "dataset": DATASET_TO_MIGRATE,
                "uuid": str(uuid.uuid4())
            }
            
            # Insert document into MongoDB
            collection.insert_one(document)
            print(f"Inserted document for {file_path}")

    except S3Error as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    migrate_images()
