from minio import Minio
import json
import io
from tqdm import tqdm

# MinIO setup
minio_client = Minio(
    '192.168.3.5:9000',  
    access_key='v048BpXpWrsVIHUfdAix',  
    secret_key='4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu', 
    secure=False  # Set to True if using HTTPS
)

rank_model_id = 9
source_bucket_name = 'datasets'  
destination_bucket_name = 'datasets'  
source_path = 'environmental/data/ranking/aggregate'  
destination_path = f'ranks/{rank_model_id}/data/ranking/aggregate'  

# Ensure the destination bucket exists
if not minio_client.bucket_exists(destination_bucket_name):
    minio_client.make_bucket(destination_bucket_name)

# List all objects in the source bucket with the given source path
objects = minio_client.list_objects(source_bucket_name, prefix=source_path, recursive=True)
objects = list(objects)  # Convert generator to list for tqdm

total_objects = len(objects)
print(f"Total JSON files to migrate: {total_objects}")

# Iterate over objects and copy them to the destination bucket
for obj in tqdm(objects, desc="Migrating JSON files"):
    if obj.object_name.endswith('.json'):
        # Get the object from the source bucket
        response = minio_client.get_object(source_bucket_name, obj.object_name)
        
        # Read the entire response data
        json_data = json.loads(response.read())
        response.close()
        response.release_conn()

        # Add rank_model_id as the first field in the JSON data
        updated_data = {
            "rank_model_id": rank_model_id,
            "task": json_data.get("task"),
            "username": json_data.get("username"),
            "image_1_metadata": json_data.get("image_1_metadata"),
            "image_2_metadata": json_data.get("image_2_metadata"),
            "selected_image_index": json_data.get("selected_image_index"),
            "selected_image_hash": json_data.get("selected_image_hash"),
            "training_mode": "rank_active_learning",
            "rank_active_learning_policy_id": None,
            "datetime": json_data.get("datetime")
        }

        # Convert the modified JSON data back to a JSON string with indentation
        json_data_str = json.dumps(updated_data, indent=4)

        # Define the object name in the destination bucket
        destination_object_name = obj.object_name.replace(source_path, destination_path)

        # Upload the JSON data to the destination bucket
        minio_client.put_object(
            destination_bucket_name,
            destination_object_name,
            data=io.BytesIO(json_data_str.encode('utf-8')),
            length=len(json_data_str),
            content_type='application/json'
        )

print("Migration of JSON files to MinIO completed successfully!")