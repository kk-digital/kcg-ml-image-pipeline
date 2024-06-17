from minio import Minio
from io import BytesIO
import json
from collections import OrderedDict

# MinIO connection details
MINIO_ENDPOINT = "192.168.3.5:9000"
MINIO_ACCESS_KEY = "v048BpXpWrsVIHUfdAix"
MINIO_SECRET_KEY = "4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu"
BUCKET_NAME = "datasets"
SPECIFIC_PATH = "ranks/00005/data"  

# New field to add
NEW_FIELD = "image_source"
NEW_VALUE = "generated_image"

def update_minio_objects():
    # Initialize MinIO client
    client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)

    # Print the full path where objects are being listed
    full_path = f"{BUCKET_NAME}/{SPECIFIC_PATH}"
    print(f"Searching objects in path: {full_path}")

    # List all objects in the specific path within the bucket
    objects = client.list_objects(BUCKET_NAME, prefix=SPECIFIC_PATH, recursive=True)

    for obj in objects:
        print(f"Processing object: {obj.object_name}")  # Print the object name

        # Get the object
        response = client.get_object(BUCKET_NAME, obj.object_name)
        data = response.read()
        response.close()
        response.release_conn()

        # Convert JSON data to a Python dictionary
        try:
            json_data = json.loads(data)
        except json.JSONDecodeError:
            print(f"Skipping non-JSON object: {obj.object_name}")
            continue

        # Create an OrderedDict to maintain the order and add the new field before "datetime"
        updated_data = OrderedDict()
        for key, value in json_data.items():
            if key == "datetime":
                updated_data[NEW_FIELD] = NEW_VALUE
            updated_data[key] = value

        # Convert back to JSON
        updated_json_data = json.dumps(updated_data, indent=4).encode('utf-8')
        updated_data_stream = BytesIO(updated_json_data)

        # Upload the updated object
        client.put_object(
            BUCKET_NAME, 
            obj.object_name, 
            updated_data_stream, 
            length=len(updated_json_data), 
            content_type='application/json'
        )
        print(f"Updated object: {obj.object_name}")

    print("All objects updated successfully.")

if __name__ == "__main__":
    update_minio_objects()