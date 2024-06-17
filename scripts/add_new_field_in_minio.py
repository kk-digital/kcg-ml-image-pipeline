from minio import Minio
from io import BytesIO
import json
from collections import OrderedDict

# MinIO connection details
MINIO_ENDPOINT = "192.168.3.5:9000"
MINIO_ACCESS_KEY = "v048BpXpWrsVIHUfdAix"
MINIO_SECRET_KEY = "4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu"
BUCKET_NAME = "datasets"
SPECIFIC_PATH = "test_minio/"  

# New field to add
NEW_FIELD = "image_source"
NEW_VALUE = "generated_image"

def update_minio_objects():
    # Initialize MinIO client
    client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=True)

    # List all objects in the specific path within the bucket
    objects = client.list_objects(BUCKET_NAME, prefix=SPECIFIC_PATH, recursive=True)

    for obj in objects:
        # Get the object
        response = client.get_object(BUCKET_NAME, obj.object_name)
        data = response.read()
        response.close()
        response.release_conn()

        # Convert JSON data to a Python dictionary
        json_data = json.loads(data)

        # Create an OrderedDict with the new field
        updated_data = OrderedDict([
            ("file_name", json_data.get("file_name", "")),
            *json_data.items(),  # Unpack the rest of the document fields
            (NEW_FIELD, NEW_VALUE),
            ("datetime", json_data.get("datetime", ""))
        ])

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
