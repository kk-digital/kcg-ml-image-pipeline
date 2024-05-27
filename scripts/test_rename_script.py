from minio import Minio
from minio.error import S3Error
import re
from minio.commonconfig import CopySource


# Initialize MinIO client
minio_client = Minio(
    "192.168.3.5:9000",  # e.g., "play.min.io"
    access_key="v048BpXpWrsVIHUfdAix",
    secret_key="4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu",
    secure=False  # or True if your MinIO server is using HTTPS
)

bucket_name = "datasets"
prefix = "test/"

# Regular expression pattern to match files with three '_vae_' substrings
pattern = re.compile(r"(_vae_vae_vae_)")

# List all objects under the given prefix
objects = minio_client.list_objects(bucket_name, prefix, recursive=True)

for obj in objects:
    if pattern.search(obj.object_name):
        # Replace three '_vae_' with two '_vae_'
        new_object_name = pattern.sub('_vae_vae_', obj.object_name, count=1)
        
        # Copy object to the new location
        copy_source = CopySource(bucket_name, obj.object_name)
        minio_client.copy_object(
            bucket_name,
            new_object_name,
            copy_source
        )
        
        # Remove the old object
        minio_client.remove_object(bucket_name, obj.object_name)
        
        print(f"Renamed {obj.object_name} to {new_object_name}")

print("All matching files have been renamed successfully.")
