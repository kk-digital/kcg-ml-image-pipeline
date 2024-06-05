from minio import Minio
from minio.error import S3Error
from minio.commonconfig import CopySource


# Initialize MinIO client
minio_client = Minio(
    "192.168.3.5:9000",  # e.g., "play.min.io"
    access_key="v048BpXpWrsVIHUfdAix",
    secret_key="4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu",
    secure=False  # or True if your MinIO server is using HTTPS
)

print("connected minio")

bucket_name = "datasets"
prefix = "ranks/"

# List all objects under the given prefix
objects = minio_client.list_objects(bucket_name, prefix, recursive=True)
print(objects)

# Find unique folder names
unique_folders = set()
for obj in objects:
    folder = obj.object_name.split('/')[1]
    unique_folders.add(folder)

# Rename folders to zero-padded format
for folder in sorted(unique_folders):
    padded_folder = f"{int(folder):05d}"
    if folder != padded_folder:
        # List objects in the old folder
        old_objects = minio_client.list_objects(bucket_name, f"{prefix}{folder}/", recursive=True)
        
        for obj in old_objects:
            old_object_name = obj.object_name
            new_object_name = old_object_name.replace(f"{prefix}{folder}/", f"{prefix}{padded_folder}/", 1)
            
            # Copy object to the new location
            copy_source = CopySource(bucket_name, old_object_name)
            minio_client.copy_object(
                bucket_name,
                new_object_name,
                copy_source
            )
            
            # Remove the old object
            minio_client.remove_object(bucket_name, old_object_name)
        
        print(f"Renamed folder {folder} to {padded_folder}")

print("All folders have been renamed successfully.")
