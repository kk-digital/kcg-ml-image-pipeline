import os
import json
from minio import Minio
import argparse
from dotenv import dotenv_values

# Load configuration from the .env file
# This configuration typically contains MinIO connection details
config = dotenv_values("./orchestration/api/.env")

# Initialize MinIO client using the configuration settings
# MinIO client is used for interacting with the MinIO server
minio_client = Minio(
    config["MINIO_ADDRESS"],
    access_key=config["MINIO_ACCESS_KEY"],
    secret_key=config["MINIO_SECRET_KEY"],
    secure=False  # Update this based on your MinIO server's configuration
)

class JobLoader:
    def __init__(self, minio_client, bucket_name):
        """Initialize the JobLoader with MinIO client and bucket name."""
        self.minio_client = minio_client
        self.bucket_name = bucket_name

    def load_job_from_minio(self, dataset, folder_name, image_name):
        """Load a specific job from MinIO based on the dataset, folder, and image name."""
        json_path = f"{dataset}/job/{folder_name}/{image_name}.json"
        try:
            # Retrieve the JSON object from MinIO
            response = self.minio_client.get_object(self.bucket_name, json_path)
            job_data = response.read().decode('utf-8')
            job = json.loads(job_data)
            pretty_job = json.dumps(job, ensure_ascii=False, indent=4)
            print(pretty_job)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in file: {json_path}. Error: {e}")
        except Exception as e:
            print(f"Error loading file {json_path}: {str(e)}")

    def load_all_jobs_from_minio(self, dataset):
        """Load all jobs from MinIO for a given dataset."""
        job_count = 0
        folder_number = 1
        while True:
            # Iterate through folders in the dataset
            folder_name = str(folder_number).zfill(4)
            prefix = f"{dataset}/job/{folder_name}/"
            objects = self.minio_client.list_objects(self.bucket_name, prefix=prefix, recursive=True)
            objects_list = list(objects)

            # Break the loop if no objects are found in the current folder
            if not objects_list:
                break

            # Load each job in the folder
            for obj in objects_list:
                if obj.object_name.endswith('.json'):
                    image_name = obj.object_name.split('/')[-1].split('.')[0]
                    self.load_job_from_minio(dataset, folder_name, image_name)
                    job_count += 1

            folder_number += 1  # Move to the next folder

        print(f"Total jobs loaded for dataset '{dataset}': {job_count}")

def main():
    """Main function to handle command-line arguments and execute job loading."""
    parser = argparse.ArgumentParser(description='Load jobs from MinIO for a given dataset.')
    parser.add_argument('dataset', type=str, help='The name of the dataset to load jobs from.')
    args = parser.parse_args()

    # Create an instance of JobLoader and load jobs from the specified dataset
    job_loader = JobLoader(minio_client, "datasets")
    job_loader.load_all_jobs_from_minio(args.dataset)

if __name__ == "__main__":
    main()
