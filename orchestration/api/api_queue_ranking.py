from fastapi import Request, APIRouter, Query, HTTPException
from datetime import datetime
from utility.minio import cmd
import os
import json
from io import BytesIO
from orchestration.api.mongo_schemas import Selection, RelevanceSelection
from .api_utils import PrettyJSONResponse
import random

router = APIRouter()


@router.post("/queue-ranking/upload")
def get_job_details(request: Request, job_uuid: str = Query(...)):  # Use Query to specify that job_uuid is a query parameter
    job = request.app.completed_jobs_collection.find_one({"uuid": job_uuid})
    if not job:
        print("Job not found")

    # Extract the bucket name, dataset name, file name, and subfolder from the output_file_path
    output_file_path = job["task_output_file_dict"]["output_file_path"]
    task_creation_time = job["task_creation_time"]
    path_parts = output_file_path.split('/')
    if len(path_parts) < 4:
        raise HTTPException(status_code=500, detail="Invalid output file path format")

    bucket_name = "datasets"
    dataset_name = path_parts[1]
    subfolder_name = path_parts[2]  # Subfolder name from the path
    original_file_name = path_parts[-1]
    file_name_without_extension = original_file_name.split('.')[0]

    # Add the date_added to job details
    job_details = {
        "job_uuid": job_uuid,
        "dataset_name": dataset_name,
        "file_name": original_file_name,
        "image_path": output_file_path,
        "image_hash": job["task_output_file_dict"]["output_file_hash"],
        "job_creation_time": task_creation_time,
        "put_type" : "single-image"
    }

    # Serialize job details to JSON
    json_data = json.dumps(job_details, indent=4).encode('utf-8')
    data = BytesIO(json_data)

    # Prepare path using the subfolder and the original file name for the JSON file
    json_file_name = f"{file_name_without_extension}.json"
    path = "queue-ranking"
    full_path = os.path.join(dataset_name, path, subfolder_name, json_file_name)

    # Upload to MinIO
    cmd.upload_data(request.app.minio_client, bucket_name, full_path, data)

    return True



@router.get("/queue-ranking/get-random-json", response_class=PrettyJSONResponse)
def get_random_json(request: Request, dataset: str = Query(...)):
    minio_client = request.app.minio_client
    bucket_name = "datasets"
    prefix = f"{dataset}/queue-ranking/"

    # List all json files in the queue-ranking directory
    json_files = cmd.get_list_of_objects_with_prefix(minio_client, bucket_name, prefix)
    json_files = [name for name in json_files if name.endswith('.json') and prefix in name]

    if not json_files:
        raise HTTPException(status_code=404, detail="No JSON files found for the given dataset")

    # Randomly select a json file
    random_file_name = random.choice(json_files)

    # Get the file content from MinIO
    data = cmd.get_file_from_minio(minio_client, bucket_name, random_file_name)
    if data is None:
        raise HTTPException(status_code=500, detail="Failed to retrieve file from MinIO")

    # Read the content of the json file
    json_content = data.read().decode('utf-8')

    # Parse JSON content to ensure it is properly formatted JSON
    try:
        json_data = json.loads(json_content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON content")

    # Assuming you want to return the JSON content directly
    return json_data
