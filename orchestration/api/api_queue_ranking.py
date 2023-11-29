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


@router.post("/ranking-queue/add-image-to-queue")
def get_job_details(request: Request, job_uuid: str = Query(...), policy: str = Query(...)):  # Use Query to specify that job_uuid is a query parameter
    job = request.app.completed_jobs_collection.find_one({"uuid": job_uuid})
    if not job:
        print("Job not found")

    # Extract the bucket name, dataset name, file name, and subfolder from the output_file_path
    output_file_path = job["task_output_file_dict"]["output_file_path"]
    task_creation_time = job["task_creation_time"]
    #prompt_generation_policy = job["prompt_generation_policy"]
    creation_date = datetime.fromisoformat(task_creation_time).strftime("%Y-%m-%d")
    path_parts = output_file_path.split('/')
    if len(path_parts) < 4:
        raise HTTPException(status_code=500, detail="Invalid output file path format")

    bucket_name = "datasets"
    dataset_name = path_parts[1]
    subfolder_name = path_parts[2]  # Subfolder name from the path
    original_file_name = path_parts[-1]
    file_name_without_extension = original_file_name.split('.')[0]

    # Add the date_added to job details
    #date_added = datetime.now().isoformat()
    job_details = {
        "job_uuid": job_uuid,
        "dataset_name": dataset_name,
        "file_name": original_file_name,
        "image_path": output_file_path,
        "image_hash": job["task_output_file_dict"]["output_file_hash"],
        "policy": policy,
        "job_creation_time": task_creation_time,
        "put_type" : "single-image"
    }

    # Serialize job details to JSON
    json_data = json.dumps(job_details, indent=4).encode('utf-8')
    data = BytesIO(json_data)

    # Prepare path using the subfolder and the original file name for the JSON file
    json_file_name = f"{creation_date}_{file_name_without_extension}.json"
    path = "ranking-queue-image"
    full_path = os.path.join(dataset_name, path, policy, subfolder_name, json_file_name)

    # Upload to MinIO
    cmd.upload_data(request.app.minio_client, bucket_name, full_path, data)

    return True


@router.post("/ranking-queue/add-image-pair-to-queue")
def get_job_details(request: Request, job_uuid_1: str = Query(...), job_uuid_2: str = Query(...), policy: str = Query(...)):
    def extract_job_details(job_uuid, suffix, policy):
        job = request.app.completed_jobs_collection.find_one({"uuid": job_uuid})
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_uuid} not found")

        output_file_path = job["task_output_file_dict"]["output_file_path"]
        task_creation_time = job["task_creation_time"]
        path_parts = output_file_path.split('/')
        if len(path_parts) < 4:
            raise HTTPException(status_code=500, detail="Invalid output file path format")

        original_file_name = path_parts[-1]

        return {
            f"job_uuid_{suffix}": job_uuid,
            "dataset_name": path_parts[1],
            f"file_name_{suffix}": original_file_name,
            f"image_path_{suffix}": output_file_path,
            f"image_hash_{suffix}": job["task_output_file_dict"]["output_file_hash"],
            "policy": policy, 
            f"job_creation_time_{suffix}": task_creation_time,
            "put_type": "pair-image"
        }

    # Extract details for both jobs
    job_details_1 = extract_job_details(job_uuid_1, "1", policy)
    job_details_2 = extract_job_details(job_uuid_2, "2", policy)

    # Create a list with two separate dictionaries
    combined_job_details = [job_details_1, job_details_2]

    # Serialize to JSON
    json_data = json.dumps(combined_job_details, indent=4).encode('utf-8')
    data = BytesIO(json_data)

    # Format the date from the first job's task_creation_time
    creation_date_1 = datetime.fromisoformat(job_details_1["job_creation_time_1"]).strftime("%Y-%m-%d")
    creation_date_2 = datetime.fromisoformat(job_details_2["job_creation_time_2"]).strftime("%Y-%m-%d")


    # Define the path for the JSON file with the formatted date
    base_file_name_1 = job_details_1['file_name_1'].split('.')[0]
    base_file_name_2 = job_details_2['file_name_2'].split('.')[0]
    json_file_name = f"{creation_date_1}_{base_file_name_1}_and_{creation_date_2}_{base_file_name_2}.json"
    full_path = os.path.join(job_details_1['dataset_name'], "ranking-queue-pair", policy, json_file_name)

    # Upload to MinIO
    cmd.upload_data(request.app.minio_client, "datasets", full_path, data)

    return True





@router.get("/ranking-queue/get-random-image", response_class=PrettyJSONResponse)
def get_random_json(request: Request, dataset: str = Query(...)):
    minio_client = request.app.minio_client
    bucket_name = "datasets"
    prefix = f"{dataset}/ranking-queue-image/"

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


@router.get("/ranking-queue/get-random-image-pair", response_class=PrettyJSONResponse)
def get_random_image_pair(request: Request, dataset: str = Query(...)):
    minio_client = request.app.minio_client
    bucket_name = "datasets"
    prefix = f"{dataset}/ranking-queue-pair/"

    # List all json files in the ranking-queue-pair directory
    json_files = cmd.get_list_of_objects_with_prefix(minio_client, bucket_name, prefix)
    json_files = [name for name in json_files if name.endswith('.json') and prefix in name]

    if not json_files:
        raise HTTPException(status_code=404, detail="No image pair JSON files found for the given dataset")

    # Randomly select a json file
    random_file_name = random.choice(json_files)

    # Get the file content from MinIO
    data = cmd.get_file_from_minio(minio_client, bucket_name, random_file_name)
    if data is None:
        raise HTTPException(status_code=500, detail="Failed to retrieve file from MinIO")

    # Read the content of the json file
    json_content = data.read().decode('utf-8')

    # Parse JSON content to ensure it is properly formatted
    try:
        json_data = json.loads(json_content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON content")

    return json_data
