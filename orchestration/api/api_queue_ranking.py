from fastapi import Request, APIRouter, Query, HTTPException
from datetime import datetime, timezone
from utility.minio import cmd
import os
import json
from io import BytesIO
from orchestration.api.mongo_schemas import Selection, RelevanceSelection
from .api_utils import PrettyJSONResponse
import random

router = APIRouter()


@router.post("/ranking-queue/add-image-to-queue")
def get_job_details(request: Request, job_uuid: str = Query(...), policy: str = Query(...)):
    job = request.app.completed_jobs_collection.find_one({"uuid": job_uuid})
    if not job:
        print("Job not found")

    output_file_path = job["task_output_file_dict"]["output_file_path"]
    task_creation_time = job["task_creation_time"]
    path_parts = output_file_path.split('/')
    if len(path_parts) < 4:
        raise HTTPException(status_code=500, detail="Invalid output file path format")

    # UTC date and time when the JSON is created
    creation_date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    datetime.now()
    job_details = {
        "job_uuid": job_uuid,
        "dataset_name": path_parts[1],
        "file_name": path_parts[-1],
        "image_path": output_file_path,
        "image_hash": job["task_output_file_dict"]["output_file_hash"],
        "active-learning-policy": policy,
        "creation_date": creation_date_utc,
        "job_creation_time": task_creation_time,
        "put_type": "single-image",
        "creation_date": creation_date_utc  # Adding creation_date in UTC+0
    }

    json_data = json.dumps(job_details, indent=4).encode('utf-8')
    data = BytesIO(json_data)

    json_file_name = f"{datetime.fromisoformat(task_creation_time).strftime('%Y-%m-%d')}_{path_parts[-1].split('.')[0]}.json"
    full_path = os.path.join(path_parts[1], "ranking-queue-image", policy, path_parts[2], json_file_name)

    cmd.upload_data(request.app.minio_client, "datasets", full_path, data)

    return True


@router.post("/ranking-queue/add-image-pair-to-queue")
def get_job_details(request: Request, job_uuid_1: str = Query(...), job_uuid_2: str = Query(...), policy: str = Query(...)):
    def extract_job_details(job_uuid, suffix):
        job = request.app.completed_jobs_collection.find_one({"uuid": job_uuid})
        if not job:
            raise HTTPException(status_code=422, detail=f"Job {job_uuid} not found")
           

        output_file_path = job["task_output_file_dict"]["output_file_path"]
        task_creation_time = job["task_creation_time"]
        path_parts = output_file_path.split('/')
        if len(path_parts) < 4:
            raise HTTPException(status_code=500, detail="Invalid output file path format")

        return {
            f"job_uuid_{suffix}": job_uuid,
            f"file_name_{suffix}": path_parts[-1],
            f"image_path_{suffix}": output_file_path,
            f"image_hash_{suffix}": job["task_output_file_dict"]["output_file_hash"],
            f"job_creation_time_{suffix}": task_creation_time,
        }

    job_details_1 = extract_job_details(job_uuid_1, "1")
    job_details_2 = extract_job_details(job_uuid_2, "2")

    if not job_details_1 or not job_details_2:
        return False

    combined_job_details = {
        "active_learning_policy": policy,
        "put_type": "pair-image",
        "dataset_name": job_details_1['image_path_1'].split('/')[1],  # Extract dataset name
        "creation_date": datetime.utcnow().isoformat(),  # UTC time
        "images": [job_details_1, job_details_2]
    }

    creation_date_1 = datetime.fromisoformat(job_details_1["job_creation_time_1"]).strftime("%Y-%m-%d")
    creation_date_2 = datetime.fromisoformat(job_details_2["job_creation_time_2"]).strftime("%Y-%m-%d")

    json_data = json.dumps([combined_job_details], indent=4).encode('utf-8')  # Note the list brackets around combined_job_details
    data = BytesIO(json_data)

    # Define the path for the JSON file
    base_file_name_1 = job_details_1['file_name_1'].split('.')[0]
    base_file_name_2 = job_details_2['file_name_2'].split('.')[0]
    json_file_name = f"{creation_date_1}_{base_file_name_1}_and_{creation_date_2}_{base_file_name_2}.json"
    full_path = f"{combined_job_details['dataset_name']}/ranking-queue-pair/{policy}/{json_file_name}"

    cmd.upload_data(request.app.minio_client, "datasets", full_path, data)

    return True



@router.get("/ranking-queue/get-random-image-v1", response_class=PrettyJSONResponse)
def get_random_json(request: Request, dataset: str = Query(...), size: int = Query(...), policy: str = Query(...)):
    minio_client = request.app.minio_client
    bucket_name = "datasets"
    prefix = f"{dataset}/ranking-queue-image/{policy}"

    # List all json files in the queue-ranking directory
    json_files = cmd.get_list_of_objects_with_prefix(minio_client, bucket_name, prefix)
    json_files = [name for name in json_files if name.endswith('.json') and prefix in name]

    if not json_files:
        print("No JSON files found for the given dataset")

    # Randomly select 'size' number of json files
    selected_files = random.sample(json_files, min(size, len(json_files)))

    results = []
    for file_path in selected_files:
        # Get just the filename without the path
        file_name = os.path.basename(file_path)

        # Get the file content from MinIO
        data = cmd.get_file_from_minio(minio_client, bucket_name, file_path)
        if data is None:
            continue  # Skip if file not found or error occurs

        # Read and parse the content of the json file
        json_content = data.read().decode('utf-8')
        try:
            json_data = json.loads(json_content)
            # Construct the result with the 'json_file_name' and the rest of the content
            result = {
                'json_file_name': file_name
            }
            result.update(json_data)  # Merge the content under the same dictionary
            results.append(result)
        except json.JSONDecodeError:
            continue  # Skip on JSON decode error

    return results


@router.get("/ranking-queue/get-random-image-pair-v1", response_class=PrettyJSONResponse)
def get_random_image_pair(request: Request, dataset: str = Query(...), size: int = Query(...), policy: str = Query(...)):
    minio_client = request.app.minio_client
    bucket_name = "datasets"
    prefix = f"{dataset}/ranking-queue-pair/{policy}"

    # List all json files in the ranking-queue-pair directory
    json_files = cmd.get_list_of_objects_with_prefix(minio_client, bucket_name, prefix)
    json_files = [name for name in json_files if name.endswith('.json') and prefix in name]

    if not json_files:
        print("No image pair JSON files found for the given dataset")

    # Randomly select 'size' number of json files
    selected_files = random.sample(json_files, min(size, len(json_files)))

    results = []
    for file_path in selected_files:
        # Get just the filename without the path
        json_file_name = os.path.basename(file_path)

        # Get the file content from MinIO
        data = cmd.get_file_from_minio(minio_client, bucket_name, file_path)
        if data is None:
            continue  # Skip if file not found or error occurs

        # Read and parse the content of the json file
        json_content = data.read().decode('utf-8')
        try:
            json_data = json.loads(json_content)
            # Add the filename to each item in the pair
            pair_data = []
            for item in json_data:
                item_with_filename = {
                    'json_file_name': json_file_name
                }
                item_with_filename.update(item)
                pair_data.append(item_with_filename)
            results.append(pair_data)
        except json.JSONDecodeError:
            continue  # Skip on JSON decode error

    return results


@router.get("/ranking-queue/get-random-image-pair-v2", response_class=PrettyJSONResponse)
def get_random_image_pair(request: Request, dataset: str = Query(...), size: int = Query(...), policy: str = Query(...)):
    minio_client = request.app.minio_client
    bucket_name = "datasets"
    prefix = f"{dataset}/ranking-queue-pair/{policy}"

    # List all json files in the ranking-queue-pair directory
    json_files = cmd.get_list_of_objects_with_prefix(minio_client, bucket_name, prefix)
    json_files = [name for name in json_files if name.endswith('.json') and prefix in name]

    if not json_files:
        print("No image pair JSON files found for the given dataset")

    # Randomly select 'size' number of json files
    selected_files = random.sample(json_files, min(size, len(json_files)))

    results = []
    for file_path in selected_files:       
        # Get the file content from MinIO
        data = cmd.get_file_from_minio(minio_client, bucket_name, file_path)
        if data is None:
            continue  # Skip if file not found or error occurs

        # Read and parse the content of the json file
        json_content = data.read().decode('utf-8')
        try:
            json_data = json.loads(json_content)
            # Add the filename to each item in the pair
            pair_data = []
            for item in json_data:
                item_with_filename = {
                    'json_file_path': file_path
                }
                item_with_filename.update(item)
                pair_data.append(item_with_filename)
            results.append(pair_data)
        except json.JSONDecodeError:
            continue  # Skip on JSON decode error

    return results


@router.delete("/ranking-queue/remove-ranking-queue-single")
def remove_single_image_from_queue(request: Request, dataset: str = Query(...), policy: str = Query(...), filename: str = Query(...)):
    # Define bucket name and construct the base path with the dataset name
    minio_client = request.app.minio_client
    bucket_name = "datasets"
    base_path = f"{dataset}/ranking-queue-image/{policy}"  # Construct path including dataset name

    # List all objects in the bucket within the specified base path
    objects = minio_client.list_objects(bucket_name, prefix=base_path, recursive=True)
    
    # Find the object with the matching filename
    object_to_remove = None
    for obj in objects:
        if filename in obj.object_name:
            object_to_remove = obj.object_name
            break
    
    if object_to_remove:
        # Remove the object from MinIO
        cmd.remove_an_object(minio_client, bucket_name, object_to_remove)
        return {"status": "success", "message": "Image removed from queue"}
    else:
        return {"status": "fail"}

@router.delete("/ranking-queue/remove-ranking-queue-pair")
def remove_image_pair_from_queue(request: Request, dataset: str = Query(...), policy: str = Query(...), filename: str = Query(...)):
    # Define bucket name and construct the base path with the dataset name
    minio_client = request.app.minio_client
    bucket_name = "datasets"
    base_path = f"{dataset}/ranking-queue-pair/{policy}"  # Adjust base path for pairs

    # List all objects in the bucket within the specified base path
    objects = minio_client.list_objects(bucket_name, prefix=base_path, recursive=True)
    
    # Find the object with the matching filename pair
    object_to_remove = None
    for obj in objects:
        if filename in obj.object_name:
            object_to_remove = obj.object_name
            break
    
    if object_to_remove:
        # Remove the object from MinIO
        cmd.remove_an_object(minio_client, bucket_name, object_to_remove)
        return {"status": "success", "message": "Image pair removed from queue"}
    else:
        return {"status": "fail"}


@router.delete("/ranking-queue/remove-ranking-queue-pair-v1")
def remove_image_pair_from_queue(request: Request, json_file_path: str = Query(...)):
    minio_client = request.app.minio_client
    bucket_name = "datasets"

    # Check if the specified file exists in MinIO
    try:
        # Attempt to get the file to ensure it exists
        _ = cmd.get_file_from_minio(minio_client, bucket_name, json_file_path)
    except Exception as e:
        # If an error occurs (like file not found), raise an HTTP exception
        raise HTTPException(status_code=404, detail=f"File not found: {json_file_path}")

    # If the file exists, proceed to remove it
    cmd.remove_an_object(minio_client, bucket_name, json_file_path)

    return {"status": "success", "message": "Image pair removed from queue"}



@router.get("/ranking-queue/get-policy-list", response_class=PrettyJSONResponse)
def get_directory_names(request: Request, dataset: str, type: str):
    if type not in ["ranking-queue-pair", "ranking-queue-image"]:
        raise HTTPException(status_code=400, detail="Invalid type parameter")

    minio_client = request.app.minio_client
    bucket_name = "datasets"
    prefix = f"{dataset}/{type}"

    # List all objects with the prefix
    objects = cmd.get_list_of_objects_with_prefix(minio_client, bucket_name, prefix)

    # Extracting unique directory names
    directories = set()
    for obj in objects:
        path_parts = obj.split('/')
        if len(path_parts) > 2:  # Ensure there's a sub-directory
            directories.add(path_parts[2])

    return list(directories)



@router.get("/ranking-queue/count-single-images")
def count_image_pairs(
    request: Request,
    dataset: str = Query(default=None),
    policy: str = Query(default=None)
):
    minio_client = request.app.minio_client
    bucket_name = "datasets"
    
    try:
        # If both dataset and policy are specified
        if dataset and policy:
            prefix = f"{dataset}/ranking-queue-image/{policy}/"
            objects = minio_client.list_objects(bucket_name, prefix=prefix, recursive=True)
            count = sum(1 for _ in objects)
        # If only dataset is specified
        elif dataset:
            prefix = f"{dataset}/ranking-queue-image/"
            objects = minio_client.list_objects(bucket_name, prefix=prefix, recursive=True)
            count = sum(1 for _ in objects)
        # If only policy is specified or neither
        else:
            # Need to iterate over possible datasets to get the count
            count = 0
            objects = minio_client.list_objects(bucket_name, recursive=False)
            for obj in objects:
                # Check if the object name contains a '/' indicating it's a directory
                if '/' in obj.object_name:
                    ds = obj.object_name.split('/')[0]
                    # Construct the prefix
                    prefix = f"{ds}/ranking-queue-image/"
                    if policy:
                        prefix += f"{policy}/"
                    # List and count objects using the prefix
                    count += sum(1 for _ in minio_client.list_objects(bucket_name, prefix=prefix, recursive=True))

        return count
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")  


@router.get("/ranking-queue/count-image-pairs")
def count_image_pairs(
    request: Request,
    dataset: str = Query(default=None),
    policy: str = Query(default=None)
):
    minio_client = request.app.minio_client
    bucket_name = "datasets"
    
    try:
        # If both dataset and policy are specified
        if dataset and policy:
            prefix = f"{dataset}/ranking-queue-pair/{policy}/"
            objects = minio_client.list_objects(bucket_name, prefix=prefix, recursive=True)
            count = sum(1 for _ in objects)
        # If only dataset is specified
        elif dataset:
            prefix = f"{dataset}/ranking-queue-pair/"
            objects = minio_client.list_objects(bucket_name, prefix=prefix, recursive=True)
            count = sum(1 for _ in objects)
        # If only policy is specified or neither
        else:
            # Need to iterate over possible datasets to get the count
            count = 0
            objects = minio_client.list_objects(bucket_name, recursive=False)
            for obj in objects:
                # Check if the object name contains a '/' indicating it's a directory
                if '/' in obj.object_name:
                    ds = obj.object_name.split('/')[0]
                    # Construct the prefix
                    prefix = f"{ds}/ranking-queue-pair/"
                    if policy:
                        prefix += f"{policy}/"
                    # List and count objects using the prefix
                    count += sum(1 for _ in minio_client.list_objects(bucket_name, prefix=prefix, recursive=True))

        return  count
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")  
        



