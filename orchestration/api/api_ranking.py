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


@router.get("/ranking/list-selection-policies")
def list_policies(request: Request):
    # hard code policies for now
    policies = ["random-uniform",
                "top k variance",
                "error sampling",
                "previously ranked"]

    return policies


@router.post("/rank/add-ranking-data-point")
def add_selection_datapoint(
    request: Request, 
    selection: Selection,
    dataset: str = Query(...)  # dataset now as a query parameter  
):
    time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    selection.datetime = time

    # prepare path
    file_name = "{}-{}.json".format(time, selection.username)
    path = "data/ranking/aggregate"
    full_path = os.path.join(dataset, path, file_name)

    # convert to bytes
    dict_data = selection.to_dict()
    json_data = json.dumps(dict_data, indent=4).encode('utf-8')
    data = BytesIO(json_data)

    # upload
    cmd.upload_data(request.app.minio_client, "datasets", full_path, data)

    image_1_hash = selection.image_1_metadata.file_hash
    image_2_hash = selection.image_2_metadata.file_hash

    # update rank count
    # get models counter
    for img_hash in [image_1_hash, image_2_hash]:
        update_image_rank_use_count(request, img_hash)

    return True


@router.post("/rank/update-image-rank-use-count", description="Update image rank use count")
def update_image_rank_use_count(request: Request, image_hash):
    counter = request.app.image_rank_use_count_collection.find_one({"image_hash": image_hash})

    if counter is None:
        # add
        count = 1
        rank_use_count_data = {"image_hash": image_hash,
                               "count": count,
                               }

        request.app.image_rank_use_count_collection.insert_one(rank_use_count_data)
    else:
        count = counter["count"]
        count += 1

        try:
            request.app.image_rank_use_count_collection.update_one(
                {"image_hash": image_hash},
                {"$set": {"count": count}})
        except Exception as e:
            raise Exception("Updating of model counter failed: {}".format(e))

    return True


@router.post("/rank/set-image-rank-use-count", description="Set image rank use count")
def set_image_rank_use_count(request: Request, image_hash, count: int):
    counter = request.app.image_rank_use_count_collection.find_one({"image_hash": image_hash})

    if counter is None:
        # add
        rank_use_count_data = {"image_hash": image_hash,
                               "count": count,
                               }

        request.app.image_rank_use_count_collection.insert_one(rank_use_count_data)
    else:
        try:
            request.app.image_rank_use_count_collection.update_one(
                {"image_hash": image_hash},
                {"$set": {"count": count}})
        except Exception as e:
            raise Exception("Updating of model counter failed: {}".format(e))

    return True


@router.get("/rank/get-image-rank-use-count", description="Get image rank use count")
def get_image_rank_use_count(request: Request, image_hash: str):
    # check if exist
    query = {"image_hash": image_hash}

    item = request.app.image_rank_use_count_collection.find_one(query)
    if item is None:
        raise HTTPException(status_code=404, detail="Image rank use count data not found")

    return item["count"]


@router.post("/ranking/submit-relevance-data")
def add_relevancy_selection_datapoint(request: Request, relevance_selection: RelevanceSelection, dataset: str = Query(...)):
    time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    relevance_selection.datetime = time

    # prepare path
    file_name = "{}-{}.json".format(time, relevance_selection.username)
    path = "data/relevancy/aggregate"
    full_path = os.path.join(dataset, path, file_name)

    # convert to bytes
    dict_data = relevance_selection.to_dict()
    json_data = json.dumps(dict_data, indent=4).encode('utf-8')
    data = BytesIO(json_data)

    # upload
    cmd.upload_data(request.app.minio_client, "datasets", full_path, data)

    return True

@router.get("/queue-ranking/upload/{job_uuid}")
def get_job_details(request: Request, job_uuid: str):
    # Find the job in the completed_jobs_collection
    job = request.app.completed_jobs_collection.find_one({"uuid": job_uuid})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Extract the bucket name, dataset name, file name, and subfolder from the output_file_path
    output_file_path = job["task_output_file_dict"]["output_file_path"]
    path_parts = output_file_path.split('/')
    if len(path_parts) < 4:
        raise HTTPException(status_code=500, detail="Invalid output file path format")

    bucket_name = "datasets"
    dataset_name = path_parts[1]
    subfolder_name = path_parts[2]  # Subfolder name from the path
    original_file_name = path_parts[-1]
    file_name_without_extension = original_file_name.split('.')[0]

    # Add the date_added to job details
    date_added = datetime.now().isoformat()
    job_details = {
        "job_uuid": job_uuid,
        "dataset_name": dataset_name,
        "file_name": original_file_name,
        "image_path": output_file_path,
        "image_hash": job["task_output_file_dict"]["output_file_hash"],
        "date_added": date_added,
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
async def get_random_json(request: Request, dataset: str = Query(...)):
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