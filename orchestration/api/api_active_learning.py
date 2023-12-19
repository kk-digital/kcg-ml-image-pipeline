from fastapi import Request, HTTPException, APIRouter, Response, Query, status
from datetime import datetime, timedelta
import math
import random
import pymongo
from utility.minio import cmd
from orchestration.api.mongo_schemas import  ActiveLearningPolicy, ActiveLearningQueuePair
from .api_utils import PrettyJSONResponse
import os
from fastapi.responses import JSONResponse
from pymongo.collection import Collection
from datetime import datetime, timezone
from typing import List
from io import BytesIO
import json

router = APIRouter()

@router.post("/active-learning-queue/add-queue-pair-mongo")
def add_queue_pair(request: Request, queue_pair: ActiveLearningQueuePair):
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

    job_details_1 = extract_job_details(queue_pair.image1_job_uuid, "1")
    job_details_2 = extract_job_details(queue_pair.image2_job_uuid, "2")

    combined_job_details = {
        "active_learning_policy_id": queue_pair.active_learning_policy_id,
        "active_learning_policy": queue_pair.active_learning_policy,
        "dataset_name": job_details_1['image_path_1'].split('/')[1],
        "metadata": queue_pair.metadata,
        "generator_string": queue_pair.generator_string,
        "creation_time": datetime.utcnow().isoformat() if not queue_pair.creation_time else queue_pair.creation_time,
        "images": [
            {
                "job_uuid_1": job_details_1["job_uuid_1"],
                "file_name_1": job_details_1["file_name_1"],
                "image_path_1": job_details_1["image_path_1"],
                "image_hash_1": job_details_1["image_hash_1"],
                "job_creation_time_1": job_details_1["job_creation_time_1"],
            },
            {
                "job_uuid_2": job_details_2["job_uuid_2"],
                "file_name_2": job_details_2["file_name_2"],
                "image_path_2": job_details_2["image_path_2"],
                "image_hash_2": job_details_2["image_hash_2"],
                "job_creation_time_2": job_details_2["job_creation_time_2"],
            }
        ]
    }

    # Insert the combined job details into MongoDB collection
    request.app.active_learning_queue_pairs_collection.insert_one(combined_job_details)

    return {"status": "success", "message": "Queue pair added successfully to MongoDB"}

@router.get("/active-learning-queue/list-queue-pairs", response_class=PrettyJSONResponse)
def list_queue_pairs(request: Request, limit: int = 10, offset: int = 0) -> List[dict]:
    queue_pairs_cursor = request.app.active_learning_queue_pairs_collection.find().skip(offset).limit(limit)
    
    # Convert the cursor to a list of dictionaries and drop the _id field
    queue_pairs = []
    for pair in queue_pairs_cursor:
        # Drop the _id field from the response
        pair.pop('_id', None)
        queue_pairs.append(pair)

    # Directly return the list of modified dictionaries
    return queue_pairs

@router.delete("/active-learning-queue/delete-all-queue-pairs")
def delete_all_queue_pairs(request: Request):
    # Delete all documents in the collection
    result = request.app.active_learning_queue_pairs_collection.delete_many({})

    # Check if documents were deleted
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="No documents found to delete")

    return {"status": "success", "message": f"Deleted {result.deleted_count} queue pair(s) from MongoDB"}



@router.post("/active-learning-queue/add-queue-pair")
def add_queue_pair(request: Request, queue_pair: ActiveLearningQueuePair):
    # Extract job details for both UUIDs
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

    job_details_1 = extract_job_details(queue_pair.image1_job_uuid, "1")
    job_details_2 = extract_job_details(queue_pair.image2_job_uuid, "2")

    if not job_details_1 or not job_details_2:
        return False
    
    creation_date_1 = datetime.fromisoformat(job_details_1["job_creation_time_1"]).strftime("%Y-%m-%d")
    creation_date_2 = datetime.fromisoformat(job_details_2["job_creation_time_2"]).strftime("%Y-%m-%d")

    combined_job_details = {
        "active_learning_policy_id": queue_pair.active_learning_policy_id,
        "active_learning_policy": queue_pair.active_learning_policy,
        "dataset_name": job_details_1['image_path_1'].split('/')[1],
        "metadata": queue_pair.metadata,
        "generator_string": queue_pair.generator_string,
        "creation_time": datetime.utcnow().isoformat() if not queue_pair.creation_time else queue_pair.creation_time,
        "images": [job_details_1, job_details_2]
    }

    json_data = json.dumps([combined_job_details], indent=4).encode('utf-8')
    data = BytesIO(json_data)

    # Define the path for the JSON file
    base_file_name_1 = job_details_1['file_name_1'].split('.')[0]
    base_file_name_2 = job_details_2['file_name_2'].split('.')[0]
    json_file_name = f"{creation_date_1}_{base_file_name_1}_and_{creation_date_2}_{base_file_name_2}.json"
    full_path = f"{combined_job_details['dataset_name']}/ranking-queue-pair/{queue_pair.active_learning_policy}/{json_file_name}"

    # Upload the data to MinIO (or other storage as per your implementation)
    cmd.upload_data(request.app.minio_client, "datasets", full_path, data)

    return True

@router.get("/active-learning-queue/get-random-image-pair", response_class=PrettyJSONResponse)
def get_random_image_pair(request: Request, dataset: str = Query(...), size: int = Query(...), active_learning_policy: str = Query(...)):
    minio_client = request.app.minio_client
    bucket_name = "datasets"
    prefix = f"{dataset}/ranking-queue-pair/{active_learning_policy}"

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
    

@router.delete("/active-learning-queue/remove-ranking-queue-pair")
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


@router.put("/active-learning-policy/add-new-policy")
def add_or_update_active_learning_policy(request: Request, policy_data: ActiveLearningPolicy):

    # Find the maximum active_learning_policy_id in the collection
    last_entry = request.app.active_learning_policies_collection.find_one({}, sort=[("active_learning_policy_id", -1)])

    if last_entry and "active_learning_policy_id" in last_entry:
        new_policy_id = last_entry["active_learning_policy_id"] + 1
    else:
        new_policy_id = 0

    # Check if the active learning policy exists
    query = {"active_learning_policy": policy_data.active_learning_policy}
    existing_policy = request.app.active_learning_policies_collection.find_one(query)

    if existing_policy is None:
        # If policy doesn't exist, add it
        policy_data.active_learning_policy_id = new_policy_id
        policy_data.creation_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        request.app.active_learning_policies_collection.insert_one(policy_data.to_dict())
        return {"status": "success", "message": "Active learning policy added successfully.", "active_learning_policy_id": new_policy_id}
    else:
        # If policy already exists, update its details
        new_values = {
            "$set": {
                "active_learning_policy_description": policy_data.active_learning_policy_description,
                "creation_time": policy_data.creation_time
            }
        }
        request.app.active_learning_policies_collection.update_one(query, new_values)
        return {"status": "success", "message": "Active learning policy updated successfully.", "active_learning_policy_id": existing_policy["active_learning_policy_id"]}


@router.get("/active-learning-policy/list-policies", response_class=PrettyJSONResponse)
def list_active_learning_policies(request: Request) -> List[ActiveLearningPolicy]:
    # Retrieve all active learning policies from the collection
    policies_cursor = request.app.active_learning_policies_collection.find({})

    # Convert the cursor to a list of ActiveLearningPolicy objects
    policies = [ActiveLearningPolicy(**policy) for policy in policies_cursor]

    return policies


@router.delete("/active-learning-policy/remove-policies")
def delete_active_learning_policy(request: Request, active_learning_policy_id: int = None):
    if active_learning_policy_id is not None:
        # Delete a specific policy
        query = {"active_learning_policy_id": active_learning_policy_id}
        policy = request.app.active_learning_policies_collection.find_one(query)

        if not policy:
            # If the policy does not exist, return a 404 error
            raise HTTPException(status_code=404, detail="Policy not found")

        # Delete the specific policy
        request.app.active_learning_policies_collection.delete_one(query)
        return {"status": "success", "message": f"Policy with ID {active_learning_policy_id} deleted successfully."}
    else:
        # If no ID is provided, delete all policies
        request.app.active_learning_policies_collection.delete_many({})
        return {"status": "success", "message": "All policies deleted successfully."}


@router.get("/active-learning/uncertainty-sampling-pair-v1", response_class=PrettyJSONResponse)
def get_ranking_comparison(
    request: Request,
    dataset: str,  
    score_type: str,  # Added score_type parameter to choose between clip_sigma_score and embedding_sigma_score
    min_score: float,
    max_score: float,
    threshold: float
):
    if score_type not in ["clip_sigma_score", "embedding_sigma_score"]:
        raise HTTPException(status_code=400, detail="Invalid score_type parameter")

    image_scores_collection: Collection = request.app.image_scores_collection

    try:
        # Fetch a random image score within the score range and the specified dataset
        first_image_cursor = image_scores_collection.aggregate([
            {"$match": {
                "score": {"$gte": min_score, "$lte": max_score},
                "dataset": dataset  # Filter by dataset
            }},
            {"$sample": {"size": 1}}
        ])
        first_image_score = next(first_image_cursor, None)

        if not first_image_score:
            {"images": []}

        # Calculate the score range for the second image using the selected score_type
        base_score = first_image_score[score_type]  # Use dynamic score_type

        # Fetch candidate images for the second image within the specified dataset
        candidates_cursor = image_scores_collection.find({
            score_type: {"$gte": min_score, "$lte": max_score},
            "image_hash": {"$ne": first_image_score['image_hash']},
            "dataset": dataset  # Filter by dataset
        })

        # Compute probabilities using sigmoid function based on the score_type
        candidates = list(candidates_cursor)
        total_probability = 0
        for candidate in candidates:
            score_diff = abs(candidate[score_type] - base_score)  # Use dynamic score_type
            probability = 1 / (1 + math.exp((score_diff - threshold) / 50))
            candidate['probability'] = probability
            total_probability += probability

        # Select the second image based on computed probabilities
        if total_probability == 0:
            {"images": []}

        random_choice = random.uniform(0, total_probability)
        cumulative = 0
        for candidate in candidates:
            cumulative += candidate['probability']
            if cumulative >= random_choice:
                second_image_score = candidate
                break

    except StopIteration:
        return JSONResponse(
            status_code=500,
            content={"message": "Error fetching images from the database."}
        )

    # Prepare the images for the response
    images = [
        {
            "image_hash": first_image_score['image_hash'],
            "image_score": first_image_score[score_type]  # Use dynamic score_type
        },
        {
            "image_hash": second_image_score['image_hash'],
            "image_score": second_image_score[score_type]  # Use dynamic score_type
        }
    ]

    return {"images": images}


@router.get("/active-learning/uncertainty-sampling-pair-v2", response_class=PrettyJSONResponse)
def get_ranking_comparison(
    request: Request,
    dataset: str,  
    score_type: str,
    min_score: float,
    max_score: float,
    threshold: float
):
    if score_type not in ["image_clip_sigma_score", "text_embedding_sigma_score"]:
        raise HTTPException(status_code=422, detail="Invalid score_type parameter")
    
    completed_jobs_collection: Collection = request.app.completed_jobs_collection

    try:

        min_score = str(min_score)
        max_score = str(max_score)
        first_image_cursor = completed_jobs_collection.aggregate([
            {"$match": {
                "task_attributes_dict." + score_type: {"$gte": min_score, "$lte": max_score},
                "task_input_dict.dataset": dataset
            }},
            {"$sample": {"size": 1}}
        ])

        first_image_score = next(first_image_cursor, None)
        if not first_image_score:
            {"images": []}

        if 'task_attributes_dict' not in first_image_score or score_type not in first_image_score['task_attributes_dict']:
            print("task_attributes_dict not found in the fetched document")

        base_score = float(first_image_score['task_attributes_dict'][score_type])
        print(base_score)
        lower_bound = str(base_score - threshold)
        upper_bound = str(base_score + threshold)

        candidates_cursor = completed_jobs_collection.find({
            "task_attributes_dict." + score_type: {"$gte": str(lower_bound), "$lte": str(upper_bound)},
            "task_output_file_dict.output_file_hash": {"$ne": first_image_score['task_output_file_dict']['output_file_hash']},
            "task_input_dict.dataset": dataset
        })

        candidates = list(candidates_cursor)

        if not candidates:
            {"images": []}

        second_image_score = random.choice(candidates)

        images = [
            {
                "image_hash": first_image_score['task_output_file_dict']['output_file_hash'],
                "image_score": first_image_score['task_attributes_dict'][score_type]
            },
            {
                "image_hash": second_image_score['task_output_file_dict']['output_file_hash'],
                "image_score": second_image_score['task_attributes_dict'][score_type]
            }
        ]

        return {"images": images}

    except StopIteration:
        print("Error fetching images from the database.")
       

