from fastapi import Request, HTTPException, APIRouter, Response, Query, status
from datetime import datetime, timedelta
import math
import random
import pymongo
from utility.minio import cmd
from orchestration.api.mongo_schemas import  ActiveLearningPolicy
from .api_utils import PrettyJSONResponse
import os
from fastapi.responses import JSONResponse
from pymongo.collection import Collection
from datetime import datetime, timezone
from typing import List

router = APIRouter()


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
       

