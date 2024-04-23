from fastapi import Request, HTTPException, APIRouter, Response, Query, status
from datetime import datetime, timedelta
import math
import random
import pymongo
from utility.minio import cmd
from orchestration.api.mongo_schema.active_learning_schemas import  ActiveLearningPolicy, RequestActiveLearningPolicy, ListActiveLearningPolicy
from orchestration.api.mongo_schemas import RankActiveLearningPolicy, RequestRankActiveLearningPolicy, ListRankActiveLearningPolicy
from .api_utils import PrettyJSONResponse, ApiResponseHandler, ErrorCode, StandardSuccessResponse, WasPresentResponse, ApiResponseHandlerV1, StandardSuccessResponseV1
import os
from datetime import datetime, timezone
from typing import List
from io import BytesIO
import json
from fastapi.encoders import jsonable_encoder
from bson import ObjectId


router = APIRouter()

@router.post("/rank-active-learning-queue/add-new-policy", 
             status_code=200,
             response_model=StandardSuccessResponseV1[RankActiveLearningPolicy],
             tags=["rank active learning policy"],
             description="Add a new rank active learning policy",
             responses=ApiResponseHandlerV1.listErrors([400,422]))
async def add_rank_active_learning_policy(request: Request, policy_data: RequestRankActiveLearningPolicy):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    last_entry = request.app.rank_active_learning_policies_collection.find_one({}, sort=[("rank_active_learning_policy_id", -1)])
    new_policy_id = last_entry["rank_active_learning_policy_id"] + 1 if last_entry else 0

    if request.app.rank_active_learning_policies_collection.find_one({"rank_active_learning_policy": policy_data.rank_active_learning_policy}):
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="Policy already exists",
            http_status_code=400
        )

    creation_time = datetime.now(timezone.utc)
    policy_to_insert = policy_data.dict()
    policy_to_insert["rank_active_learning_policy_id"] = new_policy_id
    policy_to_insert["creation_time"] = creation_time

    request.app.rank_active_learning_policies_collection.insert_one(policy_to_insert)


    response_data = {
        "rank_active_learning_policy_id": new_policy_id,
        "rank_active_learning_policy": policy_data.rank_active_learning_policy,
        "rank_active_learning_policy_description": policy_data.rank_active_learning_policy_description,
        "creation_time": creation_time.isoformat(),
    }

    return api_response_handler.create_success_response_v1(response_data=response_data, http_status_code=200)


@router.put("/rank-active-learning-queue/update-rank-active-learning-policy",
            status_code=200,
            tags=["rank active learning policy"],
            description="Update an existing rank active learning policy",
            response_model=StandardSuccessResponseV1[RankActiveLearningPolicy], 
            responses=ApiResponseHandlerV1.listErrors([400, 404, 422]))
async def update_rank_active_learning_policy(request: Request, policy_id: int, policy_update: RequestRankActiveLearningPolicy):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    
    # Fetch the existing policy to ensure it exists and get its creation time
    existing_policy = request.app.rank_active_learning_policies_collection.find_one({"rank_active_learning_policy_id": policy_id})
    if not existing_policy:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.ELEMENT_NOT_FOUND,
            error_string="Policy not found",
            http_status_code=404
        )

    # Prepare the update data without altering the policy ID
    update_data = policy_update.dict(exclude_unset=True)
    
    # Perform the update operation, excluding the ID from changes
    update_result = request.app.rank_active_learning_policies_collection.update_one(
        {"rank_active_learning_policy_id": policy_id},
        {"$set": update_data}
    )

    if update_result.modified_count == 0:
        # Handle the case where no changes were made, potentially because the policy was not found
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string="No changes applied or policy not found",
            http_status_code=400
        )

    # Construct the response data, including the unchanged policy ID and creation time
    response_data = {
        "rank_active_learning_policy_id": policy_id,
        "rank_active_learning_policy": update_data.get("rank_active_learning_policy", existing_policy["rank_active_learning_policy"]),
        "rank_active_learning_policy_description": update_data.get("rank_active_learning_policy_description", existing_policy["rank_active_learning_policy_description"]),
        "creation_time": existing_policy["creation_time"].isoformat()  # Assuming the creation_time is stored in the existing_policy
    }

    # Return a success response with the updated policy details
    return api_response_handler.create_success_response_v1(
        response_data=response_data,
        http_status_code=200
    )


@router.get("/rank-active-learning-queue/list-rank-active-learning-policies",
            response_model=StandardSuccessResponseV1[ListRankActiveLearningPolicy],  
            tags=["rank active learning policy"],
            description="List rank active learning policies",
            responses=ApiResponseHandlerV1.listErrors([422, 500]))
def list_rank_active_learning_policies(request: Request):
    api_response_handler = ApiResponseHandlerV1(request)
    
    try:
        policies_cursor = request.app.rank_active_learning_policies_collection.find({})
        policies_list = list(policies_cursor)

        # Convert ObjectId and datetime to strings directly in the list comprehension
        policies_list = [{k: v.isoformat() if isinstance(v, datetime) else str(v) if isinstance(v, ObjectId) else v for k, v in policy.items()} for policy in policies_list]

        if not policies_list:
            return api_response_handler.create_success_response([], 200)

        return api_response_handler.create_success_response_v1(response_data={"policies":policies_list}, http_status_code=200)
    except Exception as e:
        return api_response_handler.create_error_response_v1(error_code=ErrorCode.OTHER_ERROR, error_string=str(e), http_status_code=500)
    

@router.delete("/rank-active-learning-queue/remmove-active-learning-policy", 
               response_model=StandardSuccessResponseV1[WasPresentResponse],
               status_code=200,
               tags=["rank active learning policy"],
               description="Removing an existing policy by ID",
               responses=ApiResponseHandlerV1.listErrors([422, 500]))
def delete_rank_active_learning_policy(request: Request, policy_id: int ):
    response_handler = ApiResponseHandlerV1(request)
    
    try:
        was_present = False

        # Check if the policy is being used by any queue pairs
        if request.app.active_learning_queue_pairs_collection.find_one({"rank_active_learning_policy_id": policy_id}):
            return response_handler.create_error_response_v1(error_code=ErrorCode.INVALID_PARAMS, error_string="Cannot delete policy: It is being used by one or more queue pairs", http_status_code=400)

        # Check if the policy exists and delete it
        policy = request.app.rank_active_learning_policies_collection.find_one({"rank_active_learning_policy_id": policy_id})
        if policy:
            was_present = True
            request.app.rank_active_learning_policies_collection.delete_one({"rank_active_learning_policy_id": policy_id})

        # Return a success response indicating if the policy was deleted
        return response_handler.create_success_delete_response_v1( was_present, 200)
    except Exception as e:
        # Handle any unexpected errors
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal server error", 500)    