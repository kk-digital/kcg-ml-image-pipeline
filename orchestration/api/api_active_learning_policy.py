from fastapi import Request, HTTPException, APIRouter, Response, Query, status
from datetime import datetime, timedelta
import math
import random
import pymongo
from utility.minio import cmd
from orchestration.api.mongo_schema.active_learning_schemas import  ActiveLearningPolicy, RequestActiveLearningPolicy
from .api_utils import PrettyJSONResponse, ApiResponseHandler, ErrorCode, StandardSuccessResponse, WasPresentResponse
import os
from datetime import datetime, timezone
from typing import List
from io import BytesIO
import json
from fastapi.encoders import jsonable_encoder
from bson import ObjectId


router = APIRouter()


@router.post("/active-learning-queue/add-policy")
def add_policy(request: Request, policy_data: ActiveLearningPolicy):
    # Find the maximum active_learning_policy_id in the collection
    last_entry = request.app.active_learning_policies_collection.find_one({}, sort=[("active_learning_policy_id", -1)])
    new_policy_id = last_entry["active_learning_policy_id"] + 1 if last_entry else 0

    # Ensure that the policy does not already exist
    if request.app.active_learning_policies_collection.find_one({"active_learning_policy": policy_data.active_learning_policy}):
        raise HTTPException(status_code=400, detail="Policy already exists")

    # Add the new policy
    policy_data.active_learning_policy_id = new_policy_id
    policy_data.creation_time = datetime.now(timezone.utc).isoformat()
    request.app.active_learning_policies_collection.insert_one(policy_data.to_dict())

    return {"status": "success", "message": "New active learning policy added.", "active_learning_policy_id": new_policy_id}


@router.put("/active-learning-queue/update-policy")
def update_policy(request: Request, policy_id: int, policy_update: ActiveLearningPolicy):
    # Ensure that the policy exists
    existing_policy = request.app.active_learning_policies_collection.find_one({"active_learning_policy_id": policy_id})
    if not existing_policy:
        raise HTTPException(status_code=404, detail="Policy not found")

    # Update the policy
    update_data = policy_update.dict(exclude_unset=True)
    request.app.active_learning_policies_collection.update_one({"active_learning_policy_id": policy_id}, {"$set": update_data})

    return {"status": "success", "message": "Active learning policy updated.", "active_learning_policy_id": policy_id}



@router.get("/active-learning-queue/list-policies", response_class=PrettyJSONResponse)
def list_active_learning_policies(request: Request) -> List[ActiveLearningPolicy]:
    # Retrieve all active learning policies from the collection
    policies_cursor = request.app.active_learning_policies_collection.find({})

    # Convert the cursor to a list of ActiveLearningPolicy objects
    policies = [ActiveLearningPolicy(**policy) for policy in policies_cursor]

    # Check if the policies list is empty
    if len(policies) == 0:
        return []

    return policies


@router.delete("/active-learning-queue/remove-policies")
def delete_active_learning_policy(request: Request, active_learning_policy_id: int = None):
    if active_learning_policy_id is not None:
        # Check if any queue pairs are using this policy
        if request.app.active_learning_queue_pairs_collection.find_one({"active_learning_policy_id": active_learning_policy_id}):
            raise HTTPException(status_code=400, detail="Cannot delete policy: It is being used by one or more queue pairs")

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
        # If no ID is provided, check if there are any policies used by queue pairs
        if request.app.active_learning_queue_pairs_collection.find_one({"active_learning_policy_id": {"$exists": True}}):
            raise HTTPException(status_code=400, detail="Cannot delete all policies: One or more are being used by queue pairs")

        # Delete all policies
        request.app.active_learning_policies_collection.delete_many({})
        return {"status": "success", "message": "All policies deleted successfully."}

# new apis with new names and reponses

@router.post("/active-learning-queue/policy/add", description="adding policy for active learning", response_class=PrettyJSONResponse)
def add_policy(request: Request, policy_data: ActiveLearningPolicy):
    response_handler = ApiResponseHandler(request)

    try:
        # Find the maximum active_learning_policy_id in the collection
        last_entry = request.app.active_learning_policies_collection.find_one({}, sort=[("active_learning_policy_id", -1)])
        new_policy_id = last_entry["active_learning_policy_id"] + 1 if last_entry else 0

        # Ensure that the policy does not already exist
        if request.app.active_learning_policies_collection.find_one({"active_learning_policy": policy_data.active_learning_policy}):
            return response_handler.create_error_response(ErrorCode.INVALID_PARAMS, "Policy already exists", 400)

        # Add the new policy
        policy_data.active_learning_policy_id = new_policy_id
        policy_data.creation_time = datetime.now(timezone.utc).isoformat()
        inserted_policy_dict = policy_data.dict()
        inserted_result = request.app.active_learning_policies_collection.insert_one(inserted_policy_dict)

        # Retrieve the inserted policy, converting ObjectId to string
        inserted_policy = request.app.active_learning_policies_collection.find_one({"_id": inserted_result.inserted_id})
        if inserted_policy:
            # Convert _id from ObjectId to string and handle non-serializable data
            inserted_policy['_id'] = str(inserted_policy['_id'])
            inserted_policy = jsonable_encoder(inserted_policy)
            return response_handler.create_success_response(inserted_policy)

        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Failed to retrieve the inserted policy", 500)

    except Exception as e:
        # Handle unexpected errors
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal server error", 500)



@router.put("/active-learning-queue/policy/update", description="updating existed policy", response_class=PrettyJSONResponse)
def update_policy(request: Request, policy_id: int, policy_update: ActiveLearningPolicy):
    response_handler = ApiResponseHandler(request)
    try:
        existing_policy = request.app.active_learning_policies_collection.find_one({"active_learning_policy_id": policy_id})
        if not existing_policy:
            return response_handler.create_error_response(ErrorCode.ELEMENT_NOT_FOUND, "Policy not found", 404)

        update_data = policy_update.dict(exclude_unset=True)
        request.app.active_learning_policies_collection.update_one({"active_learning_policy_id": policy_id}, {"$set": update_data})

        return response_handler.create_success_response({"status": "success", "message": "Active learning policy updated.", "active_learning_policy_id": policy_id})
    except Exception as e:
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal server error", 500)



@router.get("/active-learning-queue/policy/list", description="returning a list of policies", response_class=PrettyJSONResponse)
def list_active_learning_policies(request: Request):
    response_handler = ApiResponseHandler(request)
    try:
        policies_cursor = request.app.active_learning_policies_collection.find({})
        policies = [ActiveLearningPolicy(**policy) for policy in policies_cursor]

        if len(policies) == 0:
            return response_handler.create_success_response({"policies": []})

        return response_handler.create_success_response({"policies": policies})
    except Exception as e:
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal server error", 500)


@router.delete("/active-learning-queue/policy/remove", description="Removing an existing policy", response_class=PrettyJSONResponse)
def delete_active_learning_policy(request: Request, active_learning_policy_id: int = None):
    response_handler = ApiResponseHandler(request)
    try:
        was_present = False

        if active_learning_policy_id is not None:
            if request.app.active_learning_queue_pairs_collection.find_one({"active_learning_policy_id": active_learning_policy_id}):
                return response_handler.create_error_response(ErrorCode.INVALID_PARAMS, "Cannot delete policy: It is being used by one or more queue pairs", 400)

            policy = request.app.active_learning_policies_collection.find_one({"active_learning_policy_id": active_learning_policy_id})
            if policy:
                was_present = True
                request.app.active_learning_policies_collection.delete_one({"active_learning_policy_id": active_learning_policy_id})
        else:
            if request.app.active_learning_queue_pairs_collection.find_one({"active_learning_policy_id": {"$exists": True}}):
                return response_handler.create_error_response(ErrorCode.INVALID_PARAMS, "Cannot delete all policies: One or more are being used by queue pairs", 400)

            if request.app.active_learning_policies_collection.count_documents({}) > 0:
                was_present = True
                request.app.active_learning_policies_collection.delete_many({})

        return response_handler.create_success_response({"wasPresent": was_present})

    except Exception as e:
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal server error", 500)



# Standardized APIs
    
@router.post("/active-learning-queue/policy", 
             status_code=200,
             response_model=StandardSuccessResponse[ActiveLearningPolicy],
             tags=["active learning policy"],
             description="Add a new active learning policy",
             responses=ApiResponseHandler.listErrors([400,422]))
def add_policy(request: Request, policy_data: RequestActiveLearningPolicy):
    api_response_handler = ApiResponseHandler(request)

    last_entry = request.app.active_learning_policies_collection.find_one({}, sort=[("active_learning_policy_id", -1)])
    new_policy_id = last_entry["active_learning_policy_id"] + 1 if last_entry else 0

    if request.app.active_learning_policies_collection.find_one({"active_learning_policy": policy_data.active_learning_policy}):
        return api_response_handler.create_error_response(
            ErrorCode.INVALID_PARAMS,
            "Policy already exists",
            400
        )

    creation_time = datetime.now(timezone.utc)
    policy_to_insert = policy_data.dict()
    policy_to_insert["active_learning_policy_id"] = new_policy_id
    policy_to_insert["creation_time"] = creation_time

    request.app.active_learning_policies_collection.insert_one(policy_to_insert)


    response_data = {
        "active_learning_policy_id": new_policy_id,
        "active_learning_policy": policy_data.active_learning_policy,
        "active_learning_policy_description": policy_data.active_learning_policy_description,
        "creation_time": creation_time.isoformat(),
    }

    return api_response_handler.create_success_response(response_data, 200)


@router.put("/active-learning-queue/{policy_id}",
            status_code=200,
            tags=["active learning policy"],
            description="Update an existing active learning policy",
            response_model=StandardSuccessResponse[ActiveLearningPolicy], 
            responses=ApiResponseHandler.listErrors([400, 404, 422]))
def update_policy(request: Request, policy_id: int, policy_update: RequestActiveLearningPolicy):
    api_response_handler = ApiResponseHandler(request)
    
    # Fetch the existing policy to ensure it exists and get its creation time
    existing_policy = request.app.active_learning_policies_collection.find_one({"active_learning_policy_id": policy_id})
    if not existing_policy:
        return api_response_handler.create_error_response(
            ErrorCode.ELEMENT_NOT_FOUND,
            "Policy not found",
            404
        )

    # Prepare the update data without altering the policy ID
    update_data = policy_update.dict(exclude_unset=True)
    
    # Perform the update operation, excluding the ID from changes
    update_result = request.app.active_learning_policies_collection.update_one(
        {"active_learning_policy_id": policy_id},
        {"$set": update_data}
    )

    if update_result.modified_count == 0:
        # Handle the case where no changes were made, potentially because the policy was not found
        return api_response_handler.create_error_response(
            ErrorCode.OTHER_ERROR,
            "No changes applied or policy not found",
            400
        )

    # Construct the response data, including the unchanged policy ID and creation time
    response_data = {
        "active_learning_policy_id": policy_id,
        "active_learning_policy": update_data.get("active_learning_policy", existing_policy["active_learning_policy"]),
        "active_learning_policy_description": update_data.get("active_learning_policy_description", existing_policy["active_learning_policy_description"]),
        "creation_time": existing_policy["creation_time"].isoformat()  # Assuming the creation_time is stored in the existing_policy
    }

    # Return a success response with the updated policy details
    return api_response_handler.create_success_response(
        response_data,
        200
    )


@router.get("/active-learning-queue/policies",
            response_model=StandardSuccessResponse[List[ActiveLearningPolicy]],  
            tags=["active learning policy"],
            description="List active learning policies",
            responses=ApiResponseHandler.listErrors([422, 500]))
def list_active_learning_policies(request: Request):
    api_response_handler = ApiResponseHandler(request)
    
    try:
        policies_cursor = request.app.active_learning_policies_collection.find({})
        policies_list = list(policies_cursor)

        # Convert ObjectId and datetime to strings directly in the list comprehension
        policies_list = [{k: v.isoformat() if isinstance(v, datetime) else str(v) if isinstance(v, ObjectId) else v for k, v in policy.items()} for policy in policies_list]

        if not policies_list:
            return api_response_handler.create_success_response([], 200)

        return api_response_handler.create_success_response(policies_list, 200)
    except Exception as e:
        return api_response_handler.create_error_response(ErrorCode.OTHER_ERROR, str(e), 500)
    

@router.delete("/active-learning-queue/policy/remove/{policy_id}", 
               response_model=StandardSuccessResponse[WasPresentResponse],
               status_code=200,
               tags=["active learning policy"],
               description="Removing an existing policy by ID",
               responses=ApiResponseHandler.listErrors([422, 500]))
def delete_active_learning_policy(request: Request, policy_id: int ):
    response_handler = ApiResponseHandler(request)
    
    try:
        was_present = False

        # Check if the policy is being used by any queue pairs
        if request.app.active_learning_queue_pairs_collection.find_one({"active_learning_policy_id": policy_id}):
            return response_handler.create_error_response(ErrorCode.INVALID_PARAMS, "Cannot delete policy: It is being used by one or more queue pairs", 400)

        # Check if the policy exists and delete it
        policy = request.app.active_learning_policies_collection.find_one({"active_learning_policy_id": policy_id})
        if policy:
            was_present = True
            request.app.active_learning_policies_collection.delete_one({"active_learning_policy_id": policy_id})

        # Return a success response indicating if the policy was deleted
        return response_handler.create_success_response({"wasPresent": was_present}, 200)
    except Exception as e:
        # Handle any unexpected errors
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal server error", 500)