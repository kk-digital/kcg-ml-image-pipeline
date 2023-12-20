from fastapi import Request, HTTPException, APIRouter, Response, Query, status
from datetime import datetime, timedelta
import math
import random
import pymongo
from utility.minio import cmd
from orchestration.api.mongo_schemas import  ActiveLearningPolicy
from .api_utils import PrettyJSONResponse
import os
from datetime import datetime, timezone
from typing import List
from io import BytesIO
import json

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