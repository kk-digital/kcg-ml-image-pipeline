from fastapi import Request, HTTPException, APIRouter, Response, Query, status
from datetime import datetime, timedelta
import pymongo
from utility.minio import cmd
from orchestration.api.mongo_schema.active_learning_schemas import  ActiveLearningPolicy, ActiveLearningQueuePair
from .api_utils import ApiResponseHandlerV1, ErrorCode
import os
from fastapi.responses import JSONResponse
from pymongo.collection import Collection
from datetime import datetime, timezone
from typing import List
from io import BytesIO
from bson import ObjectId
from typing import Optional
import json

router = APIRouter()


@router.post("/ranking/add-image-pair")
def get_job_details(request: Request, job_uuid_1: str = Query(...), job_uuid_2: str = Query(...), active_learning_policy_id: int = Query(...), rank_model_id: int = Query(...), metadata: str = Query(None), generation_string: str = Query(None) ):
    api_response_handler = ApiResponseHandlerV1(request)

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


    rank = request.app.rank_model_models_collection.find_one(
        {"rank_model_id": rank_model_id}
    )

    if not rank:
        raise HTTPException(status_code=404, detail=f"rank with ID {rank} not found")  
    
    rank_model_string = rank.get("rank_model_string", "")


    policy = request.app.active_learning_policies_collection.find_one(
        {"active_learning_policy_id": active_learning_policy_id}
    )
    if not policy:
        raise HTTPException(status_code=404, detail=f"Active learning policy with ID {active_learning_policy_id} not found")  
    
    combined_job_details = {
        "rank_model_id": rank_model_id,
        "rank_model_string": rank_model_string,
        "active_learning_policy_id": active_learning_policy_id,
        "active_learning_policy": policy["active_learning_policy"],
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
    full_path = f"{combined_job_details['dataset_name']}/data/rank/{rank_model_string}/{json_file_name}"

    cmd.upload_data(request.app.minio_client, "datasets", full_path, data)

    mongo_combined_job_details = {"file_name": json_file_name, **combined_job_details}

    request.app.rank_active_learning_pairs_collection.insert_one(mongo_combined_job_details)

    return api_response_handler.create_success_response_v1(
            response_data="added",
            http_status_code=200
        )

@router.get("/ranking/list-rank-active-learning-datapoint",
            description="Get image rank scores by model id. Returns as descending order of scores",
            status_code=200,
            tags=["score"],  
            responses=ApiResponseHandlerV1.listErrors([400, 422]))
def get_image_rank_scores_by_model_id(request: Request):
    api_response_handler = ApiResponseHandlerV1(request)
    
    # check if exist
    items = list(request.app.rank_active_learning_pairs_collection.find({}))
    
    if not items:
        # If no items found, use ApiResponseHandler to return a standardized error response
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="No scores found for specified model_id.",
            http_status_code=400
        )
    
    score_data = []
    for item in items:
        # remove the auto generated '_id' field
        item.pop('_id', None)
        score_data.append(item)
    
    # Return a standardized success response with the score data
    return api_response_handler.create_success_response_v1(
        response_data=score_data,
        http_status_code=200
    )



