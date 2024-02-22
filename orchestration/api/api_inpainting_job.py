from fastapi import Request, APIRouter, HTTPException, Query, Body
from utility.path import separate_bucket_and_file_path
from utility.minio import cmd
import uuid
from datetime import datetime, timedelta
from orchestration.api.mongo_schemas import Task
from orchestration.api.api_dataset import get_sequential_id
import pymongo
from .api_utils import PrettyJSONResponse
from typing import List
import json
import paramiko
from typing import Optional
import csv
from .api_utils import ApiResponseHandlerV1, ErrorCode, StandardSuccessResponseV1, AddJob, WasPresentResponse
from pymongo import UpdateMany
from fastapi.encoders import jsonable_encoder

router = APIRouter()


# -------------------- Get -------------------------

@router.get("/queue/inpainting-generation/get-job", tags=["inpainting jobs"])
def get_job(request: Request, task_type= None, model_type=""):
    query = {}

    if task_type:
        query["task_type"] = task_type

    if model_type:    
        query["task_type"] = {"$regex": model_type}

    # Query to find the n newest elements based on the task_completion_time
    job = request.app.pending_inpainting_jobs_collection.find_one(query, sort=[("task_creation_time", pymongo.ASCENDING)])

    if job is None:
        raise HTTPException(status_code=204)

    # delete from pending
    request.app.pending_inpainting_jobs_collection.delete_one({"uuid": job["uuid"]})
    # add to in progress
    request.app.in_progress_inpainting_jobs_collection.insert_one(job)

    # remove the auto generated field
    job.pop('_id', None)

    return job

 # --------------------- Add ---------------------------

@router.post("/queue/inpainting-generation/add-job", 
             description="Add a job to db",
             status_code=200,
             tags=["inpainting jobs"],
             response_model=StandardSuccessResponseV1[AddJob],
             responses=ApiResponseHandlerV1.listErrors([500]))
def add_job(request: Request, task: Task):
    task_dict = jsonable_encoder(task)
    api_response_handler = ApiResponseHandlerV1(request, body_data=task_dict)
    try:
        if task.uuid in ["", None]:
            # Generate UUID since it's empty
            task.uuid = str(uuid.uuid4())

        # Add task creation time
        task.task_creation_time = datetime.now()

        # Check if file_path is blank and dataset is provided
        if (task.task_input_dict is None or "file_path" not in task.task_input_dict or task.task_input_dict["file_path"] in ['', "[auto]", "[default]"]) and "dataset" in task.task_input_dict:
            dataset_name = task.task_input_dict["dataset"]
            sequential_id_arr = get_sequential_id(request, dataset=dataset_name)
            new_file_path = "{}.jpg".format(sequential_id_arr[0])
            task.task_input_dict["file_path"] = new_file_path

        # Insert task into pending_jobs_collection
        request.app.pending_inpainting_jobs_collection.insert_one(task.dict())


        # Convert datetime to ISO 8601 formatted string for JSON serialization
        creation_time_iso = task.task_creation_time.isoformat() if task.task_creation_time else None
        # Use ApiResponseHandler for standardized success response
        return api_response_handler.create_success_response_v1(
            response_data={"uuid": task.uuid, "creation_time": creation_time_iso},
            http_status_code=200
        )

    except Exception as e:
        # Log the error and return a standardized error response
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )
    
# -------------- Get jobs count ----------------------
    
@router.get("/queue/inpainting-generation/pending-count", tags=["inpainting jobs"])
def get_pending_job_count(request: Request):
    count = request.app.pending_inpainting_jobs_collection.count_documents({})
    return count


@router.get("/queue/inpainting-generation/in-progress-count", tags=["inpainting jobs"])
def get_in_progress_job_count(request: Request):
    count = request.app.in_progress_inpainting_jobs_collection.count_documents({})
    return count


@router.get("/queue/inpainting-generation/completed-count", tags=["inpainting jobs"])
def get_completed_job_count(request: Request):
    count = request.app.completed_inpainting_jobs_collection.count_documents({})
    return count    



# ----------------- delete jobs ----------------------

@router.delete("/queue/inpainting-generation/delete-all-pending",
               description="remove all pending jobs",
               response_model=StandardSuccessResponseV1[WasPresentResponse],
               tags=["inpainting jobs"],
               responses=ApiResponseHandlerV1.listErrors([500]))
def clear_all_pending_jobs(request: Request):
    api_response_handler = ApiResponseHandlerV1(request)
    try:
        was_present = request.app.pending_inpainting_jobs_collection.count_documents({}) > 0
        request.app.pending_inpainting_jobs_collection.delete_many({})

        return api_response_handler.create_success_response_v1(
            response_data={"wasPresent": was_present},
            http_status_code=200
        )
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )
    
@router.delete("/queue/inpainting-generation/delete-all-in-progress",
               description="remove all in-progress jobs",
               response_model=StandardSuccessResponseV1[WasPresentResponse],
               tags=["inpainting jobs"],
               responses=ApiResponseHandlerV1.listErrors([500]))
def clear_all_in_progress_jobs(request: Request):
    api_response_handler = ApiResponseHandlerV1(request)
    try:
        was_present = request.app.in_progress_inapinting_jobs_collection.count_documents({}) > 0
        request.app.in_progress_inpainting_jobs_collection.delete_many({})

        return api_response_handler.create_success_response_v1(
            response_data={"wasPresent": was_present},
            http_status_code=200
        )
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )    
    
    
@router.delete("/queue/inpainting-generation/delete-all-completed",
               description="remove all completed jobs",
               response_model=StandardSuccessResponseV1[WasPresentResponse],
               tags=["inpainting jobs"],
               responses=ApiResponseHandlerV1.listErrors([500]))
def clear_all_in_progress_jobs(request: Request):
    api_response_handler = ApiResponseHandlerV1(request)
    try:
        was_present = request.app.completed_inpainting_jobs_collection.count_documents({}) > 0
        request.app.completed_inpainting_jobs_collection.delete_many({})

        return api_response_handler.create_success_response_v1(
            response_data={"wasPresent": was_present},
            http_status_code=200
        )
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )    
    

 # --------------------- List ----------------------

@router.get("/queue/inpainting-generation/list-pending", response_class=PrettyJSONResponse, tags=["inpainting jobs"])
def get_list_pending_jobs(request: Request):
    jobs = list(request.app.pending_inpainting_jobs_collection.find({}))

    for job in jobs:
        job.pop('_id', None)

    return jobs


@router.get("/queue/inpainting-generation/list-in-progress", response_class=PrettyJSONResponse, tags=["inpainting jobs"])
def get_list_in_progress_jobs(request: Request):
    jobs = list(request.app.in_progress_inpainting_jobs_collection.find({}))

    for job in jobs:
        job.pop('_id', None)

    return jobs


@router.get("/queue/inpainting-generation/list-completed", response_class=PrettyJSONResponse, tags=["inpainting jobs"])
def get_list_completed_jobs(request: Request, limit: Optional[int] = Query(10, alias="limit")):
    # Use the limit parameter in the find query to limit the results
    jobs = list(request.app.completed_inpainting_jobs_collection.find({}).limit(limit))

    for job in jobs:
        job.pop('_id', None)

    return jobs

@router.get("/queue/inpainting-generation/list-completed-by-dataset", response_class=PrettyJSONResponse, tags=["inpainting jobs"])
def get_list_completed_jobs_by_dataset(request: Request, dataset, limit: Optional[int] = Query(10, alias="limit")):
    # Use the limit parameter in the find query to limit the results
    jobs = list(request.app.completed_inpainting_jobs_collection.find({"task_input_dict.dataset": dataset}).limit(limit))

    for job in jobs:
        job.pop('_id', None)

    return jobs    



# ---------------- Update -------------------


@router.put("/queue/inpainting-generation/update-completed", description="Update in progress inpainting job and mark as completed.", tags=['inpainting jobs'])
def update_job_completed(request: Request, task: Task):
    # check if exist
    job = request.app.in_progress_inpainting_jobs_collection.find_one({"uuid": task.uuid})
    if job is None:
        return False
    
    # add to completed
    request.app.completed_inpainting_jobs_collection.insert_one(task.to_dict())

    # remove from in progress
    request.app.in_progress_inpainting_jobs_collection.delete_one({"uuid": task.uuid})

    return True



