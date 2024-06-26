from fastapi import Request, APIRouter, HTTPException, Query, Body, File, UploadFile
from utility.path import separate_bucket_and_file_path
from utility.minio import cmd
import uuid
from datetime import datetime, timedelta
from orchestration.api.mongo_schemas import Task, ListTask
from orchestration.api.api_inpainting_dataset import get_sequential_id_inpainting
import pymongo
from .api_utils import PrettyJSONResponse
from typing import List
import json
import paramiko
from typing import Optional
import csv
from .api_utils import ApiResponseHandlerV1, ErrorCode, StandardSuccessResponseV1, AddJob, WasPresentResponse,CountResponse
from pymongo import UpdateMany
from fastapi.encoders import jsonable_encoder

router = APIRouter()


# -------------------- Get -------------------------

@router.get("/queue/inpainting-generation/get-job", tags=["deprecated3"], description="changed with /queue/inpainting-generation/set-pending-job-as-in-progress ")
def get_job(request: Request, task_type= None, model_type=""):
    query = {}

    if task_type:
        query["task_type"] = task_type

    if model_type:    
        query["task_type"] = {"$regex": model_type}

    # Query to find the n newest elements based on the task_completion_time
    job = request.app.pending_inpainting_jobs_collection.find_one(query, sort=[("task_creation_time", pymongo.ASCENDING)])

    if job is None:
        raise HTTPException(status_code=404)

    # delete from pending
    request.app.pending_inpainting_jobs_collection.delete_one({"uuid": job["uuid"]})
    # add to in progress
    request.app.in_progress_inpainting_jobs_collection.insert_one(job)

    # remove the auto generated field
    job.pop('_id', None)

    return job

@router.get("/queue/inpainting-generation/set-pending-job-as-in-progress", 
            tags=["inpainting jobs"],
            description="Update in pending inpainting job and mark as in progress.",
            response_model=StandardSuccessResponseV1[Task],
            responses=ApiResponseHandlerV1.listErrors([400, 500]))
def get_job(request: Request, task_type= None, model_type=""):
    api_response_handler = ApiResponseHandlerV1(request)

    try:
    
        query = {}

        if task_type:
            query["task_type"] = task_type

        if model_type:    
            query["task_type"] = {"$regex": model_type}

        # Query to find the n newest elements based on the task_completion_time
        job = request.app.pending_inpainting_jobs_collection.find_one(query, sort=[("task_creation_time", pymongo.ASCENDING)])

        if job is None:
            return api_response_handler.create_error_response_v1(
                    error_code=ErrorCode.ELEMENT_NOT_FOUND,
                    error_string="Job not found in in-progress collection",
                    http_status_code=400
                )

        # delete from pending
        request.app.pending_inpainting_jobs_collection.delete_one({"uuid": job["uuid"]})
        # add to in progress
        request.app.in_progress_inpainting_jobs_collection.insert_one(job)

        # remove the auto generated field
        job.pop('_id', None)

        return api_response_handler.create_success_response_v1(
                response_data=job,
                http_status_code=200
            )
    
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )

 # --------------------- Add ---------------------------

@router.post("/queue/inpainting-generation/add-job-with-upload", 
             description="Add a job to db",
             status_code=200,
             tags=["inpainting jobs"],
             response_model=StandardSuccessResponseV1[AddJob],
             responses=ApiResponseHandlerV1.listErrors([422, 500]))
async def add_job_with_upload(request: Request, task: Task = Body(...), mask_image: UploadFile = File(...), input_image: UploadFile = File(...)):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        if task.uuid in ["", None]:
            # Generate UUID since it's empty
            task.uuid = str(uuid.uuid4())

        # Add task creation time
        task.task_creation_time = datetime.now()

        # Check if file_path is blank and dataset is provided
        if (task.task_input_dict is None or "file_path" not in task.task_input_dict or task.task_input_dict["file_path"] in ['', "[auto]", "[default]"]) and "dataset" in task.task_input_dict:
            
            dataset_name = task.task_input_dict["dataset"]
            sequential_id_arr = get_sequential_id_inpainting(request, dataset=dataset_name)
            
            new_file_path = "{}.jpg".format(sequential_id_arr[0])
            
            task.task_input_dict["file_path"] = new_file_path

        init_mask = "{0}/{1}_mask.jpg".format(dataset_name, sequential_id_arr[0])
        init_img = "{0}/{1}_input_image.jpg".format(dataset_name, sequential_id_arr[0])
        
        task.task_input_dict["init_img"] = init_mask
        task.task_input_dict["init_mask"] = init_img

        cmd.upload_data(request.app.minio_client, "datasets-inpainting", init_mask, mask_image.file)
        cmd.upload_data(request.app.minio_client, "datasets-inpainting", init_img, input_image.file)

        # Insert task into pending_jobs_collection
        request.app.pending_inpainting_jobs_collection.insert_one(task.to_dict())


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
    
@router.post("/queue/inpainting-generation/add-job", 
             description="Add a job to db",
             status_code=200,
             tags=["inpainting jobs"],
             response_model=StandardSuccessResponseV1[AddJob],
             responses=ApiResponseHandlerV1.listErrors([422, 500]))
async def add_job(request: Request, task: Task):

    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        if task.uuid in ["", None]:
            # Generate UUID since it's empty
            task.uuid = str(uuid.uuid4())

        # Add task creation time
        task.task_creation_time = datetime.now()

        # Check if file_path is blank and dataset is provided
        if (task.task_input_dict is None or "file_path" not in task.task_input_dict or task.task_input_dict["file_path"] in ['', "[auto]", "[default]"]) and "dataset" in task.task_input_dict:
            dataset_name = task.task_input_dict["dataset"]
            sequential_id_arr = get_sequential_id_inpainting(request, dataset=dataset_name)
            new_file_path = "{}.jpg".format(sequential_id_arr[0])
            task.task_input_dict["file_path"] = new_file_path

        # Insert task into pending_jobs_collection
        request.app.pending_inpainting_jobs_collection.insert_one(task.to_dict())


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
    
@router.get("/queue/inpainting-generation/pending-count", tags=["deprecated3"], description="changed with /queue/inpainting-generation/get-pending-jobs-count ")
def get_pending_job_count(request: Request):
    count = request.app.pending_inpainting_jobs_collection.count_documents({})
    return count


@router.get("/queue/inpainting-generation/in-progress-count", tags=["deprecated3"], description="changed with /queue/inpainting-generation/get-in-progress-jobs-count")
def get_in_progress_job_count(request: Request):
    count = request.app.in_progress_inpainting_jobs_collection.count_documents({})
    return count


@router.get("/queue/inpainting-generation/completed-count", tags=["deprecated3"],description="changed with /queue/inpainting-generation/get-completed-jobs-count")
def get_completed_job_count(request: Request):
    count = request.app.completed_inpainting_jobs_collection.count_documents({})
    return count    


 # --------------------- List ----------------------

@router.get("/queue/inpainting-generation/list-pending", response_class=PrettyJSONResponse, tags=["deprecated3"], description="changed with /queue/inpainting-generation/list-pending-jobs ")
def get_list_pending_jobs(request: Request):
    jobs = list(request.app.pending_inpainting_jobs_collection.find({}))

    for job in jobs:
        job.pop('_id', None)

    return jobs


@router.get("/queue/inpainting-generation/list-in-progress", response_class=PrettyJSONResponse, tags=["deprecated3"], description="changed with /queue/inpainting-generation/list-in-progress-jobs")
def get_list_in_progress_jobs(request: Request):
    jobs = list(request.app.in_progress_inpainting_jobs_collection.find({}))

    for job in jobs:
        job.pop('_id', None)

    return jobs


@router.get("/queue/inpainting-generation/list-completed", response_class=PrettyJSONResponse, tags=["deprecated3"], description="changed with /queue/inpainting-generation/list-completed-jobs")
def get_list_completed_jobs(request: Request, limit: Optional[int] = Query(10, alias="limit")):
    # Use the limit parameter in the find query to limit the results
    jobs = list(request.app.completed_inpainting_jobs_collection.find({}).limit(limit))

    for job in jobs:
        job.pop('_id', None)

    return jobs

@router.get("/queue/inpainting-generation/list-completed-by-dataset", response_class=PrettyJSONResponse, tags=["deprecated3"], description="changed with /queue/inpainting-generation/list-completed-jobs")
def get_list_completed_jobs_by_dataset(request: Request, dataset, limit: Optional[int] = Query(10, alias="limit")):
    # Use the limit parameter in the find query to limit the results
    jobs = list(request.app.completed_inpainting_jobs_collection.find({"task_input_dict.dataset": dataset}).limit(limit))

    for job in jobs:
        job.pop('_id', None)

    return jobs    



# ---------------- Update -------------------


@router.put("/queue/inpainting-generation/update-completed", description="changed with /queue/inpainting-generation/set-in-progress-job-as-completed", tags=['deprecated3'])
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




@router.get("/queue/inpainting-generation/move-job-to-in-progress", 
            description="gets the oldest pending job and moves it to the 'in-progress' queue",
            tags=["inpainting jobs"], 
            response_model=StandardSuccessResponseV1[Task], 
            responses=ApiResponseHandlerV1.listErrors([422, 500]))
async def get_job(request: Request, task_type: Optional[str] = None, model_type: Optional[str] = ""):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        query = {}
        if task_type:
            query["task_type"] = task_type
        if model_type:
            query["task_type"] = {"$regex": model_type}

        # Query to find the newest element based on the task_creation_time
        job = request.app.pending_inpainting_jobs_collection.find_one(query, sort=[("task_creation_time", pymongo.ASCENDING)])
        
        if job is None:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="No job found",
                http_status_code=404
            )

        # Delete from pending
        request.app.pending_inpainting_jobs_collection.delete_one({"uuid": job["uuid"]})
        # Add to in progress
        request.app.in_progress_inpainting_jobs_collection.insert_one(job)
        # Remove the auto-generated field
        job.pop('_id', None)

        return api_response_handler.create_success_response_v1(
            response_data=job,
            http_status_code=200
        )
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=f"Internal server error: {str(e)}",
            http_status_code=500
        )



@router.get("/queue/inpainting-generation/get-pending-jobs-count", tags=["inpainting jobs"], response_model=StandardSuccessResponseV1[CountResponse], responses=ApiResponseHandlerV1.listErrors([500]))
async def get_pending_job_count(request: Request):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        count = request.app.pending_inpainting_jobs_collection.count_documents({})
        return api_response_handler.create_success_response_v1(
            response_data={"count": count},
            http_status_code=200
        )
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=f"Internal server error: {str(e)}",
            http_status_code=500
        )

@router.get("/queue/inpainting-generation/get-in-progress-jobs-count", tags=["inpainting jobs"], response_model=StandardSuccessResponseV1[CountResponse], responses=ApiResponseHandlerV1.listErrors([500]))
async def get_in_progress_job_count(request: Request):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        count = request.app.in_progress_inpainting_jobs_collection.count_documents({})
        return api_response_handler.create_success_response_v1(
            response_data={"count": count},
            http_status_code=200
        )
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=f"Internal server error: {str(e)}",
            http_status_code=500
        )

@router.get("/queue/inpainting-generation/get-completed-jobs-count", tags=["inpainting jobs"], response_model=StandardSuccessResponseV1[CountResponse], responses=ApiResponseHandlerV1.listErrors([500]))
async def get_completed_job_count(request: Request):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        count = request.app.completed_inpainting_jobs_collection.count_documents({})
        return api_response_handler.create_success_response_v1(
            response_data={"count": count},
            http_status_code=200
        )
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=f"Internal server error: {str(e)}",
            http_status_code=500
        )



@router.get("/queue/inpainting-generation/list-pending-jobs", 
            description="List all pending inpainting jobs", 
            tags=["inpainting jobs"],
            response_model=StandardSuccessResponseV1[ListTask],
            responses=ApiResponseHandlerV1.listErrors([500]))
def get_list_pending_jobs(request: Request):
    api_response_handler = ApiResponseHandlerV1(request)
    
    jobs = list(request.app.pending_inpainting_jobs_collection.find({}))
    for job in jobs:
        job.pop('_id', None)
    
    return api_response_handler.create_success_response_v1(
        response_data={"jobs": jobs},
        http_status_code=200
    )


@router.get("/queue/inpainting-generation/list-in-progress-jobs", 
            description="List all in-progress inpainting jobs", 
            tags=["inpainting jobs"],
            response_model=StandardSuccessResponseV1[ListTask],
            responses=ApiResponseHandlerV1.listErrors([500]))
def get_list_in_progress_jobs(request: Request):
    api_response_handler = ApiResponseHandlerV1(request)
    
    jobs = list(request.app.in_progress_inpainting_jobs_collection.find({}))
    for job in jobs:
        job.pop('_id', None)
    
    return api_response_handler.create_success_response_v1(
        response_data={"jobs": jobs},
        http_status_code=200
    )


@router.get("/queue/inpainting-generation/list-completed-jobs", 
            description="List completed inpainting jobs with an optional dataset filter and a limit", 
            tags=["inpainting jobs"],
            response_model=StandardSuccessResponseV1[ListTask],
            responses=ApiResponseHandlerV1.listErrors([500]))
def get_list_completed_jobs(request: Request, 
                            limit: Optional[int] = Query(10, alias="limit"), 
                            dataset: Optional[str] = Query(None, alias="dataset")):
    api_response_handler = ApiResponseHandlerV1(request)
    
    query = {}
    if dataset:
        query["task_input_dict.dataset"] = dataset

    jobs = list(request.app.completed_inpainting_jobs_collection.find(query).limit(limit))
    for job in jobs:
        job.pop('_id', None)
    
    return api_response_handler.create_success_response_v1(
        response_data={"jobs": jobs},
        http_status_code=200
    )


@router.put("/queue/inpainting-generation/set-in-progress-job-as-completed", 
            description="Update in progress inpainting job and mark as completed.", 
            tags=['inpainting jobs'],
            response_model=StandardSuccessResponseV1[Task],
            responses=ApiResponseHandlerV1.listErrors([400, 500]))
def update_job_completed(request: Request, task: Task):
    api_response_handler = ApiResponseHandlerV1(request)

    try:
        # Check if the job exists in the in-progress collection
        job = request.app.in_progress_inpainting_jobs_collection.find_one({"uuid": task.uuid})
        if job is None:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="Job not found in in-progress collection",
                http_status_code=400
            )

        # Add to completed collection
        request.app.completed_inpainting_jobs_collection.insert_one(task.to_dict())

        # Remove from in-progress collection
        request.app.in_progress_inpainting_jobs_collection.delete_one({"uuid": task.uuid})

        return api_response_handler.create_success_response_v1(
            response_data=task,
            http_status_code=200
        )

    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )
