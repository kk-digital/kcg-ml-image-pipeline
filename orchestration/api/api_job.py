from fastapi import Request, APIRouter, HTTPException
from utility.path import separate_bucket_and_file_path
from utility.minio import cmd
import uuid
from datetime import datetime, timedelta
from orchestration.api.mongo_schemas import Task
from orchestration.api.api_dataset import get_sequential_id
import pymongo
from .api_utils import PrettyJSONResponse

router = APIRouter()


# -------------------- Get -------------------------


@router.get("/queue/image-generation/get-job")
def get_job(request: Request, task_type: str = None):
    query = {}
    if task_type != None:
        query = {"task_type": task_type}

    # Query to find the n newest elements based on the task_completion_time

    job = request.app.pending_jobs_collection.find_one(query, sort=[("task_creation_time", pymongo.ASCENDING)])

    if job is None:
        raise HTTPException(status_code=204)

    # delete from pending
    request.app.pending_jobs_collection.delete_one({"uuid": job["uuid"]})
    # add to in progress
    request.app.in_progress_jobs_collection.insert_one(job)

    # remove the auto generated field
    job.pop('_id', None)

    return job

 # --------------------- Add ---------------------------
@router.post("/queue/image-generation/add", description="Add a job to db")
def add_job(request: Request, task: Task):
    if task.uuid in ["", None]:
        # generate since its empty
        task.uuid = str(uuid.uuid4())

    # add task creation time
    task.task_creation_time = datetime.now()

    # check if file_path is blank
    if (task.task_input_dict is None or "file_path" not in task.task_input_dict or task.task_input_dict["file_path"] in [
        '', "[auto]", "[default]"]) and "dataset" in task.task_input_dict:
        dataset_name = task.task_input_dict["dataset"]
        # get file path
        sequential_id_arr = get_sequential_id(request, dataset=dataset_name)
        new_file_path = "{}.jpg".format(sequential_id_arr[0])
        task.task_input_dict["file_path"] = new_file_path

    request.app.pending_jobs_collection.insert_one(task.to_dict())

    return {"uuid": task.uuid, "creation_time": task.task_creation_time}


@router.get("/queue/image-generation/get-jobs-count-last-hour")
def get_jobs_count_last_hour(request: Request, dataset):

    # Calculate the timestamp for one hour ago
    current_time = datetime.now()
    time_ago = current_time - timedelta(hours=1)

    # Query the collection to count the documents created in the last hour
    pending_query = {"task_input_dict.dataset": dataset, "task_creation_time": {"$gte": time_ago}}
    in_progress_query = {"task_input_dict.dataset": dataset, "task_creation_time": {"$gte": time_ago}}
    completed_query = {"task_input_dict.dataset": dataset, "task_creation_time": {"$gte": time_ago}}

    count = 0

    # Take into account pending & in progress & completed jobs
    pending_count = request.app.pending_jobs_collection.count_documents(pending_query)
    in_progress_count = request.app.in_progress_jobs_collection.count_documents(in_progress_query)
    completed_count = request.app.completed_jobs_collection.count_documents(completed_query)

    print("pending, ", pending_count)
    print("in_progress_count, ", in_progress_count)
    print("completed_count, ", completed_count)

    count += pending_count
    count += in_progress_count
    count += completed_count

    return count


@router.get("/queue/image-generation/get-jobs-count-last-n-hour")
def get_jobs_count_last_n_hour(request: Request, dataset, hours: int):

    # Calculate the timestamp for one hour ago
    current_time = datetime.now()
    time_ago = current_time - timedelta(hours=hours)

    # Query the collection to count the documents created in the last hour
    pending_query = {"task_input_dict.dataset": dataset, "task_creation_time": {"$gte": time_ago}}
    in_progress_query = {"task_input_dict.dataset": dataset, "task_creation_time": {"$gte": time_ago}}
    completed_query = {"task_input_dict.dataset": dataset, "task_creation_time": {"$gte": time_ago}}

    count = 0

    # Take into account pending & in progress & completed jobs
    pending_count = request.app.pending_jobs_collection.count_documents(pending_query)
    in_progress_count = request.app.in_progress_jobs_collection.count_documents(in_progress_query)
    completed_count = request.app.completed_jobs_collection.count_documents(completed_query)

    print("pending, ", pending_count)
    print("in_progress_count, ", in_progress_count)
    print("completed_count, ", completed_count)

    count += pending_count
    count += in_progress_count
    count += completed_count

    return count


# -------------- Get jobs count ----------------------
@router.get("/queue/image-generation/pending-count")
def get_pending_job_count(request: Request):
    count = request.app.pending_jobs_collection.count_documents({})
    return count


@router.get("/queue/image-generation/in-progress-count")
def get_in_progress_job_count(request: Request):
    count = request.app.in_progress_jobs_collection.count_documents({})
    return count


@router.get("/queue/image-generation/completed-count")
def get_completed_job_count(request: Request):
    count = request.app.completed_jobs_collection.count_documents({})
    return count


@router.get("/queue/image-generation/failed-count")
def get_failed_job_count(request: Request):
    count = request.app.failed_jobs_collection.count_documents({})
    return count


# ----------------- delete jobs ----------------------
@router.delete("/queue/image-generation/clear-all-pending")
def clear_all_pending_jobs(request: Request):
    request.app.pending_jobs_collection.delete_many({})

    return True


@router.delete("/queue/image-generation/clear-all-in-progress")
def clear_all_in_progress_jobs(request: Request):
    request.app.in_progress_jobs_collection.delete_many({})

    return True


@router.delete("/queue/image-generation/clear-all-failed")
def clear_all_failed_jobs(request: Request):
    request.app.failed_jobs_collection.delete_many({})

    return True

@router.delete("/queue/image-generation/clear-all-completed")
def clear_all_completed_jobs(request: Request):
    request.app.completed_jobs_collection.delete_many({})

    return True



 # --------------------- List ----------------------

@router.get("/queue/image-generation/list-pending", response_class=PrettyJSONResponse)
def get_list_pending_jobs(request: Request):
    jobs = list(request.app.pending_jobs_collection.find({}))

    for job in jobs:
        job.pop('_id', None)

    return jobs


@router.get("/queue/image-generation/list-in-progress", response_class=PrettyJSONResponse)
def get_list_in_progress_jobs(request: Request):
    jobs = list(request.app.in_progress_jobs_collection.find({}))

    for job in jobs:
        job.pop('_id', None)

    return jobs


@router.get("/queue/image-generation/list-completed", response_class=PrettyJSONResponse)
def get_list_completed_jobs(request: Request):
    jobs = list(request.app.completed_jobs_collection.find({}))

    for job in jobs:
        job.pop('_id', None)

    return jobs


@router.get("/queue/image-generation/list-failed", response_class=PrettyJSONResponse)
def get_list_failed_jobs(request: Request):
    jobs = list(request.app.failed_jobs_collection.find({}))

    for job in jobs:
        job.pop('_id', None)

    return jobs


@router.get("/queue/image-generation/count-completed")
def count_completed(request: Request, dataset: str = None):

    jobs = list(request.app.completed_jobs_collection.find({
        'task_input_dict.dataset': dataset
    }))

    return len(jobs)

@router.get("/queue/image-generation/count-pending")
def count_completed(request: Request, dataset: str = None):

    jobs = list(request.app.pending_jobs_collection.find({
        'task_input_dict.dataset': dataset
    }))

    return len(jobs)

@router.get("/queue/image-generation/count-in-progress")
def count_completed(request: Request, dataset: str = None):

    jobs = list(request.app.in_progress_jobs_collection.find({
        'task_input_dict.dataset': dataset
    }))

    return len(jobs)

# ---------------- Update -------------------


@router.put("/queue/image-generation/update-completed", description="Update in progress job and mark as completed.")
def update_job_completed(request: Request, task: Task):
    # check if exist
    job = request.app.in_progress_jobs_collection.find_one({"uuid": task.uuid})
    if job is None:
        raise HTTPException(status_code=404)

    # add to completed
    request.app.completed_jobs_collection.insert_one(task.to_dict())

    # remove from in progress
    request.app.in_progress_jobs_collection.delete_one({"uuid": task.uuid})

    return True


@router.put("/queue/image-generation/update-failed", description="Update in progress job and mark as failed.")
def update_job_failed(request: Request, task: Task):
    # check if exist
    job = request.app.in_progress_jobs_collection.find_one({"uuid": task.uuid})
    if job is None:
        raise HTTPException(status_code=404)

    # add to failed
    request.app.failed_jobs_collection.insert_one(task.to_dict())

    # remove from in progress
    request.app.in_progress_jobs_collection.delete_one({"uuid": task.uuid})

    return True

@router.delete("/queue/image-generation/cleanup-completed-and-orphaned")
def cleanup_completed_and_orphaned_jobs(request: Request):

    jobs = request.app.completed_jobs_collection.find({})
    for job in jobs:
        file_exists = True
        try:
            file_path = job['task_output_file_dict']['output_file_path']
            bucket_name, file_path = separate_bucket_and_file_path(file_path)
            file_exists = cmd.is_object_exists(request.app.minio_client, bucket_name, file_path)
        except Exception as e:
            file_exists = False

        if not file_exists:
            # remove from in progress
            request.app.completed_jobs_collection.delete_one({"uuid": job['uuid']})

    return True



# --------------- Job generation rate ---------------------

@router.get("/job/get-dataset-job-per-second")
def get_job_generation_rate(request: Request, dataset: str, sample_size : int):

    # 1. Take last N=50 image generations (only time stamp when it was submitted)
    # 2. Sort by Time Stamp
    # 3. Use TimeStamp of Oldest, divided by N=50;
    # to get Images/Second = ImageTaskGenerationRate (images/second estimate), over window of last N=50 images
    query = {
        'task_input_dict.dataset': dataset
    }
    # Query to find the n newest elements based on the task_completion_time
    jobs = list(request.app.completed_jobs_collection.find(query).sort("task_creation_time",
                                                                    pymongo.DESCENDING).limit(sample_size))

    total_jobs = len(jobs)
    if total_jobs == 0:
        return 1.0

    job_per_second = 0.0
    for job in jobs:
        task_start_time = job['task_start_time']
        task_completion_time = job['task_completion_time']

        task_start_time = datetime.strptime(task_start_time, '%Y-%m-%d %H:%M:%S')
        task_completion_time = datetime.strptime(task_completion_time, '%Y-%m-%d %H:%M:%S')

        difference_in_seconds = (task_completion_time - task_start_time).total_seconds()
        this_job_per_second = 1.0 / difference_in_seconds
        job_per_second += this_job_per_second / total_jobs

    return job_per_second
