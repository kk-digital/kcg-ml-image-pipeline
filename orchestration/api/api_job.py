from fastapi import Request, APIRouter, HTTPException
import uuid
from datetime import datetime
from orchestration.api.mongo_schemas import Task
from orchestration.api.api_dataset import get_sequential_id

router = APIRouter()


# -------------------- Get -------------------------


@router.get("/job/get-job")
def get_job(request: Request, task_type: str = None):
    query = {}
    if task_type != None:
        query = {"task_type": task_type}

    # find
    job = request.app.pending_jobs_collection.find_one(query)
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
@router.post("/job/add", description="Add a job to db")
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




# -------------- Get jobs count ----------------------
@router.get("/job/pending-count")
def get_pending_job_count(request: Request):
    count = request.app.pending_jobs_collection.count_documents({})
    return count


@router.get("/job/in-progress-count")
def get_in_progress_job_count(request: Request):
    count = request.app.in_progress_jobs_collection.count_documents({})
    return count


@router.get("/job/completed-count")
def get_completed_job_count(request: Request):
    count = request.app.completed_jobs_collection.count_documents({})
    return count


@router.get("/job/failed-count")
def get_failed_job_count(request: Request):
    count = request.app.failed_jobs_collection.count_documents({})
    return count



# ----------------- delete jobs ----------------------
@router.delete("/job/clear-all-pending")
def clear_all_pending_jobs(request: Request):
    request.app.pending_jobs_collection.delete_many({})

    return True


@router.delete("/job/clear-all-in-progress")
def clear_all_in_progress_jobs(request: Request):
    request.app.in_progress_jobs_collection.delete_many({})

    return True


@router.delete("/job/clear-all-failed")
def clear_all_failed_jobs(request: Request):
    request.app.failed_jobs_collection.delete_many({})

    return True

@router.delete("/job/clear-all-completed")
def clear_all_completed_jobs(request: Request):
    request.app.completed_jobs_collection.delete_many({})

    return True



 # --------------------- List ----------------------

@router.get("/job/list-pending")
def get_list_pending_jobs(request: Request):
    jobs = list(request.app.pending_jobs_collection.find({}))

    for job in jobs:
        job.pop('_id', None)

    return jobs


@router.get("/job/list-in-progress")
def get_list_in_progress_jobs(request: Request):
    jobs = list(request.app.in_progress_jobs_collection.find({}))

    for job in jobs:
        job.pop('_id', None)

    return jobs


@router.get("/job/list-completed")
def get_list_completed_jobs(request: Request):
    jobs = list(request.app.completed_jobs_collection.find({}))

    for job in jobs:
        job.pop('_id', None)

    return jobs


@router.get("/job/list-failed")
def get_list_failed_jobs(request: Request):
    jobs = list(request.app.failed_jobs_collection.find({}))

    for job in jobs:
        job.pop('_id', None)

    return jobs


# ---------------- Update -------------------


@router.put("/job/update-completed", description="Update in progress job and mark as completed.")
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


@router.put("/job/update-failed", description="Update in progress job and mark as failed.")
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
