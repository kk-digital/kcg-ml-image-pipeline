from fastapi import Request, APIRouter, HTTPException
import uuid
from datetime import datetime
from orchestration.api.mongo_schemas import TrainingTask
from orchestration.api.api_dataset import get_sequential_id

router = APIRouter()

# -------------------- Get -------------------------
@router.get("/training/get-job")
def get_job(request: Request, task_type: str = None):
    query = {}
    if task_type != None:
        query = {"task_type": task_type}

    # find
    job = request.app.training_pending_jobs_collection.find_one(query)
    if job is None:
        raise HTTPException(status_code=204)

    # delete from pending
    request.app.training_pending_jobs_collection.delete_one({"uuid": job["uuid"]})
    # add to in progress
    request.app.training_in_progress_jobs_collection.insert_one(job)

    # remove the auto generated field
    job.pop('_id', None)

    return job

 # --------------------- Add ---------------------------
@router.post("/training/add", description="Add a job to db")
def add_job(request: Request, training_task: TrainingTask):
    if training_task.uuid in ["", None]:
        # generate since its empty
        training_task.uuid = str(uuid.uuid4())

    # add task creation time
    training_task.task_creation_time = datetime.now()
    request.app.training_pending_jobs_collection.insert_one(training_task.to_dict())

    return {"uuid": training_task.uuid, "creation_time": training_task.task_creation_time}




# -------------- Get jobs count ----------------------
@router.get("/training/pending-count")
def get_pending_job_count(request: Request):
    count = request.app.training_pending_jobs_collection.count_documents({})
    return count


@router.get("/training/in-progress-count")
def get_in_progress_job_count(request: Request):
    count = request.app.training_in_progress_jobs_collection.count_documents({})
    return count


@router.get("/training/completed-count")
def get_completed_job_count(request: Request):
    count = request.app.training_completed_jobs_collection.count_documents({})
    return count


@router.get("/training/failed-count")
def get_failed_job_count(request: Request):
    count = request.app.training_failed_jobs_collection.count_documents({})
    return count



# ----------------- delete jobs ----------------------
@router.delete("/training/clear-all-pending")
def clear_all_pending_jobs(request: Request):
    request.app.training_pending_jobs_collection.delete_many({})

    return True


@router.delete("/training/clear-all-in-progress")
def clear_all_in_progress_jobs(request: Request):
    request.app.training_in_progress_jobs_collection.delete_many({})

    return True


@router.delete("/training/clear-all-failed")
def clear_all_failed_jobs(request: Request):
    request.app.training_failed_jobs_collection.delete_many({})

    return True

@router.delete("/training/clear-all-completed")
def clear_all_completed_jobs(request: Request):
    request.app.training_completed_jobs_collection.delete_many({})

    return True



 # --------------------- List ----------------------
@router.get("/training/list-pending")
def get_list_pending_jobs(request: Request):
    jobs = list(request.app.training_pending_jobs_collection.find({}))

    for job in jobs:
        job.pop('_id', None)

    return jobs


@router.get("/training/list-in-progress")
def get_list_in_progress_jobs(request: Request):
    jobs = list(request.app.training_in_progress_jobs_collection.find({}))

    for job in jobs:
        job.pop('_id', None)

    return jobs


@router.get("/training/list-completed")
def get_list_completed_jobs(request: Request):
    jobs = list(request.app.training_completed_jobs_collection.find({}))

    for job in jobs:
        job.pop('_id', None)

    return jobs


@router.get("/training/list-failed")
def get_list_failed_jobs(request: Request):
    jobs = list(request.app.training_failed_jobs_collection.find({}))

    for job in jobs:
        job.pop('_id', None)

    return jobs


# ---------------- Update -------------------


@router.put("/training/update-completed", description="Update in progress job and mark as completed.")
def update_job_completed(request: Request, training_task: TrainingTask):
    # check if exist
    job = request.app.training_in_progress_jobs_collection.find_one({"uuid": training_task.uuid})
    if job is None:
        raise HTTPException(status_code=404)

    # add to completed
    request.app.training_completed_jobs_collection.insert_one(training_task.to_dict())

    # remove from in progress
    request.app.training_in_progress_jobs_collection.delete_one({"uuid": training_task.uuid})

    return True


@router.put("/training/update-failed", description="Update in progress job and mark as failed.")
def update_job_failed(request: Request, training_task: TrainingTask):
    # check if exist
    job = request.app.training_in_progress_jobs_collection.find_one({"uuid": training_task.uuid})
    if job is None:
        raise HTTPException(status_code=404)

    # add to failed
    request.app.training_failed_jobs_collection.insert_one(training_task.to_dict())

    # remove from in progress
    request.app.training_in_progress_jobs_collection.delete_one({"uuid": training_task.uuid})

    return True
