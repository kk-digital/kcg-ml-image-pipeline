from fastapi import Request, APIRouter, HTTPException
import uuid
from datetime import datetime
from orchestration.api.mongo_schemas import Task
from orchestration.api.api_dataset import get_sequential_id

router = APIRouter()


# -------------------- Get -------------------------


@router.get("/get-job")
def get_job(request: Request, task_type: str = None):
    query = {}
    if task_type != None:
        query = {"task_type": task_type}

    # find
    job = request.app.pending_jobs_collection.find_one(query)
    if job is None:
        raise HTTPException(status_code=404)

    # delete from pending
    request.app.pending_jobs_collection.delete_one({"uuid": job["uuid"]})
    # add to in progress
    request.app.in_progress_jobs_collection.insert_one(job)

    # remove the auto generated field
    job.pop('_id', None)

    return job
