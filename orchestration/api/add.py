from fastapi import Request, APIRouter
import uuid
from datetime import datetime
from orchestration.api.schemas import Task

router = APIRouter()


@router.post("/add-job", description="Add a job to db")
def add_job(request: Request, task: Task):
    # generate UUID
    task.uuid = str(uuid.uuid4())
    # add task creation time
    task.task_creation_time = datetime.now()

    request.app.pending_jobs_collection.insert_one(task.to_dict())

    return {"uuid": task.uuid, "creation_time": task.task_creation_time}