import datetime
from fastapi import APIRouter, Request

from orchestration.api.mongo_schemas import Task

router = APIRouter()
@router.post("/test-task", response_model=Task, description="Test API Endpoint")
def test_task(task: Task, request: Request):
    for i in range(5):
        request.app.completed_jobs_collection.insert_one(task.to_dict())

    return task

@router.post("/test-get-image", description="Test API Endpoint")
def test_task(task: Task, request: Request):
    request.app.completed_jobs_collection.insert_one(task.to_dict())

    return task