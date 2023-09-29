from fastapi import Request, APIRouter
import uuid
from datetime import datetime
from orchestration.api.schemas import Task
from orchestration.api.get import get_sequential_id

router = APIRouter()


@router.post("/add-job", description="Add a job to db")
def add_job(request: Request, task: Task):
    if task.uuid in ["", None]:
        # generate since its empty
        task.uuid = str(uuid.uuid4())

    # add task creation time
    task.task_creation_time = datetime.now()

    # check if file_path is blank
    if task.task_input_dict is None or "file_path" not in task.task_input_dict or task.task_input_dict["file_path"] in ['', "[auto]", "[default]"]:
        dataset_name = task.task_input_dict["dataset"]
        # get file path
        sequential_id_arr = get_sequential_id(request, dataset=dataset_name)
        new_file_path = "{}.jpg".format(sequential_id_arr[0])
        task.task_input_dict["file_path"] = new_file_path

    request.app.pending_jobs_collection.insert_one(task.to_dict())

    return {"uuid": task.uuid, "creation_time": task.task_creation_time}
