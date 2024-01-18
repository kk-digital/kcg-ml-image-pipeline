from fastapi import Request, APIRouter, HTTPException, Query
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
import csv


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
    completed_query = {"task_input_dict.dataset": dataset, "task_completion_time": {"$gte": time_ago.strftime('%Y-%m-%d %H:%M:%S')}}

    count = 0

    # Take into account pending & in progress & completed jobs
    pending_count = request.app.pending_jobs_collection.count_documents(pending_query)
    in_progress_count = request.app.in_progress_jobs_collection.count_documents(in_progress_query)
    completed_count = request.app.completed_jobs_collection.count_documents(completed_query)


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
    completed_query = {"task_input_dict.dataset": dataset, "task_completion_time": {"$gte": time_ago.strftime('%Y-%m-%d %H:%M:%S')}}

    count = 0

    # Take into account pending & in progress & completed jobs
    pending_count = request.app.pending_jobs_collection.count_documents(pending_query)
    in_progress_count = request.app.in_progress_jobs_collection.count_documents(in_progress_query)
    completed_count = request.app.completed_jobs_collection.count_documents(completed_query)

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


@router.delete("/queue/image-generation/delete-completed")
def delete_completed_job(request: Request, uuid):
    query = {"uuid": uuid}
    request.app.completed_jobs_collection.delete_one(query)

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

@router.get("/queue/image-generation/list-completed-by-dataset", response_class=PrettyJSONResponse)
def get_list_completed_jobs_by_dataset(request: Request, dataset):
    jobs = list(request.app.completed_jobs_collection.find({"task_input_dict.dataset": dataset}))

    for job in jobs:
        job.pop('_id', None)

    return jobs

@router.get("/queue/image-generation/list-by-date", response_class=PrettyJSONResponse)
def get_list_completed_jobs_by_date(
    request: Request, 
    start_date: str = Query(..., description="Start date for filtering jobs"), 
    end_date: str = Query(..., description="End date for filtering jobs"),
    min_clip_sigma_score: float = Query(None, description="Minimum CLIP sigma score to filter jobs")
):
    print(f"Start Date: {start_date}, End Date: {end_date}")

    query = {
        "task_creation_time": {
            "$gte": start_date,
            "$lt": end_date
        }
    }

    # Add condition to filter by min_clip_sigma_score if provided
    if min_clip_sigma_score is not None:
        query["task_attributes_dict.image_clip_sigma_score"] = {"$gte": min_clip_sigma_score}

    jobs = list(request.app.completed_jobs_collection.find(query))

    datasets = {}
    for job in jobs:
        dataset_name = job.get("task_input_dict", {}).get("dataset")
        job_uuid = job.get("uuid")
        file_path = job.get("task_output_file_dict", {}).get("output_file_path")

        if not dataset_name or not job_uuid or not file_path:
            continue

        if dataset_name not in datasets:
            datasets[dataset_name] = []

        job_info = {
            "job_uuid": job_uuid,
            "file_path": file_path
        }

        datasets[dataset_name].append(job_info)

    return datasets






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
        return False
    
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
        return False

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


# --------------- Job info ---------------------
@router.get("/job/get-completed-job-by-hash")
def get_completed_job_by_hash(request: Request, image_hash):
    query = {"task_output_file_dict.output_file_hash": image_hash}
    job = request.app.completed_jobs_collection.find_one(query)

    if job is None:
        return None

    job.pop('_id', None)

    return job

@router.get("/job/get-job/{uuid}", response_class=PrettyJSONResponse)
def get_job_by_uuid(request: Request, uuid: str):
    # Assuming the job's UUID is stored in the 'uuid' field
    query = {"uuid": uuid}
    job = request.app.completed_jobs_collection.find_one(query)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Remove the '_id' field to avoid issues with JSON serialization
    job.pop('_id', None)

    return job

@router.get("/job/get-jobs", response_class=PrettyJSONResponse)
def get_jobs_by_uuids(request: Request, uuids: List[str] = Query(None)):
    # Assuming the job's UUID is stored in the 'uuid' field
    query = {"uuid": {"$in": uuids}}
    jobs = request.app.completed_jobs_collection.find(query)
    if jobs is None:
        raise HTTPException(status_code=404, detail="Job not found")

    job_list = []
    for job in jobs:
        job.pop('_id', None)
        job_list.append(job)

    return job_list

# --------------- Get Job With Required Fields ---------------------

@router.get("/get-image-generation/by-hash/{image_hash}", response_class=PrettyJSONResponse)
def get_job_by_image_hash(request: Request, image_hash: str, fields: List[str] = Query(None)):
    # Create a projection object that understands nested fields using dot notation
    projection = {field: 1 for field in fields} if fields else {}
    projection['_id'] = 0  # Exclude the _id field

    job = request.app.completed_jobs_collection.find_one({"task_output_file_dict.output_file_hash": image_hash}, projection)
    if job:
        # If specific fields are requested, filter the job dictionary
        if fields is not None:
            filtered_job = {}
            for field in fields:
                field_parts = field.split('.')
                if len(field_parts) == 1:
                    # Top-level field
                    filtered_job[field] = job.get(field)
                else:
                    # Nested fields
                    nested_field = job
                    for part in field_parts:
                        nested_field = nested_field.get(part, {})
                    if isinstance(nested_field, dict):
                        nested_field = None
                    filtered_job[field_parts[-1]] = nested_field
            return filtered_job
        return job
    else:
        print("Job Not Found")
    

@router.get("/get-image-generation/by-job-id/{job_id}", response_class=PrettyJSONResponse)
def get_job_by_job_id(request: Request, job_id: str, fields: List[str] = Query(None)):
    # Create a projection object that understands nested fields using dot notation
    projection = {field: 1 for field in fields} if fields else {}
    projection['_id'] = 0  # Exclude the _id field

    job = request.app.completed_jobs_collection.find_one({"uuid": job_id}, projection)
    if job:
        # If specific fields are requested, filter the job dictionary
        if fields is not None:
            filtered_job = {}
            for field in fields:
                field_parts = field.split('.')
                if len(field_parts) == 1:
                    # Top-level field
                    filtered_job[field] = job.get(field)
                else:
                    # Nested fields
                    nested_field = job
                    for part in field_parts:
                        nested_field = nested_field.get(part, {})
                    if isinstance(nested_field, dict):
                        nested_field = None
                    filtered_job[field_parts[-1]] = nested_field
            return filtered_job
        return job
    else:
        print("Job Not Found")


# --------------- Add completed job attributes ---------------------
@router.put("/job/add-attributes", description="Adds the attributes to a completed job.")
def add_attributes_job_completed(request: Request,
                                 image_hash,
                                 image_clip_score,
                                 image_clip_percentile,
                                 image_clip_sigma_score,
                                 text_embedding_score,
                                 text_embedding_percentile,
                                 text_embedding_sigma_score,
                                 delta_sigma_score):
    query = {"task_output_file_dict.output_file_hash": image_hash}

    update_query = {"$set": {"task_attributes_dict.image_clip_score": image_clip_score,
                             "task_attributes_dict.image_clip_percentile": image_clip_percentile,
                             "task_attributes_dict.image_clip_sigma_score": image_clip_sigma_score,
                             "task_attributes_dict.text_embedding_score": text_embedding_score,
                             "task_attributes_dict.text_embedding_percentile": text_embedding_percentile,
                             "task_attributes_dict.text_embedding_sigma_score": text_embedding_sigma_score,
                             "task_attributes_dict.delta_sigma_score": delta_sigma_score}}

    request.app.completed_jobs_collection.update_one(query, update_query)

    return True


@router.post("/worker-stats", response_class=PrettyJSONResponse)
def get_worker_stats(csv_file_path: str = Query(...), ssh_key_path: str = Query(...), server_address: str = Query("123.176.98.90")):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    gpu_stats_all = []

    try:
        # Open the CSV file and read port numbers
        with open(csv_file_path, mode='r', encoding='utf-8-sig') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                port_number = int(row['port_numbers'])

                try:
                    # SSH connection for each port number
                    ssh.connect(server_address, port=port_number, username='root', key_filename=ssh_key_path)

                    # Command to retrieve GPU stats
                    command = '''
                    python -c "import json; import socket; import GPUtil; print(json.dumps([{\'id\': gpu.id, \'temperature\': gpu.temperature, \'load\': gpu.load, \'total_memory\': gpu.memoryTotal, \'used_memory\': gpu.memoryUsed, \'worker_name\': socket.gethostname()} for gpu in GPUtil.getGPUs()]))"
                    '''
                    stdin, stdout, stderr = ssh.exec_command(command)

                    stderr_output = stderr.read().decode('utf-8')
                    if stderr_output:
                        print(f"Error on server {server_address}:{port_number}:", stderr_output)
                        continue

                    gpu_stats = json.loads(stdout.read().decode('utf-8'))
                    gpu_stats_all.extend(gpu_stats)

                finally:
                    ssh.close()

        return gpu_stats_all

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))