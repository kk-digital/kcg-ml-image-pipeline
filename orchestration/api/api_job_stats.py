from datetime import datetime, timedelta
from fastapi import Request, APIRouter, Query
from utility.minio import cmd

router = APIRouter()

# get job stats by job type
@router.get("/job_stats/stats_by_job_type")
def get_job_stats_by_job_type(request: Request, job_type: str = Query(...)):
    pending_count = request.app.pending_jobs_collection.count_documents({
        'task_type': job_type
    })
    
    progress_count = request.app.in_progress_jobs_collection.count_documents({
        'task_type': job_type
    })

    completed_count = request.app.completed_jobs_collection.count_documents({
        'task_type': job_type
    })

    failed_count = request.app.failed_jobs_collection.count_documents({
        'task_type': job_type
    })
    return {
            'total': pending_count +  progress_count + completed_count + failed_count,
            'pending_count': pending_count,
            'progress_count': progress_count,
            'completed_count': completed_count,
            'failed_count': failed_count
    }

# get job stats by dataset
@router.get("/job_stats/stats_by_dataset")
def get_job_stats_by_job_type(request: Request, dataset: str = Query(...)):
    pending_count = request.app.pending_jobs_collection.count_documents({
        "task_input_dict.dataset": dataset
    })
    
    progress_count = request.app.in_progress_jobs_collection.count_documents({
        "task_input_dict.dataset": dataset
    })

    completed_count = request.app.completed_jobs_collection.count_documents({
        "task_input_dict.dataset": dataset
    })

    failed_count = request.app.failed_jobs_collection.count_documents({
        "task_input_dict.dataset": dataset
    })
    return {
            'total': pending_count +  progress_count + completed_count + failed_count,
            'pending_count': pending_count,
            'progress_count': progress_count,
            'completed_count': completed_count,
            'failed_count': failed_count
    }

@router.get("/job_stats/get_generated_images_per_day")
def get_number_generated_images_per_day(request: Request, start_date: str = None, end_date: str = None):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Initialize the result dictionary
    num_by_dataset_and_day = {}

    # Iterate through each day within the date range
    current_date = start_date
    while current_date <= end_date:
        # Construct the query for the current day
        query = {
            '$or': [
                {'task_type': 'image_generation_task'},
                {'task_type': 'inpainting_generation_task'}
            ],
            'task_creation_time': {
                '$gte': current_date,
                '$lt': current_date + timedelta(days=1)
            }
        }
        
        num_by_dataset = {}
        datasets = cmd.get_list_of_objects(request.app.minio_client, "datasets")
        for dataset in datasets:
            query['task_input_dict.dataset'] = dataset
            num_images = request.app.completed_jobs_collection.count_documents(query)
            num_by_dataset[dataset] = num_images

        # Store the result for the current day in the dictionary
        num_by_dataset_and_day[current_date.strftime("%Y-%m-%d")] = num_by_dataset

        # Move to the next day
        current_date += timedelta(days=1)

    return num_by_dataset_and_day

import os

@router.get("/job_stats/get_selection_datapoints_per_day")
def get_number_selection_datapoints_per_day(request: Request, start_date: str = None, end_date: str = None):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Initialize the result dictionary
    num_by_dataset_and_day = {}

    # Iterate through each day within the date range
    current_date = start_date
    while current_date <= end_date:
        # Construct the query for the current day
        query_date = current_date.strftime("%Y-%m-%d")
        num_by_dataset = {}
        datasets = cmd.get_list_of_objects(request.app.minio_client, "datasets")
        for dataset in datasets:
            # Construct the MinIO path for selection datapoints
            datapoints_path = f"{dataset}/data/ranking/aggregate"

            # List objects in the datapoints path
            objects = request.app.minio_client.list_objects("datasets", datapoints_path)
            
            # Filter objects that match the current date
            num_datapoints = len([obj for obj in objects if query_date in obj.object_name])

            # Store the result for the current day and dataset
            num_by_dataset[dataset] = num_datapoints

        # Store the result for the current day in the dictionary
        num_by_dataset_and_day[current_date.strftime("%Y-%m-%d")] = num_by_dataset

        # Move to the next day
        current_date += timedelta(days=1)

    return num_by_dataset_and_day


