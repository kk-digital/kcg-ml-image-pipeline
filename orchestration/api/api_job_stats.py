from datetime import datetime, timedelta
from fastapi import Request, APIRouter, Query
from utility.minio import cmd
from .api_utils import  ErrorCode, ApiResponseHandlerV1, StandardSuccessResponseV1, JobStatsResponse, ListGenerationsCountPerDayResponse
from dateutil.parser import parse

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


@router.get("/queue/image-generation/get-job-counts-by-type",
            description="Get job statistics by job type",
            response_model=StandardSuccessResponseV1[JobStatsResponse],
            tags=["job-stats"],
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
async def get_job_stats_by_job_type(request: Request, job_type: str = Query(..., description="Type of job to get statistics for")):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        pending_count = request.app.pending_jobs_collection.count_documents({'task_type': job_type})
        progress_count = request.app.in_progress_jobs_collection.count_documents({'task_type': job_type})
        completed_count = request.app.completed_jobs_collection.count_documents({'task_type': job_type})
        failed_count = request.app.failed_jobs_collection.count_documents({'task_type': job_type})

        total_count = pending_count + progress_count + completed_count + failed_count
        
        stats = {
            'total': total_count,
            'pending_count': pending_count,
            'progress_count': progress_count,
            'completed_count': completed_count,
            'failed_count': failed_count
        }
        
        return response_handler.create_success_response_v1(response_data=stats, http_status_code=200)
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=f"An error occurred: {str(e)}",
            http_status_code=500
        )    

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

@router.get("/queue/image-generation/get-job-counts-by-dataset",
            description="Get job counts by dataset",
            response_model=StandardSuccessResponseV1[JobStatsResponse],
            tags=["job-stats"],
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
async def get_job_counts_by_dataset(request: Request, dataset: str = Query(..., description="Dataset to get job counts for")):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
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

        total_count = pending_count + progress_count + completed_count + failed_count

        stats = {
            'total': total_count,
            'pending_count': pending_count,
            'progress_count': progress_count,
            'completed_count': completed_count,
            'failed_count': failed_count
        }

        return response_handler.create_success_response_v1(response_data=stats, http_status_code=200)
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=f"An error occurred: {str(e)}",
            http_status_code=500
        )


@router.get("/job_stats/get_generated_images_per_day")
def get_number_generated_images_per_day(request: Request, start_date: str = Query(...), end_date: str = Query(...)):
    # Convert the date strings to datetime objects
    start_date = parse(start_date)
    end_date = parse(end_date)

    # Initialize the result dictionary
    num_by_dataset_and_day = {}

    # Iterate through each day within the date range
    current_date = start_date
    while current_date <= end_date:
        print(current_date + timedelta(days=1))
        # Construct the query for the current day
        query = {
            '$or': [
                {'task_type': 'image_generation_sd_1_5'},
                {'task_type': 'inpainting_sd_1_5'},
                {'task_type': 'image_generation_kandinsky'},
                {'task_type': 'inpainting_kandinsky'},
                {'task_type': 'img2img_generation_kandinsky'}
            ],
            'task_completion_time': {
                '$gte': current_date.strftime("%Y-%m-%d %H:%M:%S"),
                '$lte': (current_date + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
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


@router.get("/queue/image-generation/get-generations-count-per-day",
            description="Get number of generated images per day within the date range",
            response_model=StandardSuccessResponseV1[ListGenerationsCountPerDayResponse],
            tags=["queue", "image-generation"],
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
async def get_generations_count_per_day(
    request: Request,
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format")
):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        # Convert the date strings to datetime objects
        start_date_dt = parse(start_date)
        end_date_dt = parse(end_date)

        # Initialize the result dictionary
        num_by_dataset_and_day = {}

        # Iterate through each day within the date range
        current_date = start_date_dt
        while current_date <= end_date_dt:
            # Construct the query for the current day
            query = {
                '$or': [
                    {'task_type': 'image_generation_sd_1_5'},
                    {'task_type': 'inpainting_sd_1_5'},
                    {'task_type': 'image_generation_kandinsky'},
                    {'task_type': 'inpainting_kandinsky'},
                    {'task_type': 'img2img_generation_kandinsky'}
                ],
                'task_completion_time': {
                    '$gte': current_date.strftime("%Y-%m-%d %H:%M:%S"),
                    '$lte': (current_date + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
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

        return response_handler.create_success_response_v1(
            response_data={"results": num_by_dataset_and_day},
            http_status_code=200
        )
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=f"An error occurred: {str(e)}",
            http_status_code=500
        )    


@router.get("/job_stats/get_selection_datapoints_per_day")
def get_number_selection_datapoints_per_day(request: Request, start_date: str = Query(...), end_date: str = Query(...)):
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
            datapoints_path = f"{dataset}/data/ranking/aggregate/{query_date}"

            # List objects in the datapoints path
            objects = request.app.minio_client.list_objects("datasets", prefix=datapoints_path)
            
            # Filter objects that match the current date
            num_datapoints = len([obj.object_name for obj in objects])

            # Store the result for the current day and dataset
            num_by_dataset[dataset] = num_datapoints

        # Store the result for the current day in the dictionary
        num_by_dataset_and_day[current_date.strftime("%Y-%m-%d")] = num_by_dataset

        # Move to the next day
        current_date += timedelta(days=1)

    return num_by_dataset_and_day

@router.get("/rank/get-datapoints-count-per-day",
            description="Get number of selection datapoints per day within the date range",
            response_model=StandardSuccessResponseV1[ListGenerationsCountPerDayResponse],
            tags=["rank"],
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
async def get_datapoints_count_per_day(
    request: Request,
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format")
):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        # Convert the date strings to datetime objects
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Initialize the result dictionary
        num_by_dataset_and_day = {}

        # Iterate through each day within the date range
        current_date = start_date_dt
        while current_date <= end_date_dt:
            # Construct the query for the current day
            query_date = current_date.strftime("%Y-%m-%d")
            num_by_dataset = {}
            datasets = cmd.get_list_of_objects(request.app.minio_client, "datasets")
            for dataset in datasets:
                # Construct the MinIO path for selection datapoints
                datapoints_path = f"{dataset}/data/ranking/aggregate/{query_date}"

                # List objects in the datapoints path
                objects = request.app.minio_client.list_objects("datasets", prefix=datapoints_path)
                
                # Filter objects that match the current date
                num_datapoints = len([obj.object_name for obj in objects])

                # Store the result for the current day and dataset
                num_by_dataset[dataset] = num_datapoints

            # Store the result for the current day in the dictionary
            num_by_dataset_and_day[current_date.strftime("%Y-%m-%d")] = num_by_dataset

            # Move to the next day
            current_date += timedelta(days=1)

        return response_handler.create_success_response_v1(
            response_data={"results": num_by_dataset_and_day},
            http_status_code=200
        )
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=f"An error occurred: {str(e)}",
            http_status_code=500
        )