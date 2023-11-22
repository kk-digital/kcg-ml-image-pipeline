from fastapi import Request, HTTPException, APIRouter, Response, Query, status
from datetime import datetime, timedelta

import pymongo
from utility.minio import cmd
from utility.path import separate_bucket_and_file_path
from .api_utils import PrettyJSONResponse
from .api_ranking import get_image_rank_use_count
import os

router = APIRouter()



@router.get("/image/get_random_image", response_class=PrettyJSONResponse)
def get_random_image(request: Request, dataset: str = Query(...)):  # Remove the size parameter
  
    # Use $sample to get one random document
    documents = request.app.completed_jobs_collection.aggregate([
        {"$match": {"task_input_dict.dataset": dataset}},
        {"$sample": {"size": 1}}
    ])

    # Convert cursor type to list
    documents = list(documents)

    # Ensure the list isn't empty (this is just a safety check)
    if not documents:
        raise HTTPException(status_code=404, detail="No image found for the given dataset")

    # Remove the auto generated _id field from the document
    documents[0].pop('_id', None)

    # Return the image in the response
    return {"image": documents[0]}

@router.get("/image/get_image_details")
def get_image_details(request: Request, image_path: str = Query(...)):
    # Query the database to retrieve the image details by its ID
    document = request.app.completed_jobs_collection.find_one(
        {"task_output_file_dict.output_file_path": image_path}
    )

    if document is None:
        raise HTTPException(status_code=404, detail="Image not found")

    # Remove the auto-generated _id field from the document
    document.pop('_id', None)

    # Return the image details
    return {"image_details": document}  
    
@router.get("/image/get_random_image_list", response_class=PrettyJSONResponse)
def get_random_image_list(request: Request, dataset: str = Query(...), size: int = Query(1)):  
    # Use Query to get the dataset and size from query parameters

    distinct_documents = []
    tried_ids = set()

    while len(distinct_documents) < size:
        # Use $sample to get 'size' random documents
        documents = request.app.completed_jobs_collection.aggregate([
            {"$match": {"task_input_dict.dataset": dataset, "_id": {"$nin": list(tried_ids)}}},  # Exclude already tried ids
            {"$sample": {"size": size - len(distinct_documents)}}  # Only fetch the remaining needed size
        ])

        # Convert cursor type to list
        documents = list(documents)
        distinct_documents.extend(documents)

        # Store the tried image ids
        tried_ids.update([doc["_id"] for doc in documents])

        # Ensure only distinct images are retained
        seen = set()
        distinct_documents = [doc for doc in distinct_documents if doc["_id"] not in seen and not seen.add(doc["_id"])]

    for doc in distinct_documents:
        doc.pop('_id', None)  # remove the auto generated field
    
    # Return the images as a list in the response
    return {"images": distinct_documents}

@router.get("/image/get_random_previously_ranked_image_list", response_class=PrettyJSONResponse)
def get_random_previously_ranked_image_list(request: Request, dataset: str = Query(...), size: int = Query(1)):
    # Use Query to get the dataset and size from query parameters

    distinct_documents = []
    tried_ids = set()

    while len(distinct_documents) < size:
        # Use $sample to get 'size' random documents
        documents = request.app.completed_jobs_collection.aggregate([
            {"$match": {"task_input_dict.dataset": dataset, "_id": {"$nin": list(tried_ids)}}},
            # Exclude already tried ids
            {"$sample": {"size": size - len(distinct_documents)}}  # Only fetch the remaining needed size
        ])

        # Convert cursor type to list
        documents = list(documents)

        # Store the tried image ids
        tried_ids.update([doc["_id"] for doc in documents])

        # use only documents that has rank use count greater than 0
        prev_ranked_docs = []
        for doc in documents:
            print("checking ...")
            try:
                count = get_image_rank_use_count(request, doc["task_output_file_dict"]["output_file_hash"])
            except:
                count = 0
            print("checking count=", count)

            if count > 0:
                print("appending...")
                prev_ranked_docs.append(doc)

        distinct_documents.extend(prev_ranked_docs)

        # Ensure only distinct images are retained
        seen = set()
        distinct_documents = [doc for doc in distinct_documents if doc["_id"] not in seen and not seen.add(doc["_id"])]

    for doc in distinct_documents:
        doc.pop('_id', None)  # remove the auto generated field

    # Return the images as a list in the response
    return {"images": distinct_documents}

@router.get("/image/get_random_image_by_date_range", response_class=PrettyJSONResponse)
def get_random_image_date_range(
    request: Request,
    dataset: str = None,
    start_date: str = None,
    end_date: str = None,
    size: int = None
):

    query = {
        'task_input_dict.dataset': dataset
    }

    # Update the query based on provided start_date and end_date
    if start_date and end_date:
        query['task_creation_time'] = {'$gte': start_date, '$lte': end_date}
    elif start_date:
        query['task_creation_time'] = {'$gte': start_date}
    elif end_date:
        query['task_creation_time'] = {'$lte': end_date}

    # Create the aggregation pipeline
    aggregation_pipeline = [{"$match": query}]

    # Add the $sample stage if the size is provided
    if size:
        aggregation_pipeline.append({"$sample": {"size": size}})

    documents = request.app.completed_jobs_collection.aggregate(aggregation_pipeline)

    # Convert the cursor to a list
    documents = list(documents)

    # Remove the auto-generated field for each document
    for document in documents:
        document.pop('_id', None)

    return documents

"""
@router.get("/image/data-by-filepath")
def get_image_data_by_filepath(request: Request, file_path: str = None):

    bucket_name, file_path = separate_bucket_and_file_path(file_path)

    image_data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, file_path)

    # Load data into memory
    content = image_data.read()

    response = Response(content=content, media_type="image/jpeg")

    return response
"""

# TODO: deprecate
@router.get("/images/{file_path:path}")
def get_image_data_by_filepath_2(request: Request, file_path: str):
    bucket_name, file_path = separate_bucket_and_file_path(file_path)
    file_path = file_path.replace("\\", "/")
    image_data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, file_path)

    # Load data into memory
    if image_data is not None:
        content = image_data.read()
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image with this path doesn't exist") 

    response = Response(content=content, media_type="image/jpeg")

    return response
  
@router.get("/image/list-image-metadata-by-dataset", response_class=PrettyJSONResponse)
def get_images_metadata(
    request: Request,
    dataset: str = None,
    limit: int = 20,
    offset: int = 0,
    start_date: str = None,
    end_date: str = None,
    order: str = Query("desc", description="Order in which the data should be returned. 'asc' for oldest first, 'desc' for newest first"),
    time_interval: int = Query(None, description="Time interval in minutes or hours"),
    time_unit: str = Query("minutes", description="Time unit, either 'minutes' or 'hours")
):

    # Calculate the time threshold based on the current time and the specified interval
    if time_interval is not None:
        current_time = datetime.utcnow()
        if time_unit == "minutes":
            threshold_time = current_time - timedelta(minutes=time_interval)
        elif time_unit == "hours":
            threshold_time = current_time - timedelta(hours=time_interval)
        else:
            raise HTTPException(status_code=400, detail="Invalid time unit. Use 'minutes' or 'hours'.")
    else:
        threshold_time = None

    print(f"start_date: {start_date}, threshold_time: {threshold_time}")

    # Construct the initial query
    query = {
        '$or': [
            {'task_type': 'image_generation_task'},
            {'task_type': 'inpainting_generation_task'}
        ],
        'task_input_dict.dataset': dataset
    }

    # Update the query based on provided start_date, end_date, and threshold_time
    if start_date and end_date:
        query['task_creation_time'] = {'$gte': start_date, '$lte': end_date}
    elif start_date:
        query['task_creation_time'] = {'$gte': start_date}
    elif end_date:
        query['task_creation_time'] = {'$lte': end_date}
    elif threshold_time:
        query['task_creation_time'] = {'$gte': threshold_time}

    # Decide the sort order based on the 'order' parameter
    sort_order = -1 if order == "desc" else 1

    jobs = request.app.completed_jobs_collection.find(query).sort('task_creation_time', sort_order).skip(offset).limit(limit)

    images_metadata = []
    for job in jobs:
        image_meta_data = {
            'dataset': job['task_input_dict']['dataset'],
            'task_type': job['task_type'],
            'image_path': job['task_output_file_dict']['output_file_path'],
            'image_hash': job['task_output_file_dict']['output_file_hash']
        }
        images_metadata.append(image_meta_data)

    return images_metadata

@router.get("/image/get_random_image_with_time", response_class=PrettyJSONResponse)
def get_random_image_with_time(
    request: Request,
    dataset: str = Query(...),
    time_interval: int = Query(..., description="Time interval in minutes or hours"),
    time_unit: str = Query("minutes", description="Time unit, either 'minutes' or 'hours"),
    size: int = Query(1, description="Number of images to retrieve")
):
    # Calculate the time threshold based on the current time and the specified interval
    current_time = datetime.utcnow()
    if time_unit == "minutes":
        threshold_time = current_time - timedelta(minutes=time_interval)
    elif time_unit == "hours":
        threshold_time = current_time - timedelta(hours=time_interval)
    else:
        raise HTTPException(status_code=400, detail="Invalid time unit. Use 'minutes' or 'hours'.")

    # Use $match to filter documents based on dataset and creation time
    aggregation_pipeline = [
        {"$match": {
            "task_input_dict.dataset": dataset,
            "task_creation_time": {"$gte": threshold_time}
        }},
        {"$sample": {"size": size}}
    ]

    documents = request.app.completed_jobs_collection.aggregate(aggregation_pipeline)

    # Convert cursor type to list
    documents = list(documents)

    # Ensure the list isn't empty (this is just a safety check)
    if not documents:
        raise HTTPException(status_code=404, detail=f"No image found for the given dataset within the last {time_interval} {time_unit}")

    # Remove the auto-generated _id field from each document
    for document in documents:
        document.pop('_id', None)

    return documents



# New Endpoints with /static/ prefix


@router.get("/static/images/{file_path:path}")
def get_image_data_by_filepath_2(request: Request, file_path: str):
    bucket_name, file_path = separate_bucket_and_file_path(file_path)
    file_path = file_path.replace("\\", "/")
    image_data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, file_path)

    # Load data into memory
    if image_data is not None:
        content = image_data.read()
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image with this path doesn't exist") 

    response = Response(content=content, media_type="image/jpeg")

    return response


@router.get("/get-image-by-job-uuid/{job_uuid}", response_class=Response)
def get_image_by_job_uuid(request: Request, job_uuid: str):
    # Fetch the job from the completed_jobs_collection using the UUID
    job = request.app.completed_jobs_collection.find_one({"uuid": job_uuid})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Extract the output file path from the job data
    output_file_path = job.get("task_output_file_dict", {}).get("output_file_path")
    if not output_file_path:
        raise HTTPException(status_code=404, detail="Output file path not found in job data")

    original_filename = os.path.basename(output_file_path)

    # Fetch the image from MinIO
    bucket_name, file_path = separate_bucket_and_file_path(output_file_path)
    file_path = file_path.replace("\\", "/")
    image_data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, file_path)

    # Load data into memory
    if image_data is not None:
        content = image_data.read()
    else:
        raise HTTPException(
            status_code=404,
            detail="Image with this path doesn't exist") 

    # Return the image in the response
    headers = {"Content-Disposition": f"attachment; filename={original_filename}"}
    return Response(content=content, media_type="image/jpeg", headers=headers)
