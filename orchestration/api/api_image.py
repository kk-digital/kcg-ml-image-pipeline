from fastapi import Request, HTTPException, APIRouter, Response, Query, status
from datetime import datetime

import pymongo
from utility.minio import cmd
from utility.path import separate_bucket_and_file_path
from .api_utils import PrettyJSONResponse


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

@router.get("/image/list-metadata", response_class=PrettyJSONResponse)
def get_images_metadata(
    request: Request,
    dataset: str = None,
    limit: int = 20,
    offset: int = 0,
    start_date: str = None,
    end_date: str = None,
    order: str = Query("desc", description="Order in which the data should be returned. 'asc' for oldest first, 'desc' for newest first")
):

    print(f"start_date: {start_date}") 

    # Construct the initial query
    query = {
        '$or': [
            {'task_type': 'image_generation_task'},
            {'task_type': 'inpainting_generation_task'}
        ],
        'task_input_dict.dataset': dataset
    }

    # Update the query based on provided start_date and end_date
    if start_date and end_date:
        query['task_creation_time'] = {'$gte': start_date, '$lte': end_date}
    elif start_date:
        query['task_creation_time'] = {'$gte': start_date}
    elif end_date:
        query['task_creation_time'] = {'$lte': end_date}

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


# functions for getting residuals, scores and percentiles
def score(request, model_id, job):
    query = {"image_hash": job['task_output_file_dict']['output_file_hash'],
            "model_id": model_id}

    item = request.app.image_scores_collection.find_one(query)

    if item is None:
        return 0
    else:
        return item['score']

def percentile(request, model_id, job):
    query = {"image_hash": job['task_output_file_dict']['output_file_hash'],
            "model_id": model_id}

    item = request.app.image_percentiles_collection.find_one(query)

    if item is None:
        return 0
    else:
        return item['percentile']

def residual(request, model_id, job):
    query = {"image_hash": job['task_output_file_dict']['output_file_hash'],
            "model_id": model_id}
    
    item = request.app.image_residuals_collection.find_one(query)

    if item is None:
        return 0
    else:
        return item['residual']

@router.get("/image/sorted-list-metadata", response_class=PrettyJSONResponse)
def sorted_list_metadata(
    request: Request,
    dataset: str = Query(...),
    limit: int = 20,
    offset: int = 0,
    start_date: str = None,
    end_date: str = None,
    sort_order: str = 'asc',
    sort_field: str = Query('score', description="field to sort the images by, the options are 'score', 'residual' and 'percentile'.") ,
    model_id: int=Query(...) ,
    min_score: float = None,
    max_score: float = None,
    min_percentile: float = None,
    max_percentile: float = None
):
    # Construct the initial query
    query = {
        '$or': [
            {'task_type': 'image_generation_task'},
            {'task_type': 'inpainting_generation_task'}
        ],
        'task_input_dict.dataset': dataset
    }

    # Update the query based on provided start_date and end_date
    if start_date and end_date:
        query['task_creation_time'] = {'$gte': start_date, '$lte': end_date}
    elif start_date:
        query['task_creation_time'] = {'$gte': start_date}
    elif end_date:
        query['task_creation_time'] = {'$lte': end_date}

    # Retrieve image metadata
    jobs =list(request.app.completed_jobs_collection.find(query))

    # Sorting order
    reverse = sort_order == 'desc'
        
    for job in jobs:
        job['score'] = score(request, model_id, job)
        job['percentile'] = percentile(request, model_id, job)
        job['residual'] = residual(request, model_id, job)

    jobs = sorted(jobs, key=lambda x: x[sort_field], reverse=reverse)

    # Extract metadata
    images_metadata = []
    jobs=jobs[offset:offset + limit]
    for job in jobs:
        image_meta_data = {
            'dataset': job['task_input_dict']['dataset'],
            'task_type': job['task_type'],
            'image_path': job['task_output_file_dict']['output_file_path'],
            'image_hash': job['task_output_file_dict']['output_file_hash'],
            'score': job['score'],
            'percentile': job['percentile'],
            'residual': job['residual']
        }
        images_metadata.append(image_meta_data)
    
    # Filter images by min_score and max_score if provided
    if min_score is not None or max_score is not None:
        images_metadata = [
            img for img in images_metadata
            if (min_score is None or (img['score'] >= min_score)) and
            (max_score is None or (img['score'] <= max_score))
        ]

    # Filter images by min_percentile and max_percentile if provided
    if min_percentile is not None or max_percentile is not None:
        images_metadata = [
            img for img in images_metadata
            if (min_percentile is None or (img['percentile'] >= min_percentile)) and
            (max_percentile is None or (img['percentile'] <= max_percentile))
        ]

    return images_metadata




