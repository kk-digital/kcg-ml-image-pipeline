from fastapi import Request, HTTPException, APIRouter, Response, Query, status, File, UploadFile
from datetime import datetime, timedelta
from typing import Optional
import pymongo
from utility.minio import cmd
from utility.path import separate_bucket_and_file_path
from .mongo_schemas import Task, ImageMetadata, UUIDImageMetadata
from .api_utils import PrettyJSONResponse, StandardSuccessResponseV1, ApiResponseHandlerV1, UrlResponse, ErrorCode
from .api_ranking import get_image_rank_use_count
import os
from .api_utils import find_or_create_next_folder_and_index
import io
from typing import List
from PIL import Image


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
        return {"image": None}

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
        return {"image_details": None}
        
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
        filter = [
            {"$match": {"task_input_dict.dataset": dataset, "_id": {"$nin": list(tried_ids)}}},  # Exclude already tried ids
            {"$sample": {"size": size - len(distinct_documents)}}  # Only fetch the remaining needed size
        ]

        if dataset == "any":
            filter = [
                {"$match": {"_id": {"$nin": list(tried_ids)}}},
                # Exclude already tried ids
                {"$sample": {"size": size - len(distinct_documents)}}  # Only fetch the remaining needed size
            ]

        documents = request.app.completed_jobs_collection.aggregate(filter)

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
def get_random_previously_ranked_image_list(
    request: Request, 
    dataset: str = Query(...), 
    size: int = Query(1),
    prompt_generation_policy: Optional[str] = None,
    start_date: str = None,
    end_date: str = None,
    time_interval: int = Query(None, description="Time interval in minutes or hours"),
    time_unit: str = Query("minutes", description="Time unit, either 'minutes' or 'hours")
):
    distinct_documents = []
    tried_ids = set()

    match_query = {"task_input_dict.dataset": dataset, "_id": {"$nin": list(tried_ids)}}
    if prompt_generation_policy:
        match_query["prompt_generation_data.prompt_generation_policy"] = prompt_generation_policy

    # Apply the date/time filters
    if start_date and end_date:
        match_query["task_creation_time"] = {"$gte": start_date, "$lte": end_date}
    elif start_date:
        match_query["task_creation_time"] = {"$gte": start_date}
    elif end_date:
        match_query["task_creation_time"] = {"$lte": end_date}

    if time_interval is not None and time_unit:
        current_time = datetime.utcnow()
        if time_unit == "minutes":
            threshold_time = current_time - timedelta(minutes=time_interval)
        elif time_unit == "hours":
            threshold_time = current_time - timedelta(hours=time_interval)
        else:
            raise HTTPException(status_code=400, detail="Invalid time unit. Use 'minutes' or 'hours'.")

        # Convert threshold_time to a string in ISO format with milliseconds precision
        threshold_time_str = threshold_time.isoformat(timespec='milliseconds')
        time_query = match_query.get("task_creation_time", {})
        time_query["$gte"] = threshold_time_str
        match_query["task_creation_time"] = time_query

    while len(distinct_documents) < size:
        documents = request.app.completed_jobs_collection.aggregate([
            {"$match": match_query}, # Use the updated match query with prompt_generation_policy
            {"$sample": {"size": size - len(distinct_documents)}}
        ])

        documents = list(documents)
        tried_ids.update([doc["_id"] for doc in documents])

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
        seen = set()
        distinct_documents = [doc for doc in distinct_documents if doc["_id"] not in seen and not seen.add(doc["_id"])]

    for doc in distinct_documents:
        doc.pop('_id', None)  # Remove the auto-generated field

    return {"images": distinct_documents}



@router.get("/image/get_random_image_by_date_range", response_class=PrettyJSONResponse)
def get_random_image_date_range(
    request: Request,
    dataset: str = None,
    start_date: str = None,
    end_date: str = None,
    size: int = None,
    prompt_generation_policy: Optional[str] = None  # Optional query parameter
):
    query = {
        'task_input_dict.dataset': dataset
    }
    if start_date and end_date:
        query['task_creation_time'] = {'$gte': start_date, '$lte': end_date}
    elif start_date:
        query['task_creation_time'] = {'$gte': start_date}
    elif end_date:
        query['task_creation_time'] = {'$lte': end_date}

    # Include prompt_generation_policy in the query if provided
    if prompt_generation_policy:
        query['prompt_generation_data.prompt_generation_policy'] = prompt_generation_policy

    aggregation_pipeline = [{"$match": query}]
    if size:
        aggregation_pipeline.append({"$sample": {"size": size}})

    documents = request.app.completed_jobs_collection.aggregate(aggregation_pipeline)
    documents = list(documents)

    for document in documents:
        document.pop('_id', None)  # Remove the auto-generated field

    return documents

@router.get("/image/get_random_image_by_classifier_score", response_class=PrettyJSONResponse)
def get_random_image_date_range(
    request: Request,
    rank_id: int = None,
    start_date: str = None,
    end_date: str = None,
    min_score: float= 0.6,
    size: int = None,
    prompt_generation_policy: Optional[str] = None  # Optional query parameter
):
    query = {
        '$or': [
            {'task_type': 'image_generation_sd_1_5'},
            {'task_type': 'inpainting_sd_1_5'},
            {'task_type': 'image_generation_kandinsky'},
            {'task_type': 'inpainting_kandinsky'},
            {'task_type': 'img2img_generation_kandinsky'}
        ]
    }

    if start_date and end_date:
        query['task_creation_time'] = {'$gte': start_date, '$lte': end_date}
    elif start_date:
        query['task_creation_time'] = {'$gte': start_date}
    elif end_date:
        query['task_creation_time'] = {'$lte': end_date}

    # Include prompt_generation_policy in the query if provided
    if prompt_generation_policy:
        query['prompt_generation_data.prompt_generation_policy'] = prompt_generation_policy

    # If rank_id is provided, adjust the query to consider classifier scores
    if rank_id is not None:
        rank = request.app.rank_model_models_collection.find_one({'rank_model_id': rank_id})
        if rank is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Rank model with this id doesn't exist")

        classifier_id = rank.get("classifier_id")
        if classifier_id is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="This Rank has no relevance classifier model assigned to it")

        classifier_query = {'classifier_id': classifier_id}
        if min_score is not None:
            classifier_query['score'] = {'$gte': min_score}

        classifier_scores = request.app.image_classifier_scores_collection.find(classifier_query)
        if classifier_scores is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="The relevance classifier model has no scores.")

        job_uuids = [score['job_uuid'] for score in classifier_scores]
        query['job_uuid'] = {'$in': job_uuids}

    aggregation_pipeline = [{"$match": query}]
    if size:
        aggregation_pipeline.append({"$sample": {"size": size}})

    documents = request.app.completed_jobs_collection.aggregate(aggregation_pipeline)
    documents = list(documents)

    # Map job_uuid to their corresponding scores
    classifier_scores_map = {
        score['job_uuid']: score['score']
        for score in request.app.image_classifier_scores_collection.find(classifier_query)
    }

    # Add classifier score to each document
    for document in documents:
        document.pop('_id', None)  # Remove the auto-generated field
        job_uuid = document.get('job_uuid')
        score = classifier_scores_map.get(job_uuid)
        if score is not None:
            # Add classifier score as the first field
            print(f"Job UUID: {job_uuid}, Score: {score}")
            document_with_score = {'classifier_score': score}
            document_with_score.update(document)
            documents[documents.index(document)] = document_with_score

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
  
@router.get("/image/list-image-metadata-by-dataset", tags = ["deprecated2"], response_class=PrettyJSONResponse)
def get_images_metadata(
    request: Request,
    dataset: str = None,
    prompt_generation_policy: Optional[str] = None,  # Optional query parameter for prompt_generation_policy
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

        # Convert threshold_time to a string in ISO format
        threshold_time_str = threshold_time.isoformat(timespec='milliseconds') 
    else:
        threshold_time_str = None

    # Construct the initial query
    query = {
        '$or': [
            {'task_type': 'image_generation_sd_1_5'},
            {'task_type': 'inpainting_sd_1_5'},
            {'task_type': 'image_generation_kandinsky'},
            {'task_type': 'inpainting_kandinsky'},
            {'task_type': 'img2img_generation_kandinsky'}
        ],
        'task_input_dict.dataset': dataset
    }

    # Optionally add prompt_generation_policy to the query if provided
    if prompt_generation_policy:
        query['prompt_generation_data.prompt_generation_policy'] = prompt_generation_policy

    # Update the query based on provided start_date, end_date, and threshold_time_str
    if start_date and end_date:
        query['task_creation_time'] = {'$gte': start_date, '$lte': end_date}
    elif start_date:
        query['task_creation_time'] = {'$gte': start_date}
    elif end_date:
        query['task_creation_time'] = {'$lte': end_date}
    elif threshold_time_str:
        query['task_creation_time'] = {'$gte': threshold_time_str}

    # Decide the sort order based on the 'order' parameter
    sort_order = -1 if order == "desc" else 1

    # Query the completed_jobs_collection using the constructed query
    jobs = request.app.completed_jobs_collection.find(query).sort('task_creation_time', sort_order).skip(offset).limit(limit)

    # Collect the metadata for the images that match the query
    images_metadata = []
    for job in jobs:
        image_meta_data = {
            'dataset': job['task_input_dict'].get('dataset'),
            'task_type': job.get('task_type'),
            'image_path': job['task_output_file_dict'].get('output_file_path'),
            'image_hash': job['task_output_file_dict'].get('output_file_hash'),
            'prompt_generation_policy': job['prompt_generation_data'].get('prompt_generation_policy', prompt_generation_policy)  # Include the policy if present
        }
        # If prompt_generation_policy is not a filter or if it matches the job's policy, append the metadata
        if not prompt_generation_policy or image_meta_data['prompt_generation_policy'] == prompt_generation_policy:
            images_metadata.append(image_meta_data)

    # Return the metadata for the filtered images
    return images_metadata

@router.get("/image/get_random_image_with_time", response_class=PrettyJSONResponse)
def get_random_image_with_time(
    request: Request,
    dataset: str = Query(...),
    time_interval: int = Query(..., description="Time interval in minutes or hours"),
    time_unit: str = Query("minutes", description="Time unit, either 'minutes' or 'hours"),
    size: int = Query(1, description="Number of images to return"),  # Existing size parameter
    prompt_generation_policy: Optional[str] = None  # Added new parameter
):
    # Calculate the time threshold based on the current time and the specified interval
    current_time = datetime.utcnow()
    if time_unit == "minutes":
        threshold_time = current_time - timedelta(minutes=time_interval)
    elif time_unit == "hours":
        threshold_time = current_time - timedelta(hours=time_interval)
    else:
        raise HTTPException(status_code=400, detail="Invalid time unit. Use 'minutes' or 'hours'.")

    # Update the match query to include prompt_generation_policy if provided
    match_query = {
        "task_input_dict.dataset": dataset,
        "task_creation_time": {"$gte": threshold_time.strftime("%Y-%m-%dT%H:%M:%S")}
    }
    if prompt_generation_policy:
        match_query["prompt_generation_data.prompt_generation_policy"] = prompt_generation_policy

    # Use $match to filter documents based on dataset, creation time, and prompt_generation_policy
    documents = request.app.completed_jobs_collection.aggregate([
        {"$match": match_query},
        {"$sample": {"size": size}}
    ])

    # Convert cursor type to list
    documents = list(documents)

    # Ensure the list isn't empty (this is just a safety check)
    if not documents:
        print(f"No images found for the given dataset within the last {time_interval} {time_unit}")

    # Remove the auto-generated _id field from each document
    for document in documents:
        document.pop('_id', None)

    return {"images": documents}  # Return the list of images




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
        raise HTTPException(
            status_code=500,
            detail="Image with this path doesn't exist") 

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

@router.get("/list-prompt-generation-policies")
def list_prompt_generation_policies():
    return ["greedy-substitution-search-v1", 
            "quincy-greedy-prompt-search-v1", 
            "distilgpt2_han-v1", 
            "top-k", 
            "proportional-sampling-top-k", 
            "independent_approx_v1", 
            "independent-approx-v1-top-k",
            "independent-approx-substitution-search-v1",
            "proportional_sampling",
            "gradient_descent_optimization",
            "variant_generation",
            "gaussian-top-k-sphere-sampling",
            "top-k-sphere-sampling",
            "rapidly_exploring_tree_search"]




#new apis

@router.get("/image/get-random-image-v1",
            response_model=StandardSuccessResponseV1[Task],  
            description="Get a random image from a dataset",
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([404, 500]))
def get_random_image_v1(request: Request, dataset: str = Query(...)):
    response_handler = ApiResponseHandlerV1(request)

    try:
        documents = request.app.completed_jobs_collection.aggregate([
            {"$match": {"task_input_dict.dataset": dataset}},
            {"$sample": {"size": 1}}
        ])

        documents = list(documents)

        if not documents:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="No images found in dataset",
                http_status_code=404,
            )

        documents[0].pop('_id', None)

        return response_handler.create_success_response_v1(
            response_data= documents[0],
            http_status_code=200,
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )


@router.get("/image/get-image-details-v1",
            response_model=StandardSuccessResponseV1[Task],  
            description="Get details of an image",
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([404, 500]))
def get_image_details_v1(request: Request, image_path: str = Query(...)):
    response_handler = ApiResponseHandlerV1(request)

    try:
        document = request.app.completed_jobs_collection.find_one(
            {"task_output_file_dict.output_file_path": image_path}
        )

        if document is None:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="Image details not found",
                http_status_code=404,
            )

        document.pop('_id', None)

        return response_handler.create_success_response_v1(
            response_data=document,
            http_status_code=200,
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )


@router.get("/image/get-random-image-list-v1",
            response_model=StandardSuccessResponseV1[List[Task]], 
            description="Get a list of random images from a dataset",
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([404, 500]))
def get_random_image_list_v1(request: Request, dataset: str = Query(...), size: int = Query(1)):  
    response_handler = ApiResponseHandlerV1(request)

    try:
        distinct_documents = []
        tried_ids = set()

        while len(distinct_documents) < size:
            # Build filter for aggregation
            filter = [
                {"$match": {"task_input_dict.dataset": dataset, "_id": {"$nin": list(tried_ids)}}},
                {"$sample": {"size": size - len(distinct_documents)}}
            ]

            if dataset == "any":
                filter = [
                    {"$match": {"_id": {"$nin": list(tried_ids)}}},
                    {"$sample": {"size": size - len(distinct_documents)}}
                ]

            documents = request.app.completed_jobs_collection.aggregate(filter)
            documents = list(documents)
            distinct_documents.extend(documents)
            tried_ids.update([doc["_id"] for doc in documents])

            seen = set()
            distinct_documents = [doc for doc in distinct_documents if doc["_id"] not in seen and not seen.add(doc["_id"])]

        for doc in distinct_documents:
            doc.pop('_id', None)

        if not distinct_documents:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="No images found in dataset",
                http_status_code=404,
            )

        return response_handler.create_success_response_v1(
            response_data= distinct_documents,
            http_status_code=200,
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )


@router.get("/image/get-random-previously-ranked-image-list-v1",
            response_model=StandardSuccessResponseV1[List[Task]],  # Adjust the response model as needed
            description="Get a list of random, previously ranked images from a dataset with specific filters",
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([404, 500]))
def get_random_previously_ranked_image_list_v1(
    request: Request, 
    dataset: str = Query(...), 
    size: int = Query(1),
    prompt_generation_policy: Optional[str] = None,
    start_date: str = None,
    end_date: str = None,
    time_interval: int = Query(None, description="Time interval in minutes or hours"),
    time_unit: str = Query("minutes", description="Time unit, either 'minutes' or 'hours")
):
    response_handler = ApiResponseHandlerV1(request)

    try:
        distinct_documents = []
        tried_ids = set()

        match_query = {"task_input_dict.dataset": dataset, "_id": {"$nin": list(tried_ids)}}
        if prompt_generation_policy:
            match_query["prompt_generation_data.prompt_generation_policy"] = prompt_generation_policy

        # Apply the date/time filters
        if start_date or end_date:
            match_query["task_creation_time"] = {}
            if start_date:
                match_query["task_creation_time"]["$gte"] = start_date
            if end_date:
                match_query["task_creation_time"]["$lte"] = end_date

        # Process time_interval and time_unit
        if time_interval and time_unit:
            current_time = datetime.utcnow()
            if time_unit == "minutes":
                threshold_time = current_time - timedelta(minutes=time_interval)
            elif time_unit == "hours":
                threshold_time = current_time - timedelta(hours=time_interval)
            threshold_time_str = threshold_time.isoformat(timespec='milliseconds')
            match_query["task_creation_time"]["$gte"] = threshold_time_str

        while len(distinct_documents) < size:
            documents = request.app.completed_jobs_collection.aggregate([
                {"$match": match_query},
                {"$sample": {"size": size - len(distinct_documents)}}
            ])

            documents = list(documents)
            tried_ids.update([doc["_id"] for doc in documents])

            # Use existing logic to filter previously ranked images
            for doc in documents:
                count = get_image_rank_use_count(request, doc["task_output_file_dict"]["output_file_hash"])
                if count > 0:
                    distinct_documents.append(doc)

            seen = set()
            distinct_documents = [doc for doc in distinct_documents if doc["_id"] not in seen and not seen.add(doc["_id"])]

        for doc in distinct_documents:
            doc.pop('_id', None)  # Remove the auto-generated field

        return response_handler.create_success_response_v1(
            response_data=distinct_documents,
            http_status_code=200,
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )


@router.get("/image/get-random-image-by-date-range-v1",
            response_model=StandardSuccessResponseV1[List[Task]],  # Adjust response model as necessary
            description="Get a random image or a list of images from a dataset within a specific date range",
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([404, 500]))
def get_random_image_date_range_v1(
    request: Request,
    dataset: str = Query(None, description="The dataset from which to fetch the images"),
    start_date: str = Query(None, description="Start date for filtering images"),
    end_date: str = Query(None, description="End date for filtering images"),
    size: int = Query(None, description="Number of images to return"),
    prompt_generation_policy: Optional[str] = Query(None, description="Optional prompt generation policy")
):
    response_handler = ApiResponseHandlerV1(request)

    try:
        query = {'task_input_dict.dataset': dataset}
        if start_date and end_date:
            query['task_creation_time'] = {'$gte': start_date, '$lte': end_date}
        elif start_date:
            query['task_creation_time'] = {'$gte': start_date}
        elif end_date:
            query['task_creation_time'] = {'$lte': end_date}

        # Include prompt_generation_policy in the query if provided
        if prompt_generation_policy:
            query['prompt_generation_data.prompt_generation_policy'] = prompt_generation_policy

        aggregation_pipeline = [{"$match": query}]
        if size:
            aggregation_pipeline.append({"$sample": {"size": size}})

        documents = request.app.completed_jobs_collection.aggregate(aggregation_pipeline)
        documents = list(documents)

        for document in documents:
            document.pop('_id', None)  # Remove the auto-generated field

        if not documents:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="No images found within the specified date range or criteria.",
                http_status_code=404,
            )

        return response_handler.create_success_response_v1(
            response_data= documents,
            http_status_code=200,
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )


@router.get("/image/get-random-image-with-time-v1",
            response_model=StandardSuccessResponseV1[List[Task]],  # Adjust response model as necessary
            description="Get a random image or a list of images from a dataset within a specific time range",
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([400, 404, 500]))
def get_random_image_with_time_v1(
    request: Request,
    dataset: str = Query(..., description="The dataset from which to fetch the images"),
    time_interval: int = Query(..., description="Time interval in minutes or hours"),
    time_unit: str = Query("minutes", description="Time unit, either 'minutes' or 'hours"),
    size: int = Query(1, description="Number of images to return"),
    prompt_generation_policy: Optional[str] = Query(None, description="Optional prompt generation policy")
):
    response_handler = ApiResponseHandlerV1(request)

    try:
        # Calculate the time threshold based on the current time and the specified interval
        current_time = datetime.utcnow()
        if time_unit == "minutes":
            threshold_time = current_time - timedelta(minutes=time_interval)
        elif time_unit == "hours":
            threshold_time = current_time - timedelta(hours=time_interval)
        else:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string="Invalid time unit. Use 'minutes' or 'hours'.",
                http_status_code=400,
            )

        # Update the match query to include prompt_generation_policy if provided
        match_query = {
            "task_input_dict.dataset": dataset,
            "task_creation_time": {"$gte": threshold_time}
        }
        if prompt_generation_policy:
            match_query["prompt_generation_data.prompt_generation_policy"] = prompt_generation_policy

        # Use $match to filter documents based on dataset, creation time, and prompt_generation_policy
        documents = request.app.completed_jobs_collection.aggregate([
            {"$match": match_query},
            {"$sample": {"size": size}}
        ])

        documents = list(documents)  # Convert cursor type to list

        if not documents:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string=f"No images found for the given dataset within the last {time_interval} {time_unit}",
                http_status_code=404,
            )

        # Remove the auto-generated _id field from each document
        for document in documents:
            document.pop('_id', None)

        return response_handler.create_success_response_v1(
            response_data=documents,
            http_status_code=200,
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )

@router.get("/image/list-image-metadata-by-dataset-v1",
            response_model=StandardSuccessResponseV1[List[UUIDImageMetadata]], 
            description="List image metadata by dataset with optional filtering",
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([400, 404, 500]))
def get_images_metadata(
    request: Request,
    dataset: str = Query(..., description="The dataset from which to fetch the images"),
    prompt_generation_policy: Optional[str] = Query(None, description="Optional prompt generation policy"),
    limit: int = Query(20, description="Limit on the number of results returned"),
    offset: int = Query(0, description="Offset for the results to be returned"),
    start_date: str = Query(None, description="Start date for filtering results"),
    end_date: str = Query(None, description="End date for filtering results"),
    order: str = Query("desc", description="Order in which the data should be returned. 'asc' for oldest first, 'desc' for newest first"),
    time_interval: Optional[int] = Query(None, description="Time interval in minutes or hours"),
    time_unit: str = Query("minutes", description="Time unit, either 'minutes' or 'hours")
):
    response_handler = ApiResponseHandlerV1(request)

    try:
            # Calculate the time threshold based on the current time and the specified interval
        if time_interval is not None:
            current_time = datetime.utcnow()
            if time_unit == "minutes":
                threshold_time = current_time - timedelta(minutes=time_interval)
            elif time_unit == "hours":
                threshold_time = current_time - timedelta(hours=time_interval)
            else:
                raise HTTPException(status_code=400, detail="Invalid time unit. Use 'minutes' or 'hours'.")

            # Convert threshold_time to a string in ISO format
            threshold_time_str = threshold_time.isoformat(timespec='milliseconds') 
        else:
            threshold_time_str = None

        # Construct the initial query
        query = {
            '$or': [
                {'task_type': 'image_generation_sd_1_5'},
                {'task_type': 'inpainting_sd_1_5'},
                {'task_type': 'image_generation_kandinsky'},
                {'task_type': 'inpainting_kandinsky'},
                {'task_type': 'img2img_generation_kandinsky'}
            ],
            'task_input_dict.dataset': dataset
        }

        # Optionally add prompt_generation_policy to the query if provided
        if prompt_generation_policy:
            query['prompt_generation_data.prompt_generation_policy'] = prompt_generation_policy

        # Update the query based on provided start_date, end_date, and threshold_time_str
        if start_date and end_date:
            query['task_creation_time'] = {'$gte': start_date, '$lte': end_date}
        elif start_date:
            query['task_creation_time'] = {'$gte': start_date}
        elif end_date:
            query['task_creation_time'] = {'$lte': end_date}
        elif threshold_time_str:
            query['task_creation_time'] = {'$gte': threshold_time_str}

        # Decide the sort order based on the 'order' parameter
        sort_order = -1 if order == "desc" else 1

        # Query the completed_jobs_collection using the constructed query
        jobs = request.app.completed_jobs_collection.find(query).sort('task_creation_time', sort_order).skip(offset).limit(limit)

        # Collect the metadata for the images that match the query
        images_metadata = []
        for job in jobs:
            image_meta_data = {
                'uuid': job.get('uuid'),
                'dataset': job['task_input_dict'].get('dataset'),
                'task_type': job.get('task_type'),
                'image_path': job['task_output_file_dict'].get('output_file_path'),
                'image_hash': job['task_output_file_dict'].get('output_file_hash'),
                'prompt_generation_policy': job['prompt_generation_data'].get('prompt_generation_policy', prompt_generation_policy)  # Include the policy if present
            }
            # If prompt_generation_policy is not a filter or if it matches the job's policy, append the metadata
            if not prompt_generation_policy or image_meta_data['prompt_generation_policy'] == prompt_generation_policy:
                images_metadata.append(image_meta_data)

        return response_handler.create_success_response_v1(
            response_data=images_metadata,
            http_status_code=200,
            )
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )




bucket_name = "datasets"
base_folder = "patches"

@router.post("/upload-image-patches",
             status_code=201, 
             response_model=StandardSuccessResponseV1[UrlResponse],
             responses=ApiResponseHandlerV1.listErrors([400, 422, 500]),
             description="Upload Image on minio")
async def upload_image_v1(request: Request, 
                          file: UploadFile = File(...), 
                          check_size: bool = Query(True, description="Check if image is 512x512")):
    response_handler = ApiResponseHandlerV1(request)
    # Initialize MinIO client
    minio_client = cmd.get_minio_client(minio_access_key="v048BpXpWrsVIHUfdAix", minio_secret_key="4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu")
    
    # Extract the file extension
    _, file_extension = os.path.splitext(file.filename)
    # Ensure the extension is in a consistent format (e.g., lowercase) and validate it if necessary
    file_extension = file_extension.lower()
    # Validate or adjust the extension (optional, based on your requirements)
    # Find or create the next available folder and get the next image index
    next_folder, next_index = find_or_create_next_folder_and_index(minio_client, bucket_name, base_folder)

    # Construct the file path, preserving the original file extension
    file_name = f"{next_index:06}{file_extension}"  # Use the extracted file extension
    file_path = f"{next_folder}/{file_name}"

    try:
        await file.seek(0)  # Go to the start of the file
        content = await file.read()  # Read file content into bytes
        if check_size:  # Perform size check if check_size is True
            # Check if the image is 512x512
            image = Image.open(io.BytesIO(content))
            if image.size != (512, 512):
                return response_handler.create_error_response_v1(
                    error_code=ErrorCode.INVALID_PARAMS, 
                    error_string="Image must be 512x512 pixels",
                    http_status_code=422,
                )
        content_stream = io.BytesIO(content)
        # Upload the file content
        cmd.upload_data(minio_client, bucket_name, file_path, content_stream)
        full_file_path = f"{bucket_name}/{file_path}"
        return response_handler.create_success_response_v1(
            response_data=full_file_path, 
            http_status_code=201)
    except Exception as e:
        print(f"Exception occurred: {e}") 
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string="Internal server error",
            http_status_code=500, 
        )
    
@router.get("/static/images/get-image-by-path/{file_path:path}",
            description="Get image by file path",
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([404,422, 500]))
async def get_image_data_by_filepath_2(request: Request, file_path: str):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        bucket_name, file_path = separate_bucket_and_file_path(file_path)  
        file_path = file_path.replace("\\", "/")
        image_data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, file_path)

        if image_data is None:
            # Utilize the response handler for standardized error response
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="Image with this path doesn't exist",
                http_status_code=404
            )

        # Directly return the image data with the appropriate media type
        content = image_data.read()
        return Response(content=content, media_type="image/jpeg")
    except Exception as e:
        # Utilize the response handler for other exceptions
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )    


@router.get("/static/images/get-image-by-job-uuid-v1/{job_uuid}",
            description="Fetch image by job UUID",
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([404,422, 500]))
async def get_image_by_job_uuid(request: Request, job_uuid: str):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        job = request.app.completed_jobs_collection.find_one({"uuid": job_uuid})
        if not job:
            # Job not found
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="Job not found",
                http_status_code=404
            )

        output_file_path = job.get("task_output_file_dict", {}).get("output_file_path")
        if not output_file_path:
            # Output file path missing
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="Image with this path doesn't exist",
                http_status_code=404
            )

        bucket_name, file_path = separate_bucket_and_file_path(output_file_path) 
        file_path = file_path.replace("\\", "/")
        image_data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, file_path)

        if image_data is None:
            # Image not found in MinIO
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="Image with this path doesn't exist",
                http_status_code=404
            )

        content = image_data.read()
        original_filename = os.path.basename(output_file_path)
        headers = {"Content-Disposition": f"attachment; filename={original_filename}"}
        return Response(content=content, media_type="image/jpeg", headers=headers)
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )        