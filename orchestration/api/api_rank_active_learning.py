from fastapi import Request, HTTPException, APIRouter, Response, Query, status
from datetime import datetime, timedelta
import pymongo
from utility.minio import cmd
from orchestration.api.mongo_schema.active_learning_schemas import RankSelection, ListResponseRankSelection, ResponseRankSelection
from .api_utils import ApiResponseHandlerV1, ErrorCode, StandardSuccessResponseV1, StandardErrorResponseV1, WasPresentResponse, CountResponse
from orchestration.api.mongo_schema.active_learning_schemas import  RankActiveLearningPair, ListRankActiveLearningPair
import os
from fastapi.responses import JSONResponse
from pymongo.collection import Collection
from datetime import datetime, timezone
from typing import List
from io import BytesIO
from bson import ObjectId
from typing import Optional
import json
from collections import OrderedDict

router = APIRouter()


@router.post("/rank-active-learning-queue/add-image-pair",
             description="Adds a new image pair to the rank active ranking queue",
             status_code=200,
             response_model=StandardSuccessResponseV1[RankActiveLearningPair],
             tags=["Rank Active Learning"],  
             responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
def get_job_details(request: Request, job_uuid_1: str = Query(...), job_uuid_2: str = Query(...), rank_active_learning_policy_id: int = Query(...), rank_model_id: int = Query(...), metadata: str = Query(None), generation_string: str = Query(None) ):
    api_response_handler = ApiResponseHandlerV1(request)

    def extract_job_details(job_uuid, suffix):

        job = request.app.completed_jobs_collection.find_one({"uuid": job_uuid})
        if not job:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string=f"Job {job_uuid} not found",
                http_status_code=404
            )


        output_file_path = job["task_output_file_dict"]["output_file_path"]
        task_creation_time = job["task_creation_time"]
        path_parts = output_file_path.split('/')
        if len(path_parts) < 4:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string=f"Invalid output file path format",
                http_status_code=422
            )

        return {
            f"job_uuid_{suffix}": job_uuid,
            f"file_name_{suffix}": path_parts[-1],
            f"image_path_{suffix}": output_file_path,
            f"image_hash_{suffix}": job["task_output_file_dict"]["output_file_hash"],
            f"job_creation_time_{suffix}": task_creation_time,
        }

    job_details_1 = extract_job_details(job_uuid_1, "1")
    job_details_2 = extract_job_details(job_uuid_2, "2")

    if not job_details_1 or not job_details_2:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.ELEMENT_NOT_FOUND, 
            error_string="Job details not found",
            http_status_code=404
        )


    rank = request.app.rank_model_models_collection.find_one(
        {"rank_model_id": rank_model_id}
    )

    if not rank:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.ELEMENT_NOT_FOUND,
            error_string=f"Rank with ID {rank} not found",
            http_status_code=404
        )


    policy = request.app.rank_active_learning_policies_collection.find_one(
        {"rank_active_learning_policy_id": rank_active_learning_policy_id}
    )
    if not policy:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.ELEMENT_NOT_FOUND,
            error_string=f"Active learning policy with ID {rank_active_learning_policy_id} not found",
            http_status_code=404
        )

    combined_job_details = {
        "rank_model_id": rank_model_id,
        "rank_active_learning_policy_id": rank_active_learning_policy_id,
        "dataset_name": job_details_1['image_path_1'].split('/')[1],  # Extract dataset name
        "metadata": metadata,
        "generation_string": generation_string,
        "creation_date": datetime.utcnow().isoformat(),  # UTC time
        "images_data": [job_details_1, job_details_2]
    }

    creation_date_1 = datetime.fromisoformat(job_details_1["job_creation_time_1"]).strftime("%Y-%m-%d")
    creation_date_2 = datetime.fromisoformat(job_details_2["job_creation_time_2"]).strftime("%Y-%m-%d")

    json_data = json.dumps([combined_job_details], indent=4).encode('utf-8')  # Note the list brackets around combined_job_details
    data = BytesIO(json_data)

    # Define the path for the JSON file
    base_file_name_1 = job_details_1['file_name_1'].split('.')[0]
    base_file_name_2 = job_details_2['file_name_2'].split('.')[0]
    json_file_name = f"{creation_date_1}_{base_file_name_1}_and_{creation_date_2}_{base_file_name_2}.json"
    full_path = f"environmental/rank-active-learning-queue/{policy['rank_active_learning_policy']}/{json_file_name}"

    cmd.upload_data(request.app.minio_client, "datasets", full_path, data)


    mongo_combined_job_details = {"file_name": json_file_name, **combined_job_details}
    request.app.rank_active_learning_pairs_collection.insert_one(mongo_combined_job_details)

    mongo_combined_job_details.pop('_id', None)


    return api_response_handler.create_success_response_v1(
            response_data=mongo_combined_job_details,
            http_status_code=200
        )



@router.get("/rank-active-learning-queue/list-image-pairs",
            description="Lists all the rank active learning image pairs",
            response_model=StandardSuccessResponseV1[ListRankActiveLearningPair],
            status_code=200,
            tags=["Rank Active Learning"],  
            responses=ApiResponseHandlerV1.listErrors([400, 422]))
def get_image_rank_scores_by_model_id(request: Request):
    api_response_handler = ApiResponseHandlerV1(request)
    
    # check if exist
    items = list(request.app.rank_active_learning_pairs_collection.find({}))
    
    score_data = []
    for item in items:
        # remove the auto generated '_id' field
        item.pop('_id', None)
        score_data.append(item)
    
    # Return a standardized success response with the score data
    return api_response_handler.create_success_response_v1(
        response_data={"pairs": score_data},
        http_status_code=200
    )

@router.delete("/rank-active-learning-queue/delete-image-pair",
               description="Deletes an image pair from the rank active learning queue",
               response_model=StandardSuccessResponseV1[WasPresentResponse],
               status_code=200,
               tags=["Rank Active Learning"],
               responses=ApiResponseHandlerV1.listErrors([422, 500]))
async def delete_image_rank_data_point(request: Request, file_name: str = Query(..., description="The file name of the data point to delete")):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    
    try:
        # Attempt to delete the document with the specified file_name
        delete_result = request.app.rank_active_learning_pairs_collection.delete_one({"file_name": file_name})
        
        if delete_result.deleted_count == 0:
            # If no documents were deleted, it means the file_name did not exist
            return api_response_handler.create_success_response_v1(
            response_data={"wasPresent": False}, 
            http_status_code=200
        )

        # If the document was deleted successfully, return a success message
        return api_response_handler.create_success_response_v1(
            response_data={"wasPresent": True}, 
            http_status_code=200
        )
    
    except Exception as e:
        # Handle exceptions that may occur during database operation
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )
    


@router.get("/rank-active-learning-queue/count-image-pairs", 
            response_model=StandardSuccessResponseV1[CountResponse],
            status_code=200,
            tags=["Rank Active Learning"],
            description="Counts how many image pairs in the rank active learning queue for specified policy and model.",
            responses=ApiResponseHandlerV1.listErrors([500]))
async def count_queue_pairs(request: Request, 
                            policy_id: int = Query(None, description="Filter by the rank active learning policy ID"), 
                            rank_model_id: int = Query(None, description="Filter by the rank model ID")):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        # Build the query based on the provided parameters
        query = {}
        if policy_id is not None:
            query['rank_active_learning_policy_id'] = policy_id
        if rank_model_id is not None:
            query['rank_model_id'] = rank_model_id

        # Count documents in the rank_active_learning_pairs_collection with the constructed query
        count = request.app.rank_active_learning_pairs_collection.count_documents(query)

        return api_response_handler.create_success_response_v1(
            response_data={"count": count},
            http_status_code=200  
        )
    
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        ) 



@router.get("/rank-active-learning-queue/get-random-image-pair", 
            description="list random rank active learning datapoints",
            response_model=StandardSuccessResponseV1[ListRankActiveLearningPair],
            status_code=200,
            tags=["Rank Active Learning"],  
            responses=ApiResponseHandlerV1.listErrors([400, 422]))
async def random_queue_pair(request: Request, rank_model_id : Optional[int] = None, size: int = 1, dataset: Optional[str] = None, rank_active_learning_policy_id: Optional[int] = None):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        # Define the aggregation pipeline
        pipeline = []

        # Filters based on dataset and rank_active_learning_policy
        match_filter = {}
        if rank_model_id:
            match_filter["rank_model_id"] = rank_model_id
        if dataset:
            match_filter["dataset"] = dataset
        if rank_active_learning_policy_id:
            match_filter["rank_active_learning_policy_id"] = rank_active_learning_policy_id

        if match_filter:
            pipeline.append({"$match": match_filter})

        # Add the random sampling stage to the pipeline
        pipeline.append({"$sample": {"size": size}})

        # Use MongoDB's aggregation framework to randomly select documents
        random_pairs_cursor = request.app.rank_active_learning_pairs_collection.aggregate(pipeline)

        # Convert the cursor to a list of dictionaries
        random_pairs = []
        for pair in random_pairs_cursor:
            pair['_id'] = str(pair['_id'])  # Convert _id ObjectId to string
            random_pairs.append(pair)

        return api_response_handler.create_success_response_v1(
            response_data={"pairs": random_pairs},
            http_status_code=200
        )

    except Exception as e:
        # Handle exceptions that may occur during database operation
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )




@router.post("/rank-training/add-ranking-data-point", 
             status_code=201,
             tags = ['Rank Active Learning'],
             response_model=StandardSuccessResponseV1[ResponseRankSelection],
             responses=ApiResponseHandlerV1.listErrors([404,422, 500]))
async def add_datapoints(request: Request, selection: RankSelection):
    api_handler = await ApiResponseHandlerV1.createInstance(request)
    
    try:

        rank = request.app.rank_model_models_collection.find_one(
        {"rank_model_id": selection.rank_model_id}
        )

        print(type(selection.rank_model_id))
        if not rank:
            return api_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string=f"Rank with ID {rank} not found",
                http_status_code=404
            )
        

        policy = request.app.rank_active_learning_policies_collection.find_one(
        {"rank_active_learning_policy_id": selection.rank_active_learning_policy_id}
    )
        

        if not policy:
            return api_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string=f"Active learning policy with ID {selection.rank_active_learning_policy_id} not found",
                http_status_code=404
            )


        current_time = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S')
        file_name = f"{current_time}-{selection.username}.json"
        dataset = selection.image_1_metadata.file_path.split('/')[1]
        rank_model_string = rank.get("rank_model_string", None)
        rank_active_learning = policy.get("rank_active_learning_policy", None)

        dict_data = selection.to_dict()

        # Prepare ordered data for MongoDB insertion
        mongo_data = OrderedDict([
            ("_id", ObjectId()),  # Generate new ObjectId
            ("file_name", file_name),
            ("dataset", dataset),
            *dict_data.items(), # Unpack the rest of dict_data
            ("datetime", current_time)
        ])

        # Insert the ordered data into MongoDB
        request.app.ranking_datapoints_collection.insert_one(mongo_data)

        # Prepare data for MinIO upload (excluding the '_id' field)
        minio_data = mongo_data.copy()
        minio_data.pop("_id")
        minio_data.pop("file_name")
        path = f"data/rank/{rank_model_string}"
        full_path = os.path.join("environmental", path, file_name)
        json_data = json.dumps(minio_data, indent=4).encode('utf-8')
        data = BytesIO(json_data)

        # Upload data to MinIO
        cmd.upload_data(request.app.minio_client, "datasets", full_path, data)

 
        mongo_data.pop("_id")
        # Return a success response
        return api_handler.create_success_response_v1(
            response_data=mongo_data,
            http_status_code=201
        )

    except Exception as e:

        return api_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )

@router.get("/rank-training/list-ranking-datapoints",
            description="Lists all the ranking datapoints",
            response_model=StandardSuccessResponseV1[ListResponseRankSelection],
            status_code=200,
            tags=["Rank Active Learning"],  
            responses=ApiResponseHandlerV1.listErrors([400, 422]))
def list_ranking_datapoints(request: Request):
    api_response_handler = ApiResponseHandlerV1(request)
    
    # check if exist
    items = list(request.app.ranking_datapoints_collection.find({}))
    
    score_data = []
    for item in items:
        # remove the auto generated '_id' field
        item.pop('_id', None)
        score_data.append(item)
    
    # Return a standardized success response with the score data
    return api_response_handler.create_success_response_v1(
        response_data={"datapoints": score_data},
        http_status_code=200
    )

@router.get("/rank-training/sort-ranking-data-by-date", 
            description="Sort rank data by date",
            tags=["Rank Active Learning"],
            response_model=StandardSuccessResponseV1[ListResponseRankSelection],  
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
async def sort_ranking_data_by_date_v2(
    request: Request,
    dataset: str = Query(..., description="Dataset to filter by"),
    start_date: Optional[str] = Query(None, description="Start date (inclusive) in YYYY-MM-DD format"),
    rank_model_id: int = Query(None, description="Rank model ID to filter by"),
    end_date: Optional[str] = Query(None, description="End date (inclusive) in YYYY-MM-DD format"),
    skip: int = Query(0, alias="offset"),
    limit: int = Query(10, alias="limit"),
    order: str = Query("desc", regex="^(desc|asc)$")
):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        # Convert start_date and end_date strings to datetime objects, if provided
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else None

        # Build the query filter based on dataset and optional dates
        query_filter = {"dataset": dataset, "rank_model_id": rank_model_id}
        if start_date_obj or end_date_obj:
            date_filter = {}
            if start_date_obj:
                date_filter["$gte"] = start_date_obj
            if end_date_obj:
                date_filter["$lte"] = end_date_obj
            query_filter["datetime"] = date_filter 

        # Determine the sort order
        sort_order = pymongo.DESCENDING if order == "desc" else pymongo.ASCENDING

        # Fetch and sort data from MongoDB with pagination
        cursor = request.app.ranking_datapoints_collection.find(query_filter).sort(
            "datetime", sort_order  
        ).skip(skip).limit(limit)

        # Convert cursor to list of dictionaries
        ranking_data = [doc for doc in cursor]
        for doc in ranking_data:
            doc['_id'] = str(doc['_id'])  # Convert ObjectId to string for JSON serialization
        
        return response_handler.create_success_response_v1(
            response_data={"datapoints": ranking_data}, 
            http_status_code=200
        )
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=f"Internal Server Error: {str(e)}",
            http_status_code=500
        )      
    

@router.get("/rank-training/count-ranking-data-points", 
            description="Count ranking data points based on specific rank models or policies",
            tags=["Rank Active Learning"],
            response_model=StandardSuccessResponseV1[CountResponse],  
            responses=ApiResponseHandlerV1.listErrors([500]))
async def count_ranking_data(request: Request, 
                             policy_id: int = Query(None, description="Filter by the rank active learning policy ID"), 
                             rank_model_id: int = Query(None, description="Filter by the rank model ID")):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        # Build the query based on the provided parameters
        query = {}
        if policy_id is not None:
            query['rank_active_learning_policy_id'] = policy_id
        if rank_model_id is not None:
            query['rank_model_id'] = rank_model_id

        # Get the count of documents in the ranking_datapoints_collection based on the constructed query
        count = request.app.ranking_datapoints_collection.count_documents(query)

        # Return the count with a success response
        return response_handler.create_success_response_v1(
            response_data={"count": count}, 
            http_status_code=200,
        )
    except Exception as e:
        # Handle exceptions and return an error response
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )