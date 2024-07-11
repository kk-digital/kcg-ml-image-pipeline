import os.path

from fastapi import Request, HTTPException, APIRouter, Response, Query
from orchestration.api.mongo_schemas import SequentialID
from utility.minio import cmd
import json
from datetime import datetime
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from .api_utils import PrettyJSONResponse, ApiResponseHandlerV1, StandardSuccessResponseV1, ErrorCode, WasPresentResponse, DatasetResponse, SeqIdResponse, SeqIdDatasetResponse
from .mongo_schemas import FlaggedDataUpdate, RankingModel, Dataset, ListDataset
from orchestration.api.mongo_schema.selection_schemas import ListRelevanceSelection, ListRankingSelection
from pymongo import ReturnDocument
router = APIRouter()


@router.delete("/dataset/clear-sequential-id", 
               tags = ['deprecated3'],
               description="changed with /datasets/clear-all-sequential-id ")
def clear_dataset_sequential_id_jobs(request: Request):
    request.app.dataset_sequential_id_collection.delete_many({})

    return True


@router.get("/dataset/list", tags = ['deprecated3'], description= "changed with /datasets/list-datasets " )
def get_datasets(request: Request):
    objects = cmd.get_list_of_objects(request.app.minio_client, "datasets")

    return objects


@router.get("/dataset/sequential-id/{dataset}",tags = ['deprecated3'], description= "changed with /datasets/get-sequential-ids" )
def get_sequential_id(request: Request, dataset: str, limit: int = 1):
    sequential_id_arr = []

    # find
    sequential_id = request.app.dataset_sequential_id_collection.find_one({"dataset_name": dataset})
    if sequential_id is None:
        # create one
        new_sequential_id = SequentialID(dataset)

        # get the sequential id arr
        for i in range(limit):
            sequential_id_arr.append(new_sequential_id.get_sequential_id())

        # add to collection
        request.app.dataset_sequential_id_collection.insert_one(new_sequential_id.to_dict())

        return sequential_id_arr

    # if found
    found_sequential_id = SequentialID(sequential_id["dataset_name"], sequential_id["subfolder_count"],
                                       sequential_id["file_count"])
    # get the sequential id arr
    for i in range(limit):
        sequential_id_arr.append(found_sequential_id.get_sequential_id())

    new_values = {"$set": found_sequential_id.to_dict()}

    # # update existing sequential id
    request.app.dataset_sequential_id_collection.update_one({"dataset_name": dataset}, new_values)

    return sequential_id_arr

@router.delete("/dataset/delete-sequential-id", 
               tags = ['deprecated3'],
               description="delete the sequential id of a dataset")
def delete_sequential_id(request: Request, dataset_name: str):

    res= request.app.dataset_sequential_id_collection.delete_one({"dataset_name": dataset_name})

    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="No sequential id was found for that dataset name")

    return True

@router.delete("/dataset/delete-external-sequential-id", 
               tags = ['deprecated3'],
               description="delete the sequential id of an external dataset")
def delete_sequential_id(request: Request, bucket_name: str, dataset_name: str):

    res= request.app.external_dataset_sequential_id.delete_one({"bucket": bucket_name, 
                                                            "dataset": dataset_name})

    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="No sequential id was found for that bucket and dataset")

    return True


@router.delete("/dataset/clear-self-training-sequential-id", tags = ['deprecated3'], description= "changed with /datasets/clear-all-self-training-sequential-ids" )
def clear_self_training_sequential_id_jobs(request: Request):
    request.app.self_training_sequential_id_collection.delete_many({})

    return True

@router.get("/dataset/self-training-sequential-id/{dataset} ", tags = ['deprecated3'], description= " changed with /datasets/get-self-training-sequential-id")
def get_self_training_sequential_id(request: Request, dataset: str):
    dataset_path = f"{dataset}/data/latent-generator/self_training/"
    # Check and initialize if necessary
    existing_index = request.app.self_training_sequential_id_collection.find_one({"dataset": dataset})
    if existing_index is None:
        # Count the files in MinIO for the dataset to initialize the index
        files = request.app.minio_client.list_objects('datasets', prefix=dataset_path)
        files = [file.object_name for file in files]
        files_count = len(files)

        request.app.self_training_sequential_id_collection.insert_one({"dataset": dataset, "sequential_id": files_count})
    
    # Atomically fetch and increment the index
    result = request.app.self_training_sequential_id_collection.find_one_and_update(
        {"dataset": dataset},
        {"$inc": {"sequential_id": 1}},
        return_document=ReturnDocument.AFTER
    )

    result.pop("_id", None)
    
    if result:
        return result
    else:
        raise HTTPException(status_code=500, detail="Failed to fetch the sequential id")

# -------------------- Dataset rate -------------------------
@router.get("/dataset/get-rate", tags = ['deprecated3'], description= "changed wtih /datasets/settings/get-dataset-config")
def get_rate(request: Request, dataset: str):
    # find
    query = {"dataset_name": dataset}
    item = request.app.dataset_config_collection.find_one(query)
    if item is None:
        raise HTTPException(status_code=404)

    # remove the auto generated field
    item.pop('_id', None)

    return item["dataset_rate"]


@router.put("/dataset/set-rate", tags = ['deprecated3'], description= "changed wtih /datasets/settings/set-config")
def set_rate(request: Request, dataset, rate=0):
    date_now = datetime.now()
    # check if exist
    query = {"dataset_name": dataset}
    item = request.app.dataset_config_collection.find_one(query)
    if item is None:
        # add one
        dataset_config = {
            "dataset_name": dataset,
            "last_update": date_now,
            "dataset_rate": rate,
            "relevance_model": "",
            "ranking_model": "",
        }
        request.app.dataset_config_collection.insert_one(dataset_config)
    else:
        # update
        new_values = {"$set": {"last_update": date_now, "dataset_rate": rate}}
        request.app.dataset_config_collection.update_one(query, new_values)

    return True


@router.get("/dataset/get-hourly-limit", tags = ['deprecated3'], description= "changed wtih /datasets/settings/get-dataset-config")
def get_rate(request: Request, dataset: str):
    # find
    query = {"dataset_name": dataset}
    item = request.app.dataset_config_collection.find_one(query)
    if item is None:
        raise HTTPException(status_code=404)

    # remove the auto generated field
    item.pop('_id', None)

    return item["hourly_limit"]


@router.put("/dataset/set-hourly-limit", tags = ['deprecated3'], description= "changed wtih /datasets/settings/set-config")
def set_rate(request: Request, dataset, hourly_limit=0):
    date_now = datetime.now()
    # check if exist
    query = {"dataset_name": dataset}
    item = request.app.dataset_config_collection.find_one(query)
    if item is None:
        # add one
        dataset_config = {
            "dataset_name": dataset,
            "last_update": date_now,
            "dataset_rate": 0,
            "hourly_limit": hourly_limit,
            "relevance_model": "",
            "ranking_model": "",
        }
        request.app.dataset_config_collection.insert_one(dataset_config)
    else:
        # update
        new_values = {"$set": {"last_update": date_now, "hourly_limit": hourly_limit}}
        request.app.dataset_config_collection.update_one(query, new_values)

    return True


@router.get("/dataset/get-dataset-config", tags = ['deprecated3'], description= "changed wtih /datasets/settings/get-dataset-config")
def get_dataset_config(request: Request, dataset: str = Query(...)):
    # Find the item for the specific dataset
    item = request.app.dataset_config_collection.find_one({"dataset_name": dataset})

    if item is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # remove the auto generated field
    item.pop('_id', None)

    return item


@router.get("/dataset/get-all-dataset-config", tags = ['deprecated3'], description= "changed wtih /datasets/settings/get-all-dataset-config" )
def get_all_dataset_config(request: Request):
    dataset_configs = []

    # find
    items = request.app.dataset_config_collection.find({})
    if items is None:
        return []

    for item in items:
        # remove the auto generated field
        item.pop('_id', None)
        dataset_configs.append(item)

    return dataset_configs


@router.put("/dataset/set-relevance-model", tags = ['deprecated3'], description= "changed wtih /datasets/settings/set-config")
def set_relevance_model(request: Request, dataset: str, relevance_model: str):
    date_now = datetime.now()
    # check if dataset exists
    query = {"dataset_name": dataset}
    item = request.app.dataset_config_collection.find_one(query)
    
    if item is None:
        raise HTTPException(status_code=422, detail="Dataset not found")
    
    # update the relevance model
    new_values = {
        "$set": {
            "last_update": date_now,
            "relevance_model": relevance_model
        }
    }
    request.app.dataset_config_collection.update_one(query, new_values)
    return True


@router.put("/dataset/set-ranking-model", tags = ['deprecated3'], description= "changed wtih /datasets/settings/set-config")
def set_ranking_model(request: Request, dataset: str, ranking_model: str):
    date_now = datetime.now()
    # check if dataset exists
    query = {"dataset_name": dataset}
    item = request.app.dataset_config_collection.find_one(query)
    
    if item is None:
        raise HTTPException(status_code=422, detail="Dataset not found")
    
    # update the ranking model
    new_values = {
        "$set": {
            "last_update": date_now,
            "ranking_model": ranking_model
        }
    }
    request.app.dataset_config_collection.update_one(query, new_values)
    return True


@router.get("/datasets/rank/list", response_class=PrettyJSONResponse)
def list_ranking_files(request: Request, dataset: str):
    # Construct the path prefix for ranking
    path_prefix = f"{dataset}/data/ranking/aggregate"

    # Fetch the list of objects with the given prefix
    objects = cmd.get_list_of_objects_with_prefix(request.app.minio_client, "datasets", path_prefix)

    # Filter out non-JSON files
    json_files = [obj for obj in objects if obj.endswith('.json')]

    if not json_files:
        return []
    
    return json_files
    

@router.get("/datasets/rank/list-v1", tags = ['deprecated3'], description= "changed wtih /rank/list-ranking-datapoints or /rank/sort-ranking-data-by-date-v2")
def list_ranking_files(
    request: Request, 
    dataset: str, 
    start_date: str = None, 
    end_date: str = None, 
    list_size: int = Query(100),  # Parameter for list size
    offset: int = Query(0, description="Offset for pagination"),  # New parameter for pagination
    order: str = Query("desc")  # Parameter for ordering
):
    # Convert start_date and end_date strings to datetime objects
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

    # Construct the path prefix for ranking
    path_prefix = f"{dataset}/data/ranking/aggregate"

    # Fetch the list of objects with the given prefix
    objects = cmd.get_list_of_objects_with_prefix(request.app.minio_client, "datasets", path_prefix)

    # Filter out non-JSON files and apply date filters
    filtered_json_files = []
    for obj in objects:
        if obj.endswith('.json'):
            # Extract date from the filename
            file_date_str = obj.split('/')[-1].split('-')[0:3]
            file_date_str = '-'.join(file_date_str)  # Reformat to 'YYYY-MM-DD'
            file_date_obj = datetime.strptime(file_date_str, "%Y-%m-%d")

            # Apply date filtering
            if start_date_obj and file_date_obj < start_date_obj:
                continue
            if end_date_obj and file_date_obj > end_date_obj:
                continue

            filtered_json_files.append(obj)

    # Apply ordering
    if order == "desc":
        filtered_json_files.sort(reverse=True)
    else:
        filtered_json_files.sort()

    # Apply offset and list size limit
    start_index = offset
    end_index = offset + list_size
    filtered_json_files = filtered_json_files[start_index:end_index]

    if not filtered_json_files:
        return []

    return filtered_json_files


@router.get("/datasets/rank/list-v3", tags = ['deprecated3'], description= "changed wtih /rank/list-ranking-datapoints or /rank/sort-ranking-data-by-date-v2")
def list_ranking_files_v3(
    request: Request,
    dataset: str,
    start_date: str = None,
    end_date: str = None,
    list_size: int = Query(100),
    offset: int = Query(0, description="Offset for pagination"),
    order: str = Query("desc")
):
    # Convert start_date and end_date strings to datetime objects
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

    # Construct the path prefix for ranking
    path_prefix = f"{dataset}/data/ranking/aggregate"

    # Fetch the list of objects with the given prefix
    objects = cmd.get_list_of_objects_with_prefix(request.app.minio_client, "datasets", path_prefix)

    # Filter out non-JSON files and apply date filters
    filtered_json_contents = []
    for obj in objects:
        if obj.endswith('.json'):
            # Extract date from the filename
            file_date_str = obj.split('/')[-1].split('-')[0:3]
            file_date_str = '-'.join(file_date_str)  # Reformat to 'YYYY-MM-DD'
            file_date_obj = datetime.strptime(file_date_str, "%Y-%m-%d")

            # Apply date filtering
            if start_date_obj and file_date_obj < start_date_obj:
                continue
            if end_date_obj and file_date_obj > end_date_obj:
                continue

            # Fetch and load the JSON file content
            json_content = cmd.get_file_content(request.app.minio_client, "datasets", obj)
            if json_content:
                filtered_json_contents.append(json.loads(json_content))

    # Apply ordering
    if order == "desc":
        filtered_json_contents.sort(key=lambda x: x['datetime'], reverse=True)
    else:
        filtered_json_contents.sort(key=lambda x: x['datetime'])

    # Apply offset and list size limit
    start_index = offset
    end_index = offset + list_size
    filtered_json_contents = filtered_json_contents[start_index:end_index]

    if not filtered_json_contents:
        return []

    # Return the content of the JSON files
    return filtered_json_contents


def read_json_data(request, json_file):
    # Fetch the content of the specified JSON file
    response = cmd.get_file_from_minio(request.app.minio_client, "datasets", json_file)
    decoded_data = response.data.decode()
    item = json.loads(decoded_data)

    selected_image_hash = item["selected_image_hash"]
    return selected_image_hash, json_file


@router.get("/datasets/rank/list-sort-by-score", tags = ['deprecated3'], description="Deprecated without a direct alternative. Inform in the chat if you are using this endpoint")
def list_ranking_files_sort_by_score(
    request: Request, 
    dataset: str,
    model_id: int,
    start_date: str = None, 
    end_date: str = None, 
    list_size: int = Query(100, description="Number of results to return"),
    offset: int = Query(0, description="Offset for pagination"),
    order: str = Query("desc", description="Order in which the data should be returned")
):
    # Convert start_date and end_date strings to datetime objects
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

    # Construct the path prefix for ranking
    path_prefix = f"{dataset}/data/ranking/aggregate"

    # Fetch the list of objects with the given prefix
    objects = cmd.get_list_of_objects_with_prefix(request.app.minio_client, "datasets", path_prefix)

    # Filter out non-JSON files
    json_files = [obj for obj in objects if obj.endswith('.json')]

    if not json_files:
        return []

    # Query for model sigma scores
    query = {"model_id": model_id}
    sort_order = -1 if order == "desc" else 1
    model_scores = request.app.image_scores_collection.find(query).sort("score", sort_order)
    model_scores = list(model_scores)

    if len(model_scores) == 0:
        raise HTTPException(status_code=404, detail="Image rank scores data not found")

    # Read json files and filter based on date range and pagination
    json_files_selected_hash_dict = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for json_file in json_files:
            file_date_str = json_file.split('/')[-1].split('-')[0:3]
            file_date_str = '-'.join(file_date_str)
            file_date_obj = datetime.strptime(file_date_str, "%Y-%m-%d")

            # Apply date filtering
            if start_date_obj and file_date_obj < start_date_obj:
                continue
            if end_date_obj and file_date_obj > end_date_obj:
                continue

            futures.append(executor.submit(read_json_data, request=request, json_file=json_file))

        for future in as_completed(futures):
            selected_image_hash, json_file = future.result()
            json_files_selected_hash_dict[selected_image_hash] = json_file

    # Sort and paginate the results
    sorted_json_files = []
    for score_data in model_scores:
        if score_data["image_hash"] in json_files_selected_hash_dict:
            json_file = json_files_selected_hash_dict[score_data["image_hash"]]
            sorted_json_files.append(json_file)

    # Apply offset and list size limit
    start_index = offset
    end_index = offset + list_size
    sorted_json_files = sorted_json_files[start_index:end_index]

    return sorted_json_files


@router.get("/datasets/rank/list-sort-by-residual", response_class=PrettyJSONResponse, tags = ['deprecated3'], description= "changed wtih /rank/sort-ranking-data-by-residual-v1")
def list_ranking_files_sort_by_residual(
    request: Request, 
    dataset: str,
    model_id: int,
    start_date: str = None, 
    end_date: str = None, 
    list_size: int = Query(100, description="Number of results to return"),
    offset: int = Query(0, description="Offset for pagination"),
    order: str = Query("desc", description="Order in which the data should be returned")
):
    # Convert start_date and end_date strings to datetime objects
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

    # Construct the path prefix for ranking
    path_prefix = f"{dataset}/data/ranking/aggregate"

    # Fetch the list of objects with the given prefix
    objects = cmd.get_list_of_objects_with_prefix(request.app.minio_client, "datasets", path_prefix)

    # Filter out non-JSON files
    json_files = [obj for obj in objects if obj.endswith('.json')]

    if not json_files:
        return []

    # Query for model residuals
    query = {"model_id": model_id}
    sort_order = -1 if order == "desc" else 1
    model_residuals = request.app.image_residuals_collection.find(query).sort("residual", sort_order)
    model_residuals = list(model_residuals)

    if len(model_residuals) == 0:
        raise HTTPException(status_code=404, detail="Image rank residuals data not found")

    # Read json files and filter based on date range and pagination
    json_files_selected_hash_dict = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for json_file in json_files:
            file_date_str = json_file.split('/')[-1].split('-')[0:3]
            file_date_str = '-'.join(file_date_str)
            file_date_obj = datetime.strptime(file_date_str, "%Y-%m-%d")

            # Apply date filtering
            if start_date_obj and file_date_obj < start_date_obj:
                continue
            if end_date_obj and file_date_obj > end_date_obj:
                continue

            futures.append(executor.submit(read_json_data, request=request, json_file=json_file))

        for future in as_completed(futures):
            selected_image_hash, json_file = future.result()
            json_files_selected_hash_dict[selected_image_hash] = json_file

    # Sort and paginate the results
    sorted_json_files = []
    for residual_data in model_residuals:
        if residual_data["image_hash"] in json_files_selected_hash_dict:
            json_file = json_files_selected_hash_dict[residual_data["image_hash"]]
            sorted_json_files.append(json_file)

    # Apply offset and list size limit
    start_index = offset
    end_index = offset + list_size
    sorted_json_files = sorted_json_files[start_index:end_index]

    return sorted_json_files



@router.get("/datasets/relevancy/list", response_class=PrettyJSONResponse,tags = ['deprecated3'], description= "changed wtih /rank/list-relevance-datapoints" )
def list_relevancy_files(request: Request, dataset: str):
    # Construct the path prefix for relevancy
    path_prefix = f"{dataset}/data/relevancy/aggregate"

    # Fetch the list of objects with the given prefix
    objects = cmd.get_list_of_objects_with_prefix(request.app.minio_client, "datasets", path_prefix)

    # Filter out non-JSON files
    json_files = [obj for obj in objects if obj.endswith('.json')]

    if not json_files:
        return []

    return json_files


@router.put("/datasets/rank/update_datapoint", tags = ['deprecated3'], description= "changed wtih /rank/update-ranking-datapoint" )
def update_ranking_file(request: Request, dataset: str, filename: str, update_data: FlaggedDataUpdate):
    # Construct the object name based on the dataset
    object_name = f"{dataset}/data/ranking/aggregate/{filename}"

    # Fetch the content of the specified JSON file
    data = cmd.get_file_from_minio(request.app.minio_client, "datasets", object_name)

    if data is None:
        raise HTTPException(status_code=410, detail=f"File {filename} not found.")
        
    file_content = ""
    for chunk in data.stream(32 * 1024):
        file_content += chunk.decode('utf-8')

    # Load the existing content and update the flagged field, flagged_time, and flagged_by_user
    content_dict = json.loads(file_content)
    content_dict["flagged"] = update_data.flagged
    content_dict["flagged_by_user"] = update_data.flagged_by_user
    content_dict["flagged_time"] = update_data.flagged_time if update_data.flagged_time else datetime.now().isoformat()

    # Save the modified file back
    updated_content = json.dumps(content_dict, indent=2)
    updated_data = io.BytesIO(updated_content.encode('utf-8'))
    request.app.minio_client.put_object("datasets", object_name, updated_data, len(updated_content))

    return {"message": f"File {filename} has been updated."}


# New standardized apis


@router.delete("/datasets/clear-all-sequential-id",
               description="Clear all documents from the dataset sequential ID collection",
               response_model=StandardSuccessResponseV1[WasPresentResponse],  
               tags=["dataset"],
               responses=ApiResponseHandlerV1.listErrors([500]))
async def clear_dataset_sequential_id_jobs(request: Request):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        # Check if there are documents in the collection
        was_present = request.app.dataset_sequential_id_collection.count_documents({}) > 0

        if not was_present:
            # If no documents are present, return False in the wasPresent field of the response
            return response_handler.create_success_delete_response_v1(
                response_data=False, 
                http_status_code=200
            )

        # If documents are present, delete them
        request.app.dataset_sequential_id_collection.delete_many({})

        # Assuming deletion is always successful, return True in the wasPresent field
        return response_handler.create_success_delete_response_v1(
            response_data=True, 
            http_status_code=200
        )
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )


@router.get("/datasets/list-datasets",
            description="List datasets from storage",
            response_model=StandardSuccessResponseV1[DatasetResponse], 
            tags=["dataset"],
            responses=ApiResponseHandlerV1.listErrors([500]))
async def get_datasets(request: Request):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        objects = cmd.get_list_of_objects(request.app.minio_client, "datasets")
        return response_handler.create_success_response_v1(
            response_data={"datasets": objects},  # Ensure the response data structure matches your requirements
            http_status_code=200
        )
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )
    
@router.get("/datasets/get-all-sequential-ids",
            description="Get all sequential IDs for datasets",
            response_model=StandardSuccessResponseV1[SeqIdResponse],  
            tags=["dataset"],
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
async def get_all_sequential_ids(request: Request):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        # Fetch all sequential IDs from the collection
        sequential_ids = list(request.app.dataset_sequential_id_collection.find())

        # Convert MongoDB ObjectId to string for serialization
        for seq_id in sequential_ids:
            seq_id.pop('_id', None)

        # Return the sequential IDs
        return response_handler.create_success_response_v1(
            response_data={"sequential_ids": sequential_ids},
            http_status_code=200
        )
        
    except Exception as e:
        # Handle exceptions and return an error response
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )


@router.get("/datasets/get-sequential-ids",
            description="Get or create sequential ID for a dataset",
            response_model=StandardSuccessResponseV1[SeqIdResponse],  
            tags=["dataset"],
            responses=ApiResponseHandlerV1.listErrors([400,422, 500]))
async def get_sequential_id_1(request: Request, dataset: str = Query(..., description="Name of the dataset"), amount: int = Query(default=1, ge=1)):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    sequential_id_arr = []

    try:
        # Check if dataset exists in the collection or object list
        dataset_exists = cmd.get_list_of_objects(request.app.minio_client, "datasets")
        dataset_path = f"{dataset}"

        if dataset_path not in dataset_exists:
            # Return 422 error if dataset does not exist
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string=f"Dataset '{dataset}' does not exist.",
                http_status_code=422,
            )


        sequential_id = request.app.dataset_sequential_id_collection.find_one({"dataset_name": dataset})
        

        if sequential_id is None:
            # create one
            new_sequential_id = SequentialID(dataset)
            # get the sequential id arr
            for i in range(amount):
                sequential_id_arr.append(new_sequential_id.get_sequential_id())
            # add to collection
            request.app.dataset_sequential_id_collection.insert_one(new_sequential_id.to_dict())

        else:
            # if found, use the found sequential id
            found_sequential_id = SequentialID(sequential_id["dataset_name"], sequential_id.get("subfolder_count", 0),
                                               sequential_id.get("file_count", 0))
            for i in range(amount):
                sequential_id_arr.append(found_sequential_id.get_sequential_id())

            new_values = {"$set": found_sequential_id.to_dict()}
            # update existing sequential id
            request.app.dataset_sequential_id_collection.update_one({"dataset_name": dataset}, new_values)

        # Return the sequential IDs
        return response_handler.create_success_response_v1(
            response_data={"sequential_ids": sequential_id_arr}, 
            http_status_code=200
        )
        
    except Exception as e:
        # Handle exceptions and return an error response
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )
    
    
@router.delete("/datasets/clear-all-self-training-sequential-ids",
               description="Clear all documents from the self-training sequential ID collection",
               response_model=StandardSuccessResponseV1[WasPresentResponse],  
               tags=["dataset"],
               responses=ApiResponseHandlerV1.listErrors([500]))
async def clear_self_training_sequential_id_jobs(request: Request):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:

        # Check if there are documents in the collection
        was_present = request.app.self_training_sequential_id_collection.count_documents({}) > 0

        if not was_present:
            # If no documents are present, return False in the wasPresent field of the response
            return response_handler.create_success_delete_response_v1(
                response_data=False, 
                http_status_code=200
            )


        request.app.self_training_sequential_id_collection.delete_many({})
        # Assuming deletion is always successful, returning True for simplification
        return response_handler.create_success_delete_response_v1(
            response_data=True, 
            http_status_code=200
        )
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )


@router.get("/datasets/get-self-training-sequential-id",
            description="Get or create self-training sequential ID for a dataset",
            tags=["dataset"],
            response_model=StandardSuccessResponseV1[SeqIdDatasetResponse],  
            responses=ApiResponseHandlerV1.listErrors([400,422, 500]))
async def get_self_training_sequential_id(request: Request, dataset: str = Query(..., description="Name of the dataset")):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        dataset_path = f"{dataset}/data/latent-generator/self_training/"
        objects = cmd.get_list_of_objects(request.app.minio_client, "datasets")

        dataset_path = f'{dataset}'
        
        if dataset_path not in objects:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string=f"Dataset '{dataset}' does not exist.",
                http_status_code=422,
            )
        # Check and initialize if necessary
        existing_index = request.app.self_training_sequential_id_collection.find_one({"dataset": dataset})
        if existing_index is None:
            # Count the files in MinIO for the dataset to initialize the index
            files_count = sum(1 for _ in request.app.minio_client.list_objects('datasets', prefix=dataset_path))
            request.app.self_training_sequential_id_collection.insert_one({"dataset": dataset, "sequential_id": files_count})
        
        # Atomically fetch and increment the index
        result = request.app.self_training_sequential_id_collection.find_one_and_update(
            {"dataset": dataset},
            {"$inc": {"sequential_id": 1}},
            return_document=ReturnDocument.AFTER
        )

        if result:
            result.pop("_id", None)
            return response_handler.create_success_response_v1(
                response_data=result, 
                http_status_code=200
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to fetch the sequential id")
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )        
    
@router.post("/datasets/add-new-dataset",
            description="Add new dataset in MongoDB",
            tags=["dataset"],
            response_model=StandardSuccessResponseV1[Dataset],  
            responses=ApiResponseHandlerV1.listErrors([400, 422]))
async def add_new_dataset(request: Request, dataset: Dataset):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    if request.app.datasets_collection.find_one({"dataset_name": dataset.dataset_name}):
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string='Dataset already exists',
            http_status_code=400
        )

    # Find the current highest dataset_id
    highest_dataset = request.app.datasets_collection.find_one(
        sort=[("dataset_id", -1)]
    )
    next_dataset_id = (highest_dataset["dataset_id"] + 1) if highest_dataset else 0

    # Add the dataset_id to the dataset
    dataset_dict = dataset.to_dict()
    dataset_dict["dataset_id"] = next_dataset_id

    # Insert the new dataset with dataset_id
    request.app.datasets_collection.insert_one(dataset_dict)

    return response_handler.create_success_response_v1(
        response_data={"dataset_name": dataset.dataset_name, "dataset_id": next_dataset_id}, 
        http_status_code=200
    )
   
    
@router.put("/datasets/update-dataset-ids",
            description="Update datasets in MongoDB to add dataset_id sequentially",
            tags=["dataset"],
            response_model=StandardSuccessResponseV1,
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
async def update_dataset_ids(request: Request):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        # Fetch all documents
        all_datasets = list(request.app.extract_datasets_collection.find({}))

        # Update each document with a sequential dataset_id
        for idx, dataset in enumerate(all_datasets):
            request.app.extract_datasets_collection.update_one(
                {"_id": dataset["_id"]},
                {"$set": {"dataset_id": idx}}
            )

        return response_handler.create_success_response_v1(
            response_data={"message": "Datasets updated successfully"},
            http_status_code=200
        )
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )


@router.get("/datasets/list-datasets-v1",
            description="list datasets from mongodb",
            tags=["dataset"],
            response_model=StandardSuccessResponseV1[ListDataset],  
            responses=ApiResponseHandlerV1.listErrors([422]))
async def list_datasets(request: Request):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    datasets = list(request.app.datasets_collection.find({}))
    for dataset in datasets:
        dataset.pop('_id', None)

    return response_handler.create_success_response_v1(
                response_data={'datasets': datasets}, 
                http_status_code=200
            )       


@router.delete("/datasets/remove-dataset",
               description="Remove dataset and its configuration in MongoDB",
               tags=["dataset"],
               response_model=StandardSuccessResponseV1[WasPresentResponse],  
               responses=ApiResponseHandlerV1.listErrors([422]))
async def remove_dataset(request: Request, dataset: str = Query(...)):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    # Attempt to delete the dataset
    dataset_result = request.app.datasets_collection.delete_one({"dataset_name": dataset})
    # Attempt to delete the dataset configuration
    config_result = request.app.dataset_config_collection.delete_one({"dataset_name": dataset})

    # Check if either the dataset or its configuration was present and deleted
    was_present = dataset_result.deleted_count > 0

    # Using the check to determine which response to send
    if was_present:
        # If either was deleted, return True
        return response_handler.create_success_delete_response_v1(
            True, 
            http_status_code=200
        )
    else:
        # If neither was deleted, return False
        return response_handler.create_success_delete_response_v1(
            False, 
            http_status_code=200
        )