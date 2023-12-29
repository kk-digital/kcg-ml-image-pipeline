import os.path

from fastapi import Request, HTTPException, APIRouter, Response, Query
from orchestration.api.mongo_schemas import SequentialID
from utility.minio import cmd
import json
from datetime import datetime
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from .api_utils import PrettyJSONResponse
from .mongo_schemas import FlaggedDataUpdate, RankingModel
router = APIRouter()


@router.delete("/dataset/clear-sequential-id")
def clear_dataset_sequential_id_jobs(request: Request):
    request.app.dataset_sequential_id_collection.delete_many({})

    return True

@router.delete("/dataset/clear-sequential-id")
def clear_dataset_sequential_id_jobs(request: Request):
    # Check if there are any documents in the collection before deletion
    pre_delete_count = request.app.dataset_sequential_id_collection.count_documents({})

    # Perform the deletion operation
    result = request.app.dataset_sequential_id_collection.delete_many({})

    # Determine if any documents were present and deleted
    was_present = pre_delete_count > 0

    # Return the response indicating whether the documents were present
    return {"wasPresent": was_present}

@router.get("/dataset/list")
def get_datasets(request: Request):
    objects = cmd.get_list_of_objects(request.app.minio_client, "datasets")

    return objects

@router.get("/dataset/list-v1")
def get_datasets(request: Request):
    objects = cmd.get_list_of_objects(request.app.minio_client, "datasets")

    return {"objects": objects}

@router.get("/dataset/sequential-id/{dataset}")
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

@router.get("/dataset/sequential-id-v1/{dataset}")
def get_sequential_id(request: Request, dataset: str, limit: int = 1):

    if not request.app.dataset_sequential_id_collection.find_one({"dataset_name": dataset}):
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found")

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

    return {"sequential_id_arr": sequential_id_arr}

# -------------------- Dataset rate -------------------------
@router.get("/dataset/get-rate")
def get_rate(request: Request, dataset: str):
    # find
    query = {"dataset_name": dataset}
    item = request.app.dataset_config_collection.find_one(query)
    if item is None:
        raise HTTPException(status_code=404)

    # remove the auto generated field
    item.pop('_id', None)

    return item["dataset_rate"]

@router.get("/dataset/rate")
def get_rate(request: Request, dataset: str):
    # find
    query = {"dataset_name": dataset}
    item = request.app.dataset_config_collection.find_one(query)
    if item is None:
        raise HTTPException(status_code=404)

    # remove the auto generated field
    item.pop('_id', None)

    return item["dataset_rate"]

@router.put("/dataset/set-rate")
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


@router.get("/dataset/get-hourly-limit")
def get_rate(request: Request, dataset: str):
    # find
    query = {"dataset_name": dataset}
    item = request.app.dataset_config_collection.find_one(query)
    if item is None:
        raise HTTPException(status_code=404)

    # remove the auto generated field
    item.pop('_id', None)

    return item["hourly_limit"]

@router.get("/dataset/hourly-limit")
def get_rate(request: Request, dataset: str):
    # find
    query = {"dataset_name": dataset}
    item = request.app.dataset_config_collection.find_one(query)
    if item is None:
        raise HTTPException(status_code=404)

    # remove the auto generated field
    item.pop('_id', None)

    return item["hourly_limit"]

@router.put("/dataset/set-hourly-limit")
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


@router.get("/dataset/get-dataset-config", response_class=PrettyJSONResponse)
def get_dataset_config(request: Request, dataset: str = Query(...)):
    # Find the item for the specific dataset
    item = request.app.dataset_config_collection.find_one({"dataset_name": dataset})

    if item is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # remove the auto generated field
    item.pop('_id', None)

    return item

@router.get("/dataset/dataset-config", response_class=PrettyJSONResponse)
def get_dataset_config(request: Request, dataset: str = Query(...)):
    # Find the item for the specific dataset
    item = request.app.dataset_config_collection.find_one({"dataset_name": dataset})

    if item is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # remove the auto generated field
    item.pop('_id', None)

    return item

@router.get("/dataset/get-all-dataset-config", response_class=PrettyJSONResponse)
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

@router.get("/dataset/all-dataset-config", response_class=PrettyJSONResponse)
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
    return {"dataset_configs": dataset_configs}

@router.put("/dataset/set-relevance-model")
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


@router.put("/dataset/set-ranking-model")
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

@router.get("/datasets/rank/list-v1", response_class=PrettyJSONResponse)
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

@router.get("/datasets/rank/list-v1", response_class=PrettyJSONResponse)
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

    return {"filtered_json_files": filtered_json_files}


def read_json_data(request, json_file):
    # Fetch the content of the specified JSON file
    response = cmd.get_file_from_minio(request.app.minio_client, "datasets", json_file)
    decoded_data = response.data.decode()
    item = json.loads(decoded_data)

    selected_image_hash = item["selected_image_hash"]
    return selected_image_hash, json_file


@router.get("/datasets/rank/list-sort-by-score", response_class=PrettyJSONResponse)
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

@router.get("/datasets/rank/sort-by-score", response_class=PrettyJSONResponse)
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

    return {"sorted_json_files": sorted_json_files}

@router.get("/datasets/rank/list-sort-by-residual", response_class=PrettyJSONResponse)
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

@router.get("/datasets/rank/sort-by-residual", response_class=PrettyJSONResponse)
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

    return {"sorted_json_files":sorted_json_files}


@router.get("/datasets/relevancy/list", response_class=PrettyJSONResponse)
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

@router.get("/datasets/relevancy/list-v1", response_class=PrettyJSONResponse)
def list_relevancy_files(request: Request, dataset: str):
    # Construct the path prefix for relevancy
    path_prefix = f"{dataset}/data/relevancy/aggregate"

    # Fetch the list of objects with the given prefix
    objects = cmd.get_list_of_objects_with_prefix(request.app.minio_client, "datasets", path_prefix)

    # Filter out non-JSON files
    json_files = [obj for obj in objects if obj.endswith('.json')]

    if not json_files:
        return []

    return {"json_files": json_files}

@router.get("/datasets/rank/read", response_class=PrettyJSONResponse)
def read_ranking_file(request: Request, dataset: str,
                      filename: str = Query(..., description="Filename of the JSON to read")):
    # Construct the object name for ranking
    object_name = f"{dataset}/data/ranking/aggregate/{filename}"

    # Fetch the content of the specified JSON file
    data = cmd.get_file_from_minio(request.app.minio_client, "datasets", object_name)

    if data is None:
        raise HTTPException(status_code=410, detail=f"File {filename} not found.")

    file_content = ""
    for chunk in data.stream(32 * 1024):
        file_content += chunk.decode('utf-8')

    # Return the content of the JSON file
    return json.loads(file_content)


@router.get("/datasets/relevancy/read", response_class=PrettyJSONResponse)
def read_relevancy_file(request: Request, dataset: str,
                        filename: str = Query(..., description="Filename of the JSON to read")):
    # Construct the object name for relevancy
    object_name = f"{dataset}/data/relevancy/aggregate/{filename}"

    # Fetch the content of the specified JSON file
    data = cmd.get_file_from_minio(request.app.minio_client, "datasets", object_name)

    if data is None:
        raise HTTPException(status_code=410, detail=f"File {filename} not found.")

    file_content = ""
    for chunk in data.stream(32 * 1024):
        file_content += chunk.decode('utf-8')

    # Return the content of the JSON file
    return json.loads(file_content)


@router.put("/datasets/rank/update_datapoint")
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
