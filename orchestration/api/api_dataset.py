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


@router.get("/dataset/list")
def get_datasets(request: Request):
    objects = cmd.get_list_of_objects(request.app.minio_client, "datasets")

    return objects


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


@router.get("/dataset/get-all-dataset-config", response_class=PrettyJSONResponse)
def get_all_dataset_config(request: Request):
    dataset_configs = []

    # find
    items = request.app.dataset_config_collection.find({})
    if items is None:
        raise HTTPException(status_code=404)

    for item in items:
        # remove the auto generated field
        item.pop('_id', None)
        dataset_configs.append(item)

    return dataset_configs


@router.put("/dataset/set-relevance-model")
def set_relevance_model(request: Request, dataset: str, relevance_model: str):
    date_now = datetime.now()
    # check if dataset exists
    query = {"dataset_name": dataset}
    item = request.app.dataset_config_collection.find_one(query)
    
    if item is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
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
        raise HTTPException(status_code=404, detail="Dataset not found")
    
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
        raise HTTPException(status_code=404, detail=f"No JSON files found in {path_prefix}.")

    return json_files


def read_json_data(request, json_file):
    # Fetch the content of the specified JSON file
    response = cmd.get_file_from_minio(request.app.minio_client, "datasets", json_file)
    decoded_data = response.data.decode()
    item = json.loads(decoded_data)

    selected_image_hash = item["selected_image_hash"]
    return selected_image_hash, json_file


@router.get("/datasets/rank/list-sort-by-residual", response_class=PrettyJSONResponse)
def list_ranking_files_sort_by_residual(request: Request, dataset: str,
                                        model_id: int,
                                        order: str = Query("desc",
                                                           description="Order in which the data should be returned")):
    # Construct the path prefix for ranking
    path_prefix = f"{dataset}/data/ranking/aggregate"

    # Fetch the list of objects with the given prefix
    objects = cmd.get_list_of_objects_with_prefix(request.app.minio_client, "datasets", path_prefix)

    # Filter out non-JSON files
    json_files = [obj for obj in objects if obj.endswith('.json')]

    if not json_files:
        raise HTTPException(status_code=404, detail=f"No JSON files found in {path_prefix}.")

    # get all model id residuals
    query = {"model_id": model_id}
    sort_order = -1 if order == "desc" else 1
    model_residuals = request.app.image_residuals_collection.find(query).sort("residual", sort_order)
    model_residuals = list(model_residuals)
    if len(model_residuals) == 0:
        raise HTTPException(status_code=404, detail="Image rank residuals data not found")

    # use concurrency
    # read json files and put selected hash in a dict
    json_files_selected_hash_dict = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        count = 0
        for json_file in json_files:
            futures.append(executor.submit(read_json_data, request=request, json_file=json_file))
            count += 1

        for future in as_completed(futures):
            selected_image_hash, json_file = future.result()
            json_files_selected_hash_dict[selected_image_hash] = json_file

    # get json file list
    sorted_json_files = []
    for residual_data in model_residuals:
        if residual_data["image_hash"] in json_files_selected_hash_dict:
            json_file = json_files_selected_hash_dict[residual_data["image_hash"]]
            sorted_json_files.append(json_file)

    return sorted_json_files


@router.get("/datasets/relevancy/list", response_class=PrettyJSONResponse)
def list_relevancy_files(request: Request, dataset: str):
    # Construct the path prefix for relevancy
    path_prefix = f"{dataset}/data/relevancy/aggregate"

    # Fetch the list of objects with the given prefix
    objects = cmd.get_list_of_objects_with_prefix(request.app.minio_client, "datasets", path_prefix)

    # Filter out non-JSON files
    json_files = [obj for obj in objects if obj.endswith('.json')]

    if not json_files:
        raise HTTPException(status_code=404, detail=f"No JSON files found in {path_prefix}.")

    return json_files


@router.get("/datasets/rank/read", response_class=PrettyJSONResponse)
def read_ranking_file(request: Request, dataset: str,
                      filename: str = Query(..., description="Filename of the JSON to read")):
    # Construct the object name for ranking
    object_name = f"{dataset}/data/ranking/aggregate/{filename}"

    # Fetch the content of the specified JSON file
    data = cmd.get_file_from_minio(request.app.minio_client, "datasets", object_name)

    if data is None:
        raise HTTPException(status_code=404, detail=f"File {filename} not found.")

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
        raise HTTPException(status_code=404, detail=f"File {filename} not found.")

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
        raise HTTPException(status_code=404, detail=f"File {filename} not found.")

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
