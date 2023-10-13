from fastapi import Request, HTTPException, APIRouter, Response, Query
from orchestration.api.mongo_schemas import SequentialID
from utility.minio import cmd
from datetime import datetime
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
    found_sequential_id = SequentialID(sequential_id["dataset_name"], sequential_id["subfolder_count"], sequential_id["file_count"])
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

@router.get("/dataset/get-dataset-config")
def get_dataset_config(request: Request, dataset: str = Query(...)):
    # Find the item for the specific dataset
    item = request.app.dataset_config_collection.find_one({"dataset_name": dataset})
    
    if item is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # remove the auto generated field
    item.pop('_id', None)
    
    # Fetch top-k
    top_k_item = request.app.dataset_top_k_collection.find_one({"dataset_name": dataset})
    if top_k_item:
        item["top_k"] = top_k_item["top_k"]
    else:
        item["top_k"] = None

    # Fetch generation-policy
    gen_policy_item = request.app.dataset_generation_policy_collection.find_one({"dataset_name": dataset})
    if gen_policy_item:
        item["generation_policy"] = gen_policy_item["generation_policy"]
    else:
        item["generation_policy"] = None

    return item



@router.get("/dataset/get-all-dataset-config")
def get_all_dataset_config(request: Request):
    dataset_configs = []
    
    # find
    items = request.app.dataset_config_collection.find({})
    if items is None:
        raise HTTPException(status_code=404)

    for item in items:
        # remove the auto generated field
        item.pop('_id', None)
        
        # Fetch top-k for this dataset
        top_k_item = request.app.dataset_top_k_collection.find_one({"dataset_name": item["dataset_name"]})
        if top_k_item:
            item["top_k"] = top_k_item["top_k"]
        else:
            item["top_k"] = None

        # Fetch generation-policy for this dataset
        gen_policy_item = request.app.dataset_generation_policy_collection.find_one({"dataset_name": item["dataset_name"]})
        if gen_policy_item:
            item["generation_policy"] = gen_policy_item["generation_policy"]
        else:
            item["generation_policy"] = None
        
        dataset_configs.append(item)

    return dataset_configs

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


# -------------------- Dataset generation policy -------------------------

@router.get("/dataset/get-all-dataset-generation-policy")
def get_all_dataset_generation_policy(request: Request):
    dataset_generation_policies = []
    # find
    items = request.app.dataset_generation_policy_collection.find({})
    if items is None:
        raise HTTPException(status_code=204)

    for item in items:
        # remove the auto generated field
        item.pop('_id', None)
        dataset_generation_policies.append(item)

    return dataset_generation_policies


@router.get("/dataset/get-generation-policy")
def get_generation_policy(request: Request, dataset: str):
    # find
    query = {"dataset_name": dataset}
    item = request.app.dataset_generation_policy_collection.find_one(query)
    if item is None:
        raise HTTPException(status_code=204)

    # remove the auto generated field
    item.pop('_id', None)

    return item


@router.put("/dataset/set-generation-policy")
def set_generation_policy(request: Request, dataset, generation_policy='top-k'):
    date_now = datetime.now()
    # check if exist
    # and remove all entries
    query = {"dataset_name": dataset}
    request.app.dataset_generation_policy_collection.delete_many(query)

    # add one
    dataset_generation_policy = {
        "dataset_name": dataset,
        "last_update": date_now,
        "generation_policy": generation_policy,
    }
    request.app.dataset_generation_policy_collection.insert_one(dataset_generation_policy)

    return True


@router.get("/dataset/get-top-k")
def get_top_k(request: Request, dataset: str):
    # find
    query = {"dataset_name": dataset}
    item = request.app.dataset_top_k_collection.find_one(query)
    if item is None:
        raise HTTPException(status_code=204)

    # remove the auto generated field
    item.pop('_id', None)

    return item


@router.put("/dataset/set-top-k")
def set_top_k(request: Request, dataset, top_k=0.1):
    date_now = datetime.now()
    # check if exist
    # and remove all entries
    query = {"dataset_name": dataset}
    request.app.dataset_top_k_collection.delete_many(query)

    # add one
    dataset_top_k = {
        "dataset_name": dataset,
        "last_update": date_now,
        "top_k": top_k,
    }
    request.app.dataset_top_k_collection.insert_one(dataset_top_k)

    return True