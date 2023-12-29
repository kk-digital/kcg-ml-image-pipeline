from fastapi import Request, HTTPException, APIRouter, Response, Query
from utility.minio import cmd
from datetime import datetime
from .api_utils import PrettyJSONResponse
from fastapi.encoders import jsonable_encoder
from bson import ObjectId

router = APIRouter()


# -------------------- Dataset generation policy -------------------------

@router.get("/dataset/settings/get-all-dataset-generation-policy", response_class=PrettyJSONResponse)
def get_all_dataset_generation_policy(request: Request):
    dataset_generation_policies = []
    # find
    items = request.app.dataset_config_collection.find({})
    if items is None:
        raise HTTPException(status_code=204)

    for item in items:
        # remove the auto generated field
        item.pop('_id', None)
        dataset_generation_policies.append(item)

    return dataset_generation_policies

@router.get("/dataset/settings/all-dataset-generation-policy", response_class=PrettyJSONResponse)
def get_all_dataset_generation_policy(request: Request):
    dataset_generation_policies = []
    # find
    items = request.app.dataset_config_collection.find({})
    if items is None:
        raise HTTPException(status_code=204)

    for item in items:
        # remove the auto generated field
        item.pop('_id', None)
        dataset_generation_policies.append(item)

    return {"dataset_generation_policies": dataset_generation_policies}

@router.get("/dataset/settings/get-generation-policy", response_class=PrettyJSONResponse)
def get_generation_policy(request: Request, dataset: str):
    # find
    query = {"dataset_name": dataset}
    item = request.app.dataset_config_collection.find_one(query)
    if item is None or "generation_policy" not in item:
        raise HTTPException(status_code=204)

    return item["generation_policy"]

@router.get("/dataset/settings/generation-policy", response_class=PrettyJSONResponse)
def get_generation_policy(request: Request, dataset: str):
    # find
    query = {"dataset_name": dataset}
    item = request.app.dataset_config_collection.find_one(query)
    if item is None or "generation_policy" not in item:
        raise HTTPException(status_code=204)

    return item["generation_policy"]

@router.put("/dataset/settings/set-generation-policy")
def set_generation_policy(request: Request, dataset, generation_policy='top-k'):
    date_now = datetime.now()
    
    # Check if exist
    query = {"dataset_name": dataset}
    item = request.app.dataset_config_collection.find_one(query)
    
    if item is None:
        # Add a new entry
        dataset_config = {
            "dataset_name": dataset,
            "last_update": date_now,
            "generation_policy": generation_policy,
            "relevance_model": "",
            "ranking_model": "",
        }
        request.app.dataset_config_collection.insert_one(dataset_config)
    else:
        # Update the existing entry
        new_values = {"$set": {"last_update": date_now, "generation_policy": generation_policy}}
        request.app.dataset_config_collection.update_one(query, new_values)

    return True


@router.get("/dataset/settings/get-top-k")
def get_top_k(request: Request, dataset: str):
    # find
    query = {"dataset_name": dataset}
    item = request.app.dataset_config_collection.find_one(query)
    if item is None or "top_k" not in item:
        raise HTTPException(status_code=204)

    return item["top_k"]


@router.put("/dataset/settings/set-top-k")
def set_top_k(request: Request, dataset, top_k=0.1):
    date_now = datetime.now()
    
    # Check if exist
    query = {"dataset_name": dataset}
    item = request.app.dataset_config_collection.find_one(query)
    
    if item is None:
        # Add a new entry
        dataset_config = {
            "dataset_name": dataset,
            "last_update": date_now,
            "top_k": top_k,
            "relevance_model": "",
            "ranking_model": "",
        }
        request.app.dataset_config_collection.insert_one(dataset_config)
    else:
        # Update the existing entry
        new_values = {"$set": {"last_update": date_now, "top_k": top_k}}
        request.app.dataset_config_collection.update_one(query, new_values)

    return True


@router.post("/dataset/settings/set-option")
def set_option(request: Request, dataset: str, generation_policy: str):
    if generation_policy not in ["generation-off", "rate-generation", "rate-generation-top-k"]:
        raise HTTPException(status_code=400, detail="Invalid generation policy. Accepted values are 'generation-off', 'rate-generation', and 'rate-generation-top-k'.")

    dataset_config = request.app.dataset_config_collection.find_one({"dataset_name": dataset})
    if dataset_config is not None:
        request.app.dataset_config_collection.update_one(
            {"dataset_name": dataset}, 
            {"$set": {"generation_policy": generation_policy}}
        )
    else:
        request.app.dataset_config_collection.insert_one(
            {"dataset_name": dataset, "generation_policy": generation_policy}
        )
    return {
        "status": "success",
        "message": "Generation policy set successfully."
    }


@router.post("/dataset/settings/set-option-v1", response_class=PrettyJSONResponse)
def set_option(request: Request, dataset: str, generation_policy: str):
    if generation_policy not in ["generation-off", "rate-generation", "rate-generation-top-k"]:
        raise HTTPException(status_code=400, detail="Invalid generation policy. Accepted values are 'generation-off', 'rate-generation', and 'rate-generation-top-k'.")

    # Check if the dataset configuration already exists
    dataset_config = request.app.dataset_config_collection.find_one({"dataset_name": dataset})
    if dataset_config is not None:
        # Update the existing configuration
        request.app.dataset_config_collection.update_one(
            {"dataset_name": dataset}, 
            {"$set": {"generation_policy": generation_policy}}
        )
    else:
        # Insert a new configuration
        inserted_id = request.app.dataset_config_collection.insert_one(
            {"dataset_name": dataset, "generation_policy": generation_policy}
        ).inserted_id
        dataset_config = {"_id": inserted_id, "dataset_name": dataset, "generation_policy": generation_policy}

    # Manually convert ObjectId to string
    dataset_config['_id'] = str(dataset_config['_id'])

    # Convert the MongoDB document to a JSON serializable format
    dataset_config = jsonable_encoder(dataset_config)
    
    # Return the updated configuration
    return dataset_config



@router.post("/dataset/settings/set-generation-relevance-threshold")
def set_generation_relevance_threshold(request: Request, dataset: str, threshold: float):
    dataset_config = request.app.dataset_config_collection.find_one({"dataset_name": dataset})
    if dataset_config is not None:
        request.app.dataset_config_collection.update_one(
            {"dataset_name": dataset}, 
            {"$set": {"relevance_threshold": threshold}}
        )
    else:
        request.app.dataset_config_collection.insert_one(
            {"dataset_name": dataset, "relevance_threshold": threshold}
        )
    return {
        "status": "success",
        "message": "Relevance threshold set successfully."
    }

@router.post("/dataset/settings/set-generation-relevance-threshold-v1", response_class=PrettyJSONResponse)
def set_generation_relevance_threshold(request: Request, dataset: str, threshold: float):
    # Check if the dataset configuration already exists
    dataset_config = request.app.dataset_config_collection.find_one({"dataset_name": dataset})
    if dataset_config is not None:
        # Update the existing configuration
        request.app.dataset_config_collection.update_one(
            {"dataset_name": dataset}, 
            {"$set": {"relevance_threshold": threshold}}
        )
    else:
        # Insert a new configuration if it doesn't exist
        request.app.dataset_config_collection.insert_one(
            {"dataset_name": dataset, "relevance_threshold": threshold}
        )

    # Re-fetch the updated configuration to ensure the latest data is returned
    updated_config = request.app.dataset_config_collection.find_one({"dataset_name": dataset})
    if updated_config:
        updated_config['_id'] = str(updated_config['_id'])

    # Convert the MongoDB document to a JSON serializable format
    json_compatible_config = jsonable_encoder(updated_config)
    
    # Return the updated configuration
    return json_compatible_config


@router.get("/dataset/settings/get-relevance-threshold")
def get_relevance_threshold(request: Request, dataset: str):
    dataset_config = request.app.dataset_config_collection.find_one({"dataset_name": dataset})
    if dataset_config is not None and "relevance_threshold" in dataset_config:
        return {"relevance_threshold": dataset_config["relevance_threshold"]}
    else:
        return {"relevance_threshold": None}

@router.get("/dataset/settings/get-relevance-policy")
def get_relevance_policy(request: Request, dataset: str):
    dataset_config = request.app.dataset_config_collection.find_one({"dataset_name": dataset})
    if dataset_config is not None:
        generation_policy = dataset_config.get("generation_policy", None)
        return {
            "generation_policy": generation_policy
        }
    else:
        return {"generation_policy": None }

@router.get("/dataset/settings/get-options-list-generation-policies")
def list_generation_policies():
    return {"generation_policies": ["generation-off", "rate-generation", "rate-generation-top-k"]}
