from fastapi import Request, HTTPException, APIRouter, Response, Query
from utility.minio import cmd
from datetime import datetime
from .api_utils import PrettyJSONResponse, ApiResponseHandlerV1, StandardSuccessResponseV1, ErrorCode, ListDatasetConfig, ResponsePolicies, ResponseDatasetConfig, DatasetConfig


router = APIRouter()


# -------------------- Dataset generation policy -------------------------

@router.get("/dataset/settings/get-all-dataset-generation-policy", tags = ['deprecated3'], description= "changed wtih /datasets/settings/get-all-dataset-config")
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


@router.get("/dataset/settings/get-generation-policy",tags = ['deprecated3'], description= "changed wtih /datasets/settings/get-dataset-config" )
def get_generation_policy(request: Request, dataset: str):
    # find
    query = {"dataset_name": dataset}
    item = request.app.dataset_config_collection.find_one(query)
    if item is None or "generation_policy" not in item:
        raise HTTPException(status_code=204)

    return item["generation_policy"]


@router.put("/dataset/settings/set-generation-policy", tags = ['deprecated3'], description= "changed wtih /datasets/settings/set-config" )
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


@router.get("/dataset/settings/get-top-k", tags = ['deprecated3'], description= "changed wtih /datasets/settings/get-dataset-config")
def get_top_k(request: Request, dataset: str):
    # find
    query = {"dataset_name": dataset}
    item = request.app.dataset_config_collection.find_one(query)
    if item is None or "top_k" not in item:
        raise HTTPException(status_code=204)

    return item["top_k"]


@router.put("/dataset/settings/set-top-k", tags = ['deprecated3'], description= "changed wtih /datasets/settings/set-config" )
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


@router.post("/dataset/settings/set-option", tags = ['deprecated3'], description= "changed wtih /datasets/settings/set-config" )
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


@router.post("/dataset/settings/set-generation-relevance-threshold", tags = ['deprecated3'], description= "changed wtih /datasets/settings/set-config" )
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


@router.get("/dataset/settings/get-relevance-threshold", tags = ['deprecated3'], description= "changed wtih /datasets/settings/get-dataset-config")
def get_relevance_threshold(request: Request, dataset: str):
    dataset_config = request.app.dataset_config_collection.find_one({"dataset_name": dataset})
    if dataset_config is not None and "relevance_threshold" in dataset_config:
        return {"relevance_threshold": dataset_config["relevance_threshold"]}
    else:
        return {"relevance_threshold": None}

@router.get("/dataset/settings/get-relevance-policy",tags = ['deprecated3'], description= "changed wtih /datasets/settings/get-dataset-config")
def get_relevance_policy(request: Request, dataset: str):
    dataset_config = request.app.dataset_config_collection.find_one({"dataset_name": dataset})
    if dataset_config is not None:
        generation_policy = dataset_config.get("generation_policy", None)
        return {
            "generation_policy": generation_policy
        }
    else:
        return {"generation_policy": None }

@router.get("/dataset/settings/get-options-list-generation-policies",tags = ['deprecated3'], description= "changed wtih /datasets/settings/list-dataset-generation-policies" )
def list_generation_policies():
    return {"generation_policies": ["generation-off", "rate-generation", "rate-generation-top-k"]}







# New apis
   

@router.put("/datasets/settings/set-config",
    description="Set or create the configuration of a dataset. Updates provided properties and leaves others unchanged or creates a new configuration if none exists.",
    tags=["dataset"],
    response_model=StandardSuccessResponseV1[ResponseDatasetConfig],
    responses=ApiResponseHandlerV1.listErrors([422, 500]),
)
async def set_dataset_config(request: Request, config: DatasetConfig):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        # Verify if the dataset exists in MinIO
        dataset_exists = cmd.get_list_of_objects(request.app.minio_client, "datasets")
        dataset_path = f"{config.dataset_name}"

        if dataset_path not in dataset_exists:
            # Return 422 error if the dataset does not exist in MinIO
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string=f"Dataset '{config.dataset_name}' does not exist in MinIO.",
                http_status_code=422,
            )
        
        query = {"dataset_name": config.dataset_name}
        update_values = config.dict(exclude_unset=True)
        update_values["last_update"] = datetime.utcnow().isoformat()

        # Use update_one with upsert=True to update or insert if not present
        result = request.app.dataset_config_collection.update_one(
            {"dataset_name": config.dataset_name}, 
            {"$set": update_values}, 
            upsert=True
        )

        # Fetch and return the updated or new dataset configuration
        updated_item = request.app.dataset_config_collection.find_one({"dataset_name": config.dataset_name})
        if updated_item:
            updated_item.pop("_id", None)  # Remove MongoDB ObjectId

        return response_handler.create_success_response_v1(
            response_data=updated_item,
            http_status_code=200,
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )

    

@router.get("/datasets/settings/get-dataset-config",
    description="Get the configuration of a dataset. If a property is not set, it returns null.",
    tags=["dataset"],
    response_model=StandardSuccessResponseV1[ResponseDatasetConfig],
    responses=ApiResponseHandlerV1.listErrors([404, 422, 500]),
)
async def get_dataset_config(request: Request, dataset: str = Query(...)):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        # Check if the dataset exists in MinIO
        objects = cmd.get_list_of_objects(request.app.minio_client, "datasets")
        print(objects)
        dataset_path = f'{dataset}'
        
        if dataset_path not in objects:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string=f"Dataset '{dataset}' does not exist.",
                http_status_code=422,
            )

        dataset_exists = request.app.datasets_collection.find_one({"dataset_name": dataset})
        if not dataset_exists:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string=f"Dataset '{dataset}' does not exist in the database.",
                http_status_code=404,
            )
        
        item = request.app.dataset_config_collection.find_one({"dataset_name": dataset})

        # Define a default configuration that specifies all properties
        default_config = {
            "dataset_name": dataset,
            "last_update": None,
            "dataset_rate": None,
            "relevance_model": None,
            "ranking_model": None,
            "hourly_limit": None,
            "top_k": None,
            "generation_policy": None,
            "relevance_threshold": None,
        }

        if item:
            item.pop("_id", None)  # Remove MongoDB ObjectId
            try:
                # Convert 'last_update' to string, if it exists
                item["last_update"] = str(item["last_update"]) if item.get("last_update") else None
            except Exception:
                item["last_update"] = None
            
            # Merge with default_config to ensure all keys are present
            full_config = {**default_config, **item}
        else:
            # Use default config if no item found
            full_config = default_config

        return response_handler.create_success_response_v1(
            response_data=full_config, 
            http_status_code=200,
        )
    
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )


    

@router.get("/datasets/settings/get-all-dataset-config",
            description="Get configurations for all datasets. If a property is not set, it returns null.",
            response_model=StandardSuccessResponseV1[ListDatasetConfig],
            tags=["dataset"],
            responses=ApiResponseHandlerV1.listErrors([422, 500]),
        )
async def get_all_dataset_config(request: Request):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        dataset_configs = []

        # Fetch all datasets from the datasets_collection
        all_datasets = list(request.app.datasets_collection.find({}, {"dataset_name": 1}))

        # Default configuration with None values for unset properties
        default_config = {
            "dataset_name": None,  
            "last_update": None,
            "dataset_rate": None,
            "relevance_model": None,
            "ranking_model": None,
            "hourly_limit": None,
            "top_k": None,
            "generation_policy": None,
            "relevance_threshold": None,
        }

        # Iterate through all datasets to fetch or default their configs
        for dataset in all_datasets:
            dataset_name = dataset["dataset_name"]
            # Try to find a specific configuration for this dataset
            item = request.app.dataset_config_collection.find_one({"dataset_name": dataset_name})

            if item:
                item.pop("_id", None)  # Remove MongoDB ObjectId
                # Convert 'last_update' to string using a try-except block
                try:
                    item["last_update"] = str(item["last_update"])
                except Exception:
                    item["last_update"] = None  # Set to None if conversion fails
            else:
                # No specific configuration, use a copy of default and update the dataset name
                item = default_config.copy()
                item["dataset_name"] = dataset_name

            # Merge with default_config to ensure all keys are present
            full_config = {**default_config, **item}
            dataset_configs.append(full_config)

        return response_handler.create_success_response_v1(
            response_data={"configs": dataset_configs},
            http_status_code=200,
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )


@router.get("/datasets/settings/list-dataset-generation-policies",
            description="Get all generation policy",
            tags=["dataset"],
            response_model=StandardSuccessResponseV1[ResponsePolicies],  
            responses=ApiResponseHandlerV1.listErrors([422]))
async def list_generation_policies(request: Request):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    return response_handler.create_success_response_v1(
                response_data={"generation_policies": ["generation-off", "rate-generation", "rate-generation-top-k", "independent-approx-v1-top-k"]}, 
                http_status_code=200
            )

