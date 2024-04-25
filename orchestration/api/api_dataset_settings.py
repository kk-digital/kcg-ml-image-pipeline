from fastapi import Request, HTTPException, APIRouter, Response, Query
from utility.minio import cmd
from datetime import datetime
from .api_utils import PrettyJSONResponse, ApiResponseHandlerV1, StandardSuccessResponseV1, StandardErrorResponseV1, ErrorCode, WasPresentResponse, DatasetResponse, SeqIdResponse, SeqIdDatasetResponse, SetRateResponse, ListFilePathResponse, RankinModelResponse, ListDatasetConfig, DatasetConfig


router = APIRouter()


# -------------------- Dataset generation policy -------------------------

@router.get("/dataset/settings/get-all-dataset-generation-policy")
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


@router.get("/dataset/settings/get-generation-policy")
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







# New apis


# -------------------- Dataset generation policy -------------------------

@router.get("/datasets/settings/get-all-dataset-generation-policy-v1",
            description="Get all dataset generation policy",
            tags=["datasets settings"],
            response_model=StandardSuccessResponseV1[ListDatasetConfig],  
            responses=ApiResponseHandlerV1.listErrors([422, 500]))
async def get_all_dataset_generation_policy_v1(request: Request):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        dataset_generation_policies = []
        # find
        items = request.app.dataset_config_collection.find({})
        if items is None:
            raise HTTPException(status_code=204)

        for item in items:
            # remove the auto generated field
            item.pop('_id', None)
            dataset_generation_policies.append(item)

        return response_handler.create_success_response_v1(
                response_data={"configs":dataset_generation_policies}, 
                http_status_code=200
            )
        
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )   

@router.get("/datasets/settings/get-generation-policy-v1",
            description="Get dataset generation policy by dataset name",
            tags=["datasets settings"],
            response_model=StandardSuccessResponseV1[DatasetConfig],  
            responses=ApiResponseHandlerV1.listErrors([422, 500]))
async def get_generation_policy_v1(request: Request, dataset: str):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        # find
        query = {"dataset_name": dataset}
        item = request.app.dataset_config_collection.find_one(query)
        if item is None or "generation_policy" not in item:
            raise HTTPException(status_code=204)

        return response_handler.create_success_response_v1(
                response_data=item, 
                http_status_code=200
            )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )   


@router.put("/datasets/settings/set-generation-policy-v1",
            description="set generation policy",
            tags=["datasets settings"],
            response_model=StandardSuccessResponseV1[DatasetConfig],  
            responses=ApiResponseHandlerV1.listErrors([422, 500]))
async def set_generation_policy_v1(request: Request, dataset, generation_policy='top-k'):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:     
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
            item = request.app.dataset_config_collection.find_one({"dataset_name": dataset})
        else:
            # Update the existing entry
            new_values = {"$set": {"last_update": date_now, "generation_policy": generation_policy}}
            request.app.dataset_config_collection.update_one(query, new_values)
            item = request.app.dataset_config_collection.find_one({"dataset_name": dataset})

        return response_handler.create_success_response_v1(
                response_data=item, 
                http_status_code=200
            )
    
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )   


@router.get("/datasets/settings/get-top-k-v1",
            description="get top k by dataset name",
            tags=["datasets settings"],
            response_model=StandardSuccessResponseV1[DatasetConfig],  
            responses=ApiResponseHandlerV1.listErrors([404,422, 500]))
async def get_top_k_v1(request: Request, dataset: str):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try: 
        # find
        query = {"dataset_name": dataset}
        item = request.app.dataset_config_collection.find_one(query)
        if item is None or "top_k" not in item:
            return response_handler.create_error_response_v1(
            error_code=ErrorCode.ELEMENT_NOT_FOUND,
            error_string=str(e),
            http_status_code=404
        ) 

        return response_handler.create_success_response_v1(
                response_data=item, 
                http_status_code=200
            )
    
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        ) 



@router.put("/datasets/settings/set-top-k-v1",
            description="set top k by dataset name",
            tags=["datasets settings"],
            response_model=StandardSuccessResponseV1[DatasetConfig],  
            responses=ApiResponseHandlerV1.listErrors([422, 500]))
async def set_top_k_v1(request: Request, dataset: str, top_k=0.1):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
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
            item = request.app.dataset_config_collection.find_one({"dataset_name": dataset})
        else:
            # Update the existing entry
            new_values = {"$set": {"last_update": date_now, "top_k": top_k}}
            request.app.dataset_config_collection.update_one(query, new_values)
            item = request.app.dataset_config_collection.find_one({"dataset_name": dataset})

        return response_handler.create_success_response_v1(
                response_data=item, 
                http_status_code=200
            )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        ) 

@router.post("/datasets/settings/set-option-v1",
            description="set generation plicy option by dataset name",
            tags=["datasets settings"],
            response_model=StandardSuccessResponseV1[DatasetConfig],  
            responses=ApiResponseHandlerV1.listErrors([422, 500]))
async def set_option_v1(request: Request, dataset: str, generation_policy: str):
    response_handler = await ApiResponseHandlerV1.createInstance(request)  

    try:    
        if generation_policy not in ["generation-off", "rate-generation", "rate-generation-top-k"]:
            raise HTTPException(status_code=400, detail="Invalid generation policy. Accepted values are 'generation-off', 'rate-generation', and 'rate-generation-top-k'.")

        dataset_config = request.app.dataset_config_collection.find_one({"dataset_name": dataset})
        if dataset_config is not None:
            request.app.dataset_config_collection.update_one(
                {"dataset_name": dataset}, 
                {"$set": {"generation_policy": generation_policy}}
            )
            
            item = request.app.dataset_config_collection.find_one({"dataset_name": dataset})
        else:
            request.app.dataset_config_collection.insert_one(
                {"dataset_name": dataset, "generation_policy": generation_policy}
            )
            item = request.app.dataset_config_collection.find_one({"dataset_name": dataset})

        return response_handler.create_success_response_v1(
                response_data=item, 
                http_status_code=200
            )

    
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        ) 



@router.post("/datasets/settings/set-generation-relevance-threshold-v1",
             description="set relevance threshold by dataset name",
            tags=["datasets settings"],
            response_model=StandardSuccessResponseV1[DatasetConfig],  
            responses=ApiResponseHandlerV1.listErrors([422, 500]))
async def set_generation_relevance_threshold_v1(request: Request, dataset: str, threshold: float):
    response_handler = await ApiResponseHandlerV1.createInstance(request)  

    try: 
        dataset_config = request.app.dataset_config_collection.find_one({"dataset_name": dataset})
        if dataset_config is not None:
            request.app.dataset_config_collection.update_one(
                {"dataset_name": dataset}, 
                {"$set": {"relevance_threshold": threshold}}
            )
            item = request.app.dataset_config_collection.find_one({"dataset_name": dataset})
        else:
            request.app.dataset_config_collection.insert_one(
                {"dataset_name": dataset, "relevance_threshold": threshold}
            )
            item = request.app.dataset_config_collection.find_one({"dataset_name": dataset})
        return response_handler.create_success_response_v1(
                response_data=item, 
                http_status_code=200
            )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        ) 

@router.get("/datasets/settings/get-relevance-threshold-v1",
            description="get relevance threshold by dataset name",
            tags=["datasets settings"],
            response_model=StandardSuccessResponseV1[DatasetConfig],  
            responses=ApiResponseHandlerV1.listErrors([422, 500]))
async def get_relevance_threshold_v1(request: Request, dataset: str):
    response_handler = await ApiResponseHandlerV1.createInstance(request)  

    try:
        dataset_config = request.app.dataset_config_collection.find_one({"dataset_name": dataset})
        item = dataset_config or {}
        
        return response_handler.create_success_response_v1(
            response_data=item,
            http_status_code=200
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )



@router.get("/datasets/settings/get-relevance-policy-v1",
            description="get relevance policy by dataset name",
            tags=["datasets settings"],
            response_model=StandardSuccessResponseV1[DatasetConfig],  
            responses=ApiResponseHandlerV1.listErrors([422, 500]))
async def get_relevance_policy(request: Request, dataset: str):
    response_handler = await ApiResponseHandlerV1.createInstance(request)  

    try:
        dataset_config = request.app.dataset_config_collection.find_one({"dataset_name": dataset})
        item = dataset_config or {}
        
        return response_handler.create_success_response_v1(
            response_data=item,
            http_status_code=200
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )   

@router.get("/dataset/settings/get-options-list-generation-policies")
def list_generation_policies():
    return {"generation_policies": ["generation-off", "rate-generation", "rate-generation-top-k"]}
