from fastapi import Request, APIRouter, HTTPException
from orchestration.api.mongo_schemas import RankingPercentile
from .api_utils import PrettyJSONResponse, validate_date_format, ApiResponseHandler, ErrorCode, StandardSuccessResponse, WasPresentResponse, TagsListResponse, VectorIndexUpdateRequest, TagsCategoryListResponse, TagResponse
from typing import List

router = APIRouter()


@router.post("/percentile/set-image-rank-percentile", description="Set image rank percentile")
def set_image_rank_percentile(request: Request, ranking_percentile: RankingPercentile):
    # check if exists
    query = {"image_hash": ranking_percentile.image_hash,
             "model_id": ranking_percentile.model_id}
    count = request.app.image_percentiles_collection.count_documents(query)
    if count > 0:
        raise HTTPException(status_code=409, detail="Score for specific model_id and image_hash already exists.")

    request.app.image_percentiles_collection.insert_one(ranking_percentile.to_dict())

    return True


@router.get("/percentile/get-image-rank-percentile-by-hash", description="Get image rank percentile by hash")
def get_image_rank_percentile_by_hash(request: Request, image_hash: str, model_id: int):
    # check if exist
    query = {"image_hash": image_hash,
             "model_id": model_id}

    item = request.app.image_percentiles_collection.find_one(query)
    if item is None:
        return None

    # remove the auto generated field
    item.pop('_id', None)

    return item


@router.get("/percentile/get-image-rank-percentiles-by-model-id",
            description="Get image rank percentiles by model id. Returns as descending order of percentiles")
def get_image_rank_percentiles_by_model_id(request: Request, model_id: int):
    # check if exist
    query = {"model_id": model_id}
    items = request.app.image_percentiles_collection.find(query).sort("percentile", -1)
    if items is None:
        return []

    percentile_data = []
    for item in items:
        # remove the auto generated field
        item.pop('_id', None)
        percentile_data.append(item)

    return percentile_data


@router.delete("/percentile/delete-image-rank-percentiles-by-model-id", description="Delete all image rank percentiles by model id.")
def delete_image_rank_percentiles_by_model_id(request: Request, model_id: int):
    # check if exist
    query = {"model_id": model_id}
    res = request.app.image_percentiles_collection.delete_many(query)
    print(res.deleted_count, " documents deleted.")

    return None


# Standardized APIs

@router.post("/percentile/image-rank-percentile",
             status_code=200,
             description="Set image rank percentile",
             response_model=StandardSuccessResponse[RankingPercentile],
             tags=["Percentile Score"],
             responses=ApiResponseHandler.listErrors([400, 422, 500]))
def set_image_rank_percentile(request: Request, ranking_percentile: RankingPercentile):
    response_handler = ApiResponseHandler(request)
    try:
        # Check if the percentile record already exists
        query = {"image_hash": ranking_percentile.image_hash, "model_id": ranking_percentile.model_id}
        count = request.app.image_percentiles_collection.count_documents(query)
        if count > 0:
            # If the record exists, return a conflict error response
            return response_handler.create_error_response(ErrorCode.INVALID_PARAMS, "Score for specific model_id and image_hash already exists.", 400)

        # Insert the new percentile record
        request.app.image_percentiles_collection.insert_one(ranking_percentile.dict())
        
        # Return a success response with the inserted percentile data
        return response_handler.create_success_response(ranking_percentile.dict(), 200)
    except Exception as e:
        # In case of an exception, return an internal server error response
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal Server Error", 500)



@router.get("/percentile/image-rank-percentile-by-hash",
             status_code=200,
             description="Get image rank percentile by hash",
             response_model=StandardSuccessResponse[RankingPercentile],
             tags=["Percentile Score"],
             responses=ApiResponseHandler.listErrors([400, 422, 500]))
def get_image_rank_percentile_by_hash(request: Request, image_hash: str, model_id: int):
    response_handler = ApiResponseHandler(request)
    try:
        # Check if the percentile record exists
        query = {"image_hash": image_hash, "model_id": model_id}
        item = request.app.image_percentiles_collection.find_one(query)
        
        if not item:
            # If the record does not exist, return a not found error response
            return response_handler.create_error_response(ErrorCode.INVALID_PARAMS, "Image rank percentile data not found", 400)

        # Remove the auto-generated '_id' field
        item.pop('_id', None)

        # Return a success response with the percentile data
        return response_handler.create_success_response(item, 200)
    except Exception as e:
        # In case of an exception, return an internal server error response
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal Server Error", 500)
    


@router.get("/percentile/image-rank-percentiles-by-model-id",
            response_model=StandardSuccessResponse[List[RankingPercentile]],
            tags=["Percentile Score"],
            description="Get image rank percentiles by model id. Returns as descending order of percentiles",
            responses=ApiResponseHandler.listErrors([404, 422, 500]))
def get_image_rank_percentiles_by_model_id(request: Request, model_id: int):
    api_response_handler = ApiResponseHandler(request)
    try:
        query = {"model_id": model_id}
        items_cursor = request.app.image_percentiles_collection.find(query).sort("percentile", -1)
        items = list(items_cursor)

        if not items:
            return api_response_handler.create_error_response(ErrorCode.ELEMENT_NOT_FOUND, "No percentiles found for the given model_id", 404)

        # Prepare the documents, ensuring removal of '_id'
        for item in items:
            item.pop('_id', None)
        
        return api_response_handler.create_success_response(items, 200)
    except Exception as e:
        return api_response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal Server Error", 500)


@router.delete("/percentile/image-rank-percentiles-by-model-id/{model_id}",
               tags=["percentile score"],
               response_model=StandardSuccessResponse[WasPresentResponse],
               responses=ApiResponseHandler.listErrors([404, 422]),
               description="Delete all image rank percentiles by model id.")
async def delete_image_rank_percentiles_by_model_id(request: Request, model_id: int):
    response_handler = ApiResponseHandler(request)
    query = {"model_id": model_id}
    
    # Perform the deletion operation
    res = request.app.image_percentiles_collection.delete_many(query)
    
    # Prepare the response based on whether any documents were deleted
    was_present = res.deleted_count > 0
    
    if was_present:
        # If documents were deleted, return success with wasPresent: true
        return response_handler.create_success_response({"wasPresent": was_present}, 200)
    else:
        # If no documents were deleted, it might indicate the model_id didn't match any documents
        return response_handler.create_error_response(ErrorCode.ELEMENT_NOT_FOUND, "No percentiles found for the given model_id to delete", 404)