from fastapi import Request, APIRouter, HTTPException
from orchestration.api.mongo_schemas import RankingPercentile, ResponseRankingPercentile
from .api_utils import PrettyJSONResponse, validate_date_format, ApiResponseHandler, ErrorCode, StandardSuccessResponse, WasPresentResponse, ApiResponseHandlerV1, StandardSuccessResponseV1
from typing import List

router = APIRouter()


@router.post("/percentile/set-image-rank-percentile", tags = ['deprecated3'], description= "changed with /image-scores/percentiles/set-image-rank-percentile"
)
def set_image_rank_percentile(request: Request, ranking_percentile: RankingPercentile):
    # check if exists
    query = {"image_hash": ranking_percentile.image_hash,
             "model_id": ranking_percentile.model_id}
    count = request.app.image_percentiles_collection.count_documents(query)
    if count > 0:
        raise HTTPException(status_code=409, detail="Score for specific model_id and image_hash already exists.")

    request.app.image_percentiles_collection.insert_one(ranking_percentile.to_dict())

    return True


@router.get("/percentile/get-image-rank-percentile-by-hash", tags = ['deprecated3'], description= "changed with /image-scores/percentiles/get-image-rank-percentile"
)
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
            tags = ['deprecated3'], description= "changed with /image-scores/percentiles/list-image-rank-percentiles-by-model-id"
)
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


@router.delete("/percentile/delete-image-rank-percentiles-by-model-id",tags = ['deprecated3'], description= "remove scores according model id"
)
def delete_image_rank_percentiles_by_model_id(request: Request, model_id: int):
    # check if exist
    query = {"model_id": model_id}
    res = request.app.image_percentiles_collection.delete_many(query)
    print(res.deleted_count, " documents deleted.")

    return None


# Standardized APIs

@router.post("/image-scores/percentiles/set-image-rank-percentile",
             status_code=201,
             description="Sets the rank percentile of an image. The score can only be set one time per image/model combination",
             response_model=StandardSuccessResponseV1[RankingPercentile],
             tags=["image scores"],
             responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
@router.post("/percentile/image-rank-percentile",
             status_code=201,
             description="deprecated: use /image-scores/percentiles/set-image-rank-percentile ",
             response_model=StandardSuccessResponseV1[RankingPercentile],
             tags=["deprecated2"],
             responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def set_image_rank_percentile(request: Request, ranking_percentile: RankingPercentile):
    response_handler = ApiResponseHandlerV1(request)
    try:
        # Check if the percentile record already exists
        query = {"image_hash": ranking_percentile.image_hash, "model_id": ranking_percentile.model_id}
        count = request.app.image_percentiles_collection.count_documents(query)
        if count > 0:
            # If the record exists, return a conflict error response
            return response_handler.create_error_response_v1(error_code=ErrorCode.INVALID_PARAMS, error_string="Score for specific model_id and image_hash already exists.", http_status_code=400)

        # Insert the new percentile record
        request.app.image_percentiles_collection.insert_one(ranking_percentile.dict())
        
        # Return a success response with the inserted percentile data
        return response_handler.create_success_response_v1(response_data=ranking_percentile.dict(), http_status_code=201)
    except Exception as e:
        # In case of an exception, return an internal server error response
        return response_handler.create_error_response_v1(error_code=ErrorCode.OTHER_ERROR, error_string="Internal Server Error", http_status_code=500)


@router.get("/image-scores/percentiles/get-image-rank-percentile", 
             status_code=200,
             description="Get image rank percentile by hash",
             response_model=StandardSuccessResponseV1[RankingPercentile],
             tags=["image scores"],
             responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
@router.get("/percentile/image-rank-percentile-by-hash",
             status_code=200,
             description="deprecated: use /image-scores/percentiles/get-image-rank-percentile ",
             response_model=StandardSuccessResponseV1[RankingPercentile],
             tags=["deprecated2"],
             responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def get_image_rank_percentile_by_hash(request: Request, image_hash: str, model_id: int):
    response_handler = ApiResponseHandlerV1(request)
    try:
        # Check if the percentile record exists
        query = {"image_hash": image_hash, "model_id": model_id}
        item = request.app.image_percentiles_collection.find_one(query)
      

        # Remove the auto-generated '_id' field
        item.pop('_id', None)

        # Return a success response with the percentile data
        return response_handler.create_success_response_v1(response_data=item, http_status_code=200)
    except Exception as e:
        # In case of an exception, return an internal server error response
        return response_handler.create_error_response_v1(error_code=ErrorCode.OTHER_ERROR, error_string="Internal Server Error", http_status_code=500)
    

@router.get("/image-scores/percentiles/list-image-rank-percentiles-by-model-id",
            response_model=StandardSuccessResponseV1[ResponseRankingPercentile],
            tags=["image scores"],
            description="Get image rank percentiles by model id. Returns as descending order of percentiles",
            responses=ApiResponseHandlerV1.listErrors([422, 500]))
@router.get("/percentile/image-rank-percentiles-by-model-id",
            response_model=StandardSuccessResponseV1[ResponseRankingPercentile],
            tags=["deprecated2"],
            description="deprecated: use /image-scores/percentiles/list-image-rank-percentiles-by-model-id ",
            responses=ApiResponseHandlerV1.listErrors([422, 500]))
def get_image_rank_percentiles_by_model_id(request: Request, model_id: int):
    api_response_handler = ApiResponseHandlerV1(request)
    try:
        query = {"model_id": model_id}
        items_cursor = request.app.image_percentiles_collection.find(query).sort("percentile", -1)
        items = list(items_cursor)

        # Prepare the documents, ensuring removal of '_id'
        for item in items:
            item.pop('_id', None)
        
        return api_response_handler.create_success_response_v1(response_data={'percentiles': items}, http_status_code=200)
    except Exception as e:
        return api_response_handler.create_error_response_v1(error_code=ErrorCode.OTHER_ERROR, error_string="Internal Server Error", http_status_code=500)

