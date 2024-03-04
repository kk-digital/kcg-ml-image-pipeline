from fastapi import Request, APIRouter, HTTPException
from orchestration.api.mongo_schemas import RankingResidualPercentile
from .api_utils import PrettyJSONResponse, validate_date_format, ApiResponseHandler, ErrorCode, StandardSuccessResponse, WasPresentResponse, TagsListResponse, VectorIndexUpdateRequest, TagsCategoryListResponse, TagResponse, TagCountResponse, StandardSuccessResponseV1, ApiResponseHandlerV1, TagIdResponse, ListImageTag
 

router = APIRouter()


@router.post("/residual-percentile/set-image-rank-residual-percentile", description="Set image rank residual-percentile")
def set_image_rank_residual_percentile(request: Request, ranking_residual_percentile: RankingResidualPercentile):
    # check if exists
    query = {"image_hash": ranking_residual_percentile.image_hash,
             "model_id": ranking_residual_percentile.model_id}
    count = request.app.image_residual_percentiles_collection.count_documents(query)
    if count > 0:
        raise HTTPException(status_code=409, detail="Residual Percentile for specific model_id and image_hash already exists.")

    request.app.image_residual_percentiles_collection.insert_one(ranking_residual_percentile.to_dict())

    return True


@router.get("/residual-percentile/get-image-rank-residual-percentile-by-hash", description="Get image rank residual_percentile by hash")
def get_image_rank_residual_percentile_by_hash(request: Request, image_hash: str, model_id: int):
    # check if exist
    query = {"image_hash": image_hash,
             "model_id": model_id}

    item = request.app.image_residual_percentiles_collection.find_one(query)
    if item is None:
        return None

    # remove the auto generated field
    item.pop('_id', None)

    return item


@router.get("/residual-percentile/get-image-rank-residual-percentiles-by-model-id",
            description="Get image rank residual percentiles by model id. Returns as descending order of residual percentile")
def get_image_rank_residual_percentiles_by_model_id(request: Request, model_id: int):
    # check if exist
    query = {"model_id": model_id}
    items = request.app.image_residual_percentiles_collection.find(query).sort("residual_percentile", -1)
    if items is None:
        return []

    residual_percentile_data = []
    for item in items:
        # remove the auto generated field
        item.pop('_id', None)
        residual_percentile_data.append(item)

    return residual_percentile_data


@router.delete("/residual-percentile/delete-image-rank-residual-percentiles-by-model-id", description="Delete all image rank residual percentiles by model id.")
def delete_image_rank_residual_percentiles_by_model_id(request: Request, model_id: int):
    # check if exist
    query = {"model_id": model_id}
    res = request.app.image_residual_percentiles_collection.delete_many(query)
    print(res.deleted_count, " documents deleted.")

    return None



# New APIs

@router.post("/residual-percentile/set-image-rank-residual-percentile-v1", 
             description="Set image rank residual-percentile",
             status_code=200,
             responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def set_image_rank_residual_percentile_v1(request: Request, ranking_residual_percentile: RankingResidualPercentile):
    response_handler = ApiResponseHandlerV1(request)
    try:
        # check if exists
        query = {"image_hash": ranking_residual_percentile.image_hash,
                 "model_id": ranking_residual_percentile.model_id}
        count = request.app.image_residual_percentiles_collection.count_documents(query)
        if count > 0:
            # Use ApiResponseHandlerV1 to create an error response
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,  # Define this error code in your ErrorCode enum
                error_string="Residual Percentile for specific model_id and image_hash already exists.",
                http_status_code=400,
            )

        request.app.image_residual_percentiles_collection.insert_one(ranking_residual_percentile.dict())

        # Use ApiResponseHandlerV1 to create a success response
        return response_handler.create_success_response_v1(
            response_data={"message": "Residual Percentile set successfully."},
            http_status_code=200,
        )
    except Exception as e:
        # Log the exception details here, if necessary
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )


@router.get("/residual-percentile/get-image-rank-residual-percentile-by-hash-v1", 
            description="Get image rank residual_percentile by hash",
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([400, 404, 500]))
def get_image_rank_residual_percentile_by_hash_v1(request: Request, image_hash: str, model_id: int):
    response_handler = ApiResponseHandlerV1(request)
    try:
        query = {"image_hash": image_hash, "model_id": model_id}
        item = request.app.image_residual_percentiles_collection.find_one(query)
        if item is None:
            # Use ApiResponseHandlerV1 to create an error response if the item doesn't exist
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="Residual percentile not found for the given image hash and model ID.",
                http_status_code=404,
            )

        # Remove the auto-generated '_id' field from MongoDB response
        item.pop('_id', None)

        # Use ApiResponseHandlerV1 to create a success response with the found item
        return response_handler.create_success_response_v1(
            response_data=item,
            http_status_code=200,
        )
    except Exception as e:
        # Handle exceptions and return an error response
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )


@router.get("/residual-percentile/get-image-rank-residual-percentiles-by-model-id-v1", 
            description="Get image rank residual percentiles by model id. Returns as descending order of residual percentile",
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([400, 500]))
def get_image_rank_residual_percentiles_by_model_id_v1(request: Request, model_id: int):
    response_handler = ApiResponseHandlerV1(request)
    try:
        query = {"model_id": model_id}
        items = request.app.image_residual_percentiles_collection.find(query).sort("residual_percentile", -1)

        residual_percentile_data = []
        for item in items:
            # remove the auto-generated '_id' field
            item.pop('_id', None)
            residual_percentile_data.append(item)

        return response_handler.create_success_response_v1(
            response_data=residual_percentile_data,
            http_status_code=200,
        )
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )
    
@router.delete("/residual-percentile/delete-image-rank-residual-percentiles-by-model-id-v1", 
               description="Delete all image rank residual percentiles by model id.",
               status_code=200,
               responses=ApiResponseHandlerV1.listErrors([400, 500]))
def delete_image_rank_residual_percentiles_by_model_id_v1(request: Request, model_id: int):
    response_handler = ApiResponseHandlerV1(request)
    try:
        query = {"model_id": model_id}
        result = request.app.image_residual_percentiles_collection.delete_many(query)

        if result.deleted_count == 0:
            return response_handler.create_success_response_v1(
                response_data={"message": "No documents found for deletion."},
                http_status_code=200,
            )

        return response_handler.create_success_response_v1(
            response_data={"message": f"{result.deleted_count} documents deleted."},
            http_status_code=200,
        )
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )    