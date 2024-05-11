from fastapi import Request, APIRouter, HTTPException
from orchestration.api.mongo_schemas import RankingResidual, ResponseRankingResidual
from .api_utils import ApiResponseHandler, ErrorCode, StandardSuccessResponse, WasPresentResponse, ApiResponseHandlerV1, StandardSuccessResponseV1

router = APIRouter()


@router.post("/residual/set-image-rank-residual", description="Set image rank residual")
def set_image_rank_residual(request: Request, ranking_residual: RankingResidual):
    # check if exists
    query = {"image_hash": ranking_residual.image_hash,
             "model_id": ranking_residual.model_id}
    count = request.app.image_residuals_collection.count_documents(query)
    if count > 0:
        raise HTTPException(status_code=409, detail="Residual for specific model_id and image_hash already exists.")

    request.app.image_residuals_collection.insert_one(ranking_residual.to_dict())

    return True

@router.post("/image-scores/residuals/set-image-rank-residual",
             tags = ["residual"],
             status_code=201,
             description="Sets the rank residual of an image. The score can only be set one time per image/model combination",
             response_model=StandardSuccessResponseV1[RankingResidual],
             responses=ApiResponseHandlerV1.listErrors([400, 422]))
@router.post("/residual/image-rank-residual",
             tags = ["residual"],
             status_code=201,
             description="Sets the rank residual of an image. The score can only be set one time per image/model combination",
             response_model=StandardSuccessResponseV1[ResponseRankingResidual],
             responses=ApiResponseHandlerV1.listErrors([400, 422]))
async def set_image_rank_residual(request: Request, ranking_residual: RankingResidual):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    query = {"image_hash": ranking_residual.image_hash, "model_id": ranking_residual.model_id}
    count = request.app.image_residuals_collection.count_documents(query)
    
    if count > 0:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="Residual for specific model_id and image_hash already exists.",
            http_status_code=400
        )

    request.app.image_residuals_collection.insert_one(ranking_residual.dict())
    return api_response_handler.create_success_response_v1(
        response_data=ranking_residual.dict(),
        http_status_code=201
    )

@router.get("/residual/get-image-rank-residual-by-hash", description="Get image rank residual by hash")
def get_image_rank_residual_by_hash(request: Request, image_hash: str, model_id: int):
    # check if exist
    query = {"image_hash": image_hash,
             "model_id": model_id}

    item = request.app.image_residuals_collection.find_one(query)
    if item is None:
        return None

    # remove the auto generated field
    item.pop('_id', None)

    return item

@router.get("/image-scores/residuals/get-image-rank-residual", 
            description="Get image rank residual by hash",
            tags = ["residual"],
            status_code=200,
            response_model=StandardSuccessResponseV1[RankingResidual],
            responses=ApiResponseHandlerV1.listErrors([400, 422]))
@router.get("/residual/image-rank-residual-by-hash", 
            description="Get image rank residual by hash",
            tags = ["residual"],
            status_code=200,
            response_model=StandardSuccessResponseV1[RankingResidual],
            responses=ApiResponseHandlerV1.listErrors([400, 422]))
def get_image_rank_residual_by_hash(request: Request, image_hash: str, model_id: str):
    api_response_handler = ApiResponseHandlerV1(request)
    query = {"image_hash": image_hash, "model_id": model_id}
    item = request.app.image_residuals_collection.find_one(query)
    
    if item is None:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="No residual found for specified model_id and image_hash.",
            http_status_code=404
        )

    item.pop('_id', None)
    return api_response_handler.create_success_response_v1(
        response_data=item,
        http_status_code=200
    )


@router.get("/residual/get-image-rank-residuals-by-model-id",
            description="Get image rank residuals by model id. Returns as descending order of residual")
def get_image_rank_residuals_by_model_id(request: Request, model_id: int):
    # check if exist
    query = {"model_id": model_id}
    items = request.app.image_residuals_collection.find(query).sort("residual", -1)
    if items is None:
        return []
    residual_data = []
    for item in items:
        # remove the auto generated field
        item.pop('_id', None)
        residual_data.append(item)

    return residual_data


@router.get("/image-scores/residuals/list-image-rank-residuals-by-model-id",
            description="Get image rank residuals by model id. Returns as descending order of residual",
            tags = ["residual"],
            status_code=200,
            response_model=StandardSuccessResponseV1[ResponseRankingResidual],
            responses=ApiResponseHandlerV1.listErrors([422]))
@router.get("/residual/image-rank-residuals-by-model-id",
            description="Get image rank residuals by model id. Returns as descending order of residual",
            tags = ["residual"],
            status_code=200,
            response_model=StandardSuccessResponseV1[RankingResidual],
            responses=ApiResponseHandlerV1.listErrors([422]))
def get_image_rank_residuals_by_model_id(request: Request, model_id: str):
    api_response_handler = ApiResponseHandlerV1(request)
    query = {"model_id": model_id}
    items = list(request.app.image_residuals_collection.find(query).sort("residual", -1))
    

    for item in items:
        item.pop('_id', None)
    return api_response_handler.create_success_response_v1(
        response_data={'residuals': items},
        http_status_code=200
    )


@router.delete("/residual/delete-image-rank-residuals-by-model-id", description="Delete all image rank residuals by model id.")
def delete_image_rank_residuals_by_model_id(request: Request, model_id: int):
    # check if exist
    query = {"model_id": model_id}
    res = request.app.image_residuals_collection.delete_many(query)
    print(res.deleted_count, " documents deleted.")

    return None


@router.delete("/image-scores/residuals/delete-all-image-rank-residuals-by-model-id", 
               description="Delete all image rank residuals by model id.",
               tags = ["residual"],
               status_code=200,
               response_model=StandardSuccessResponseV1[WasPresentResponse],
               responses=ApiResponseHandlerV1.listErrors([422]))
@router.delete("/residual/image-rank-residuals-by-model-id", 
               description="Delete all image rank residuals by model id.",
               tags = ["residual"],
               status_code=200,
               response_model=StandardSuccessResponseV1[WasPresentResponse],
               responses=ApiResponseHandlerV1.listErrors([422]))
def delete_image_rank_residuals_by_model_id(request: Request, model_id: str):
    api_response_handler = ApiResponseHandlerV1(request)
    query = {"model_id": model_id}
    res = request.app.image_residuals_collection.delete_many(query)
    
    was_present = res.deleted_count > 0
    if was_present:
        return api_response_handler.create_success_delete_response_v1(
            True,
            http_status_code=200
        )
    else:
        return api_response_handler.create_success_delete_response_v1(
            False,
            http_status_code=200
        )
