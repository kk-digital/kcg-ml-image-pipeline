from fastapi import Request, APIRouter, HTTPException
from orchestration.api.mongo_schemas import RankingSigmaScore, ResponseRankingSigmaScore
from .api_utils import PrettyJSONResponse, ApiResponseHandler, ErrorCode, StandardSuccessResponse, WasPresentResponse, ApiResponseHandlerV1, StandardSuccessResponseV1
from typing import List

router = APIRouter()


@router.post("/sigma-score/set-image-rank-sigma-score", description="Set image rank sigma_score")
def set_image_rank_sigma_score(request: Request, ranking_sigma_score: RankingSigmaScore):
    # check if exists
    query = {"image_hash": ranking_sigma_score.image_hash,
             "model_id": ranking_sigma_score.model_id}
    count = request.app.image_sigma_scores_collection.count_documents(query)
    if count > 0:
        raise HTTPException(status_code=409, detail="Score for specific model_id and image_hash already exists.")

    request.app.image_sigma_scores_collection.insert_one(ranking_sigma_score.to_dict())

    return True


@router.get("/sigma-score/get-image-rank-sigma-score-by-hash", description="Get image rank sigma_score by hash")
def get_image_rank_sigma_score_by_hash(request: Request, image_hash: str, model_id: int):
    # check if exist
    query = {"image_hash": image_hash,
             "model_id": model_id}

    item = request.app.image_sigma_scores_collection.find_one(query)
    if item is None:
        print("Image rank sigma_score data not found")

    # remove the auto generated field
    item.pop('_id', None)

    return item


@router.get("/sigma-score/get-image-rank-sigma-scores-by-model-id",
            description="Get image rank sigma_scores by model id. Returns as descending order of sigma_scores")
def get_image_rank_sigma_scores_by_model_id(request: Request, model_id: int):
    # check if exist
    query = {"model_id": model_id}
    items = request.app.image_sigma_scores_collection.find(query).sort("sigma_score", -1)
    if items is None:
        return []

    sigma_score_data = []
    for item in items:
        # remove the auto generated field
        item.pop('_id', None)
        sigma_score_data.append(item)

    return sigma_score_data


@router.delete("/sigma-score/delete-image-rank-sigma-scores-by-model-id", description="Delete all image rank sigma_scores by model id.")
def delete_image_rank_sigma_scores_by_model_id(request: Request, model_id: int):
    # check if exist
    query = {"model_id": model_id}
    res = request.app.image_sigma_scores_collection.delete_many(query)
    print(res.deleted_count, " documents deleted.")

    return None


# Standardized APIs

@router.post("/image-scores/sigma-scores/set-image-rank-sigma-score",
             status_code=201,  
             description="Sets the rank sigma_score of an image. The score can only be set one time per image/model combination",
             tags=["sigma score"],
             response_model=StandardSuccessResponseV1[RankingSigmaScore],
             responses=ApiResponseHandlerV1.listErrors([400, 422]))
@router.post("/sigma-score/image-rank-sigma-score",
             status_code=201,  
             description="Sets the rank sigma_score of an image. The score can only be set one time per image/model combination",
             tags=["sigma score"],
             response_model=StandardSuccessResponseV1[RankingSigmaScore],
             responses=ApiResponseHandlerV1.listErrors([400, 422]))
def set_image_rank_sigma_score(request: Request, ranking_sigma_score: RankingSigmaScore):
    response_handler = ApiResponseHandlerV1(request)
    query = {"image_hash": ranking_sigma_score.image_hash, "model_id": ranking_sigma_score.model_id}
    count = request.app.image_sigma_scores_collection.count_documents(query)

    if count > 0:
        # If a sigma score already exists for the given image_hash and model_id, return an error response
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS, 
            error_string="Score for specific model_id and image_hash already exists.", 
            http_status_code=400)

    # If the sigma score does not exist, insert the new score
    request.app.image_sigma_scores_collection.insert_one(ranking_sigma_score.dict())

    # Return a success response with the inserted sigma score data
    return response_handler.create_success_response_v1(response_data=ranking_sigma_score.dict(), http_status_code=201)


@router.get("/image-scores/sigma-scores/get-image-rank-sigma-score", 
            status_code=200,
            description="Get image rank sigma_score by hash",
            tags=["sigma score"],
            response_model=StandardSuccessResponseV1[RankingSigmaScore],  
            responses=ApiResponseHandlerV1.listErrors([422, 500]))
@router.get("/sigma-score/image-rank-sigma-score-by-hash", 
            status_code=200,
            description="Get image rank sigma_score by hash",
            tags=["sigma score"],
            response_model=StandardSuccessResponseV1[RankingSigmaScore],  
            responses=ApiResponseHandlerV1.listErrors([422, 500]))
def get_image_rank_sigma_score_by_hash(request: Request, image_hash: str, model_id: int):
    response_handler = ApiResponseHandlerV1(request)
    query = {"image_hash": image_hash, "model_id": model_id}

    item = request.app.image_sigma_scores_collection.find_one(query)
    if not item:
        # If no sigma score found for the given image_hash and model_id, return an error response
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.ELEMENT_NOT_FOUND, 
            error_string="Image rank sigma_score data not found",
            http_status_code=404)

    # Remove the auto-generated '_id' field from the MongoDB document before returning
    item.pop('_id', None)

    # Return a success response with the found sigma score data
    return response_handler.create_success_response_v1(response_data=item, http_status_code=200)

@router.get("/image-scores/sigma-scores/list-image-rank-sigma-scores-by-model-id",
            response_model=StandardSuccessResponseV1[ResponseRankingSigmaScore],
            tags=["sigma score"],
            description="Get image rank sigma_scores by model id. Returns as descending order of sigma_scores",
            responses=ApiResponseHandlerV1.listErrors([422, 500]))
@router.get("/sigma-score/image-rank-sigma-scores-by-model-id",
            response_model=StandardSuccessResponseV1[ResponseRankingSigmaScore],
            tags=["sigma score"],
            description="Get image rank sigma_scores by model id. Returns as descending order of sigma_scores",
            responses=ApiResponseHandlerV1.listErrors([422, 500]))
def image_rank_sigma_scores_by_model_id(request: Request, model_id: int):
    response_handler = ApiResponseHandlerV1(request)
    try:
        query = {"model_id": model_id}
        items_cursor = request.app.image_sigma_scores_collection.find(query).sort("sigma_score", -1)
        items = list(items_cursor)

        # Prepare the documents, ensuring removal of '_id'
        for item in items:
            item.pop('_id', None)
        
        return response_handler.create_success_response_v1(response_data={'scores': items}, http_status_code=200)
    except Exception as e:
        return response_handler.create_error_response_v1(error_code=ErrorCode.OTHER_ERROR, error_string=str(e), http_status_code=500)


@router.delete("/image-scores/sigma-scores/delete-all-image-rank-sigma-scores-by-model-id",
               tags=["sigma score"],
               response_model=StandardSuccessResponseV1[WasPresentResponse],
               responses=ApiResponseHandlerV1.listErrors([404, 422]))
@router.delete("/sigma-score/image-rank-sigma-scores-by-model-id",
               tags=["sigma score"],
               response_model=StandardSuccessResponseV1[WasPresentResponse],
               responses=ApiResponseHandlerV1.listErrors([404, 422]))
def delete_image_rank_sigma_scores_by_model_id(request: Request, model_id: int):
    response_handler = ApiResponseHandlerV1(request)
    query = {"model_id": model_id}
    
    # Perform the deletion operation
    res = request.app.image_sigma_scores_collection.delete_many(query)
    
    # Check if any documents were deleted and prepare the response accordingly
    was_present = res.deleted_count > 0
    
    if was_present:
        return response_handler.create_success_delete_response_v1(
            True,
            http_status_code=200
        )
    else:
        return response_handler.create_success_delete_response_v1(
            False,
            http_status_code=200
        )