from fastapi import Request, APIRouter, HTTPException
from orchestration.api.mongo_schemas import RankingScore
from .api_utils import ApiResponseHandler, ErrorCode, StandardSuccessResponse, WasPresentResponse

router = APIRouter()


@router.post("/score/set-image-rank-score", description="Set image rank score")
def set_image_rank_score(request: Request, ranking_score: RankingScore):
    # check if exists
    query = {"image_hash": ranking_score.image_hash,
             "model_id": ranking_score.model_id}
    count = request.app.image_scores_collection.count_documents(query)
    if count > 0:
        raise HTTPException(status_code=409, detail="Score for specific model_id and image_hash already exists.")

    request.app.image_scores_collection.insert_one(ranking_score.to_dict())

    return True


@router.post("/score/image-rank", 
             status_code=200,
             description="Set image rank score",
             tags=["score"],  
             response_model=StandardSuccessResponse[RankingScore],
             responses=ApiResponseHandler.listErrors([400, 422]))  # Added 409 for conflict
def set_image_rank_score(request: Request, ranking_score: RankingScore):
    api_response_handler = ApiResponseHandler(request)
    
    # check if exists
    query = {"image_hash": ranking_score.image_hash,
             "model_id": ranking_score.model_id}
    count = request.app.image_scores_collection.count_documents(query)
    if count > 0:
        # Using ApiResponseHandler for standardized error response
        return api_response_handler.create_error_response(
            ErrorCode.INVALID_PARAMS,
            "Score for specific model_id and image_hash already exists.",
            400
        )

    # Insert the new ranking score
    request.app.image_scores_collection.insert_one(ranking_score.dict())

    # Using ApiResponseHandler for standardized success response
    return api_response_handler.create_success_response(
        ranking_score.dict(),
        200  
    )


@router.get("/score/get-image-rank-score-by-hash", description="Get image rank score by hash")
def get_image_rank_score_by_hash(request: Request, image_hash: str, model_id: int):
    # check if exist
    query = {"image_hash": image_hash,
             "model_id": model_id}

    item = request.app.image_scores_collection.find_one(query)
    if item is None:
        return None

    # remove the auto generated field
    item.pop('_id', None)

    return item


@router.get("/score/image-rank-score-by-hash", 
            description="Get image rank score by hash",
            status_code=200,
            tags=["score"],  
            response_model=StandardSuccessResponse[RankingScore],  # Specify the expected response model, adjust as needed
            responses=ApiResponseHandler.listErrors([400,422]))
def get_image_rank_score_by_hash(request: Request, image_hash: str, model_id: str):
    api_response_handler = ApiResponseHandler(request)

    # check if exists
    query = {"image_hash": image_hash, "model_id": model_id}
    item = request.app.image_scores_collection.find_one(query)

    if item is None:
        # Return a standardized error response if not found
        return api_response_handler.create_error_response(
            ErrorCode.INVALID_PARAMS,
            "Score for specified model_id and image_hash does not exist.",
            404
        )

    # Remove the auto generated '_id' field before returning
    item.pop('_id', None)

    # Return a standardized success response
    return api_response_handler.create_success_response(
        item,
        200
    )


@router.get("/score/get-image-rank-scores-by-model-id",
            description="Get image rank scores by model id. Returns as descending order of scores")
def get_image_rank_scores_by_model_id(request: Request, model_id: int):
    # check if exist
    query = {"model_id": model_id}
    items = request.app.image_scores_collection.find(query).sort("score", -1)
    if items is None:
        return []
    
    score_data = []
    for item in items:
        # remove the auto generated field
        item.pop('_id', None)
        score_data.append(item)

    return score_data


@router.get("/score/image-rank-scores-by-model-id",
            description="Get image rank scores by model id. Returns as descending order of scores",
            status_code=200,
            tags=["score"],  
            response_model=StandardSuccessResponse[RankingScore],  # Adjust the response model as needed
            responses=ApiResponseHandler.listErrors([400, 422]))
def get_image_rank_scores_by_model_id(request: Request, model_id: str):
    api_response_handler = ApiResponseHandler(request)
    
    # check if exist
    query = {"model_id": model_id}
    items = list(request.app.image_scores_collection.find(query).sort("score", -1))
    
    if not items:
        # If no items found, use ApiResponseHandler to return a standardized error response
        return api_response_handler.create_error_response(
            ErrorCode.INVALID_PARAMS,
            "No scores found for specified model_id.",
            400
        )
    
    score_data = []
    for item in items:
        # remove the auto generated '_id' field
        item.pop('_id', None)
        score_data.append(item)
    
    # Return a standardized success response with the score data
    return api_response_handler.create_success_response(
        response_data=score_data,
        http_status_code=200
    )


@router.delete("/score/delete-image-rank-scores-by-model-id", description="Delete all image rank scores by model id.")
def delete_image_rank_scores_by_model_id(request: Request, model_id: int):
    # check if exist
    query = {"model_id": model_id}
    res = request.app.image_scores_collection.delete_many(query)
    print(res.deleted_count, " documents deleted.")

    return None


@router.delete("/score/image-rank-scores-by-model-id", 
               description="Delete all image rank scores by model id.",
               status_code=200,
               tags=["score"],  
               response_model=StandardSuccessResponse[WasPresentResponse],
               responses=ApiResponseHandler.listErrors([422]))
def delete_image_rank_scores_by_model_id(request: Request, model_id: str):
    api_response_handler = ApiResponseHandler(request)
    
    query = {"model_id": model_id}
    res = request.app.image_scores_collection.delete_many(query)
    
    was_present = res.deleted_count > 0
    
    return api_response_handler.create_success_response(
        response_data={"wasPresent": was_present},
        http_status_code=200
    )


@router.delete("/score/image-rank-score-by-hash", 
               description="Delete image rank score by specific hash.",
               status_code=200,
               response_model=StandardSuccessResponse[WasPresentResponse],
               responses=ApiResponseHandler.listErrors([422]))
def delete_image_rank_score_by_hash(request: Request, image_hash: str, model_id: str):
    api_response_handler = ApiResponseHandler(request)
    
    # Adjust the query to include model_id
    query = {"image_hash": image_hash, "model_id": model_id}
    res = request.app.image_scores_collection.delete_one(query)
    
    was_present = res.deleted_count > 0
    
    # Use ApiResponseHandler to return the standardized response
    return api_response_handler.create_success_response(
        response_data={"wasPresent": was_present},
        http_status_code=200
    )
