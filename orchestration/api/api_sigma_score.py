from fastapi import Request, APIRouter, HTTPException
from orchestration.api.mongo_schemas import RankingSigmaScore
from .api_utils import PrettyJSONResponse, ApiResponseHandler, ErrorCode, StandardSuccessResponse, WasPresentResponse, TagsListResponse, VectorIndexUpdateRequest, TagsCategoryListResponse, TagResponse, TagCountResponse
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

@router.post("/sigma-score/image-rank-sigma-score",
             status_code=201,  # Use 201 for resource creation
             description="Set image rank sigma_score",
             tags=["sigma score"],
             response_model=StandardSuccessResponse[RankingSigmaScore],
             responses=ApiResponseHandler.listErrors([400, 422]))
async def set_image_rank_sigma_score(request: Request, ranking_sigma_score: RankingSigmaScore):
    response_handler = ApiResponseHandler(request)
    query = {"image_hash": ranking_sigma_score.image_hash, "model_id": ranking_sigma_score.model_id}
    count = request.app.image_sigma_scores_collection.count_documents(query)

    if count > 0:
        # If a sigma score already exists for the given image_hash and model_id, return an error response
        return response_handler.create_error_response(ErrorCode.INVALID_PARAMS, "Score for specific model_id and image_hash already exists.", 400)

    # If the sigma score does not exist, insert the new score
    request.app.image_sigma_scores_collection.insert_one(ranking_sigma_score.dict())

    # Return a success response with the inserted sigma score data
    return response_handler.create_success_response(ranking_sigma_score.dict(), 201)


@router.get("/sigma-score/image-rank-sigma-score-by-hash", 
            status_code=200,
            description="Get image rank sigma_score by hash",
            tags=["sigma score"],
            response_model=StandardSuccessResponse[RankingSigmaScore],  
            responses=ApiResponseHandler.listErrors([422, 500]))
def get_image_rank_sigma_score_by_hash(request: Request, image_hash: str, model_id: int):
    response_handler = ApiResponseHandler(request)
    query = {"image_hash": image_hash, "model_id": model_id}

    item = request.app.image_sigma_scores_collection.find_one(query)
    if not item:
        # If no sigma score found for the given image_hash and model_id, return an error response
        return response_handler.create_error_response(ErrorCode.ELEMENT_NOT_FOUND, "Image rank sigma_score data not found", 404)

    # Remove the auto-generated '_id' field from the MongoDB document before returning
    item.pop('_id', None)

    # Return a success response with the found sigma score data
    return response_handler.create_success_response(item, 200)


@router.get("/sigma-score/image-rank-sigma-scores-by-model-id",
            response_model=StandardSuccessResponse[List[RankingSigmaScore]],
            tags=["sigma score"],
            description="Get image rank sigma_scores by model id. Returns as descending order of sigma_scores",
            responses=ApiResponseHandler.listErrors([422, 500]))
def image_rank_sigma_scores_by_model_id(request: Request, model_id: int):
    response_handler = ApiResponseHandler(request)
    try:
        query = {"model_id": model_id}
        items_cursor = request.app.image_sigma_scores_collection.find(query).sort("sigma_score", -1)
        items = list(items_cursor)

        if not items:
            return response_handler.create_success_response({"response": []}, 200)

        # Prepare the documents, ensuring removal of '_id'
        for item in items:
            item.pop('_id', None)
        
        return response_handler.create_success_response(items, 200)
    except Exception as e:
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, str(e), 500)



@router.delete("/sigma-score/image-rank-sigma-scores-by-model-id/{model_id}",
               tags=["sigma score"],
               response_model=StandardSuccessResponse[WasPresentResponse],
               responses=ApiResponseHandler.listErrors([404, 422]))
async def delete_image_rank_sigma_scores_by_model_id(request: Request, model_id: int):
    response_handler = ApiResponseHandler(request)
    query = {"model_id": model_id}
    
    # Perform the deletion operation
    res = request.app.image_sigma_scores_collection.delete_many(query)
    
    # Check if any documents were deleted and prepare the response accordingly
    was_present = res.deleted_count > 0
    
    if was_present:
        return response_handler.create_success_response({"wasPresent": was_present}, 200)
    else:
        # If no documents were deleted, it might indicate the model_id didn't match any documents
        return response_handler.create_error_response(ErrorCode.ELEMENT_NOT_FOUND, "No sigma_scores found for the given model_id to delete", 404)