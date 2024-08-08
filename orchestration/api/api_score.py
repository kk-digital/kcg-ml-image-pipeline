from datetime import datetime
from fastapi import Request, APIRouter, HTTPException, Query
from typing import Optional

from pymongo import UpdateOne
from orchestration.api.mongo_schemas import RankingScore, ResponseRankingScore, ListRankingScore
from .api_utils import ApiResponseHandler, ErrorCode, StandardSuccessResponse, WasPresentResponse, ApiResponseHandlerV1, StandardSuccessResponseV1

router = APIRouter()


@router.post("/score/set-image-rank-score", 
             tags=['deprecated3'], 
             description="changed with /image-scores/scores/set-rank-score")
async def set_image_rank_score(request: Request, ranking_score: RankingScore):
    # Check if image exists in the completed_jobs_collection
    image_data = request.app.completed_jobs_collection.find_one(
        {"task_output_file_dict.output_file_hash": ranking_score.image_hash},
        {"task_output_file_dict.output_file_hash": 1}
    )
    if not image_data:
        raise HTTPException(status_code=404, detail="Image with the given hash not found in completed jobs collection.")

    # Check if the score already exists in image_rank_scores_collection
    query = {"image_hash": ranking_score.image_hash, "rank_model_id": ranking_score.rank_model_id}
    count = request.app.image_rank_scores_collection.count_documents(query)
    if count > 0:
        raise HTTPException(status_code=409, detail="Score for specific rank_model_id and image_hash already exists.")

    # Add the image_source property set to "generated_image"
    ranking_score_data = ranking_score.dict()
    ranking_score_data['image_source'] = "generated_image"

    # Insert the new ranking score
    request.app.image_rank_scores_collection.insert_one(ranking_score_data)

    return True


@router.post("/image-scores/scores/set-rank-score", 
             status_code=201,
             description="Sets the rank score of an image. The score can only be set one time per image/model combination",
             tags=["image scores"],  
             response_model=StandardSuccessResponseV1[ResponseRankingScore],
             responses=ApiResponseHandlerV1.listErrors([400, 422])) 
@router.post("/score/set-rank-score", 
             status_code=201,
             description="deprecated: use /image-scores/scores/set-rank-score",
             tags=["deprecated2"],  
             response_model=StandardSuccessResponseV1[RankingScore],
             responses=ApiResponseHandlerV1.listErrors([400, 422])) 
async def set_image_rank_score(
    request: Request, 
    ranking_score: RankingScore, 
    image_source: str = Query(..., regex="^(generated_image|extract_image|external_image)$")
):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    # Check if rank_id exists in rank_model_models_collection
    model_exists = request.app.rank_model_models_collection.find_one(
        {"rank_model_id": ranking_score.rank_id},
        {"_id": 1}
    )
    if not model_exists:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="The provided rank_model_id does not exist in rank_model_models_collection.",
            http_status_code=400
        )

    # Determine the appropriate collection based on image_source
    if image_source == "generated_image":
        collection = request.app.completed_jobs_collection
    elif image_source == "extract_image":
        collection = request.app.extracts_collection
    elif image_source == "external_image":
        collection = request.app.external_images_collection
    else:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="Invalid image source provided.",
            http_status_code=422
        )

    # Fetch additional data from the determined collection
    if image_source == "generated_image":
        image_data = collection.find_one(
            {"uuid": ranking_score.uuid},
            {"task_output_file_dict.output_file_hash": 1}
        )
        if not image_data or 'task_output_file_dict' not in image_data or 'output_file_hash' not in image_data['task_output_file_dict']:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string=f"Image with UUID {ranking_score.uuid} not found or missing 'output_file_hash' in task_output_file_dict.",
                http_status_code=422
            )
        image_hash = image_data['task_output_file_dict']['output_file_hash']
    else:
        image_data = collection.find_one({"uuid": ranking_score.uuid}, {"image_hash": 1})
        if not image_data:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string=f"Image with UUID {ranking_score.uuid} not found in {image_source} collection.",
                http_status_code=422
            )
        image_hash = image_data['image_hash']

    # Check if the score already exists in image_rank_scores_collection
    query = {
        "uuid": ranking_score.uuid,
        "image_hash": image_hash,
        "rank_model_id": ranking_score.rank_model_id
    }
    count = request.app.image_rank_scores_collection.count_documents(query)
    if count > 0:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="Score for specific uuid, rank_model_id and image_hash already exists.",
            http_status_code=400
        )

    # Insert the new ranking score
    ranking_score_data = ranking_score.dict()
    ranking_score_data['image_source'] = image_source
    ranking_score_data['image_hash'] = image_hash
    ranking_score_data["creation_time"] = datetime.utcnow().isoformat(),
    request.app.image_rank_scores_collection.insert_one(ranking_score_data)

    ranking_score_data.pop('_id', None)

    return api_response_handler.create_success_response_v1(
        response_data=ranking_score_data,
        http_status_code=201  
    )

@router.post("/image-scores/scores/set-rank-score-batch", 
             status_code=200,
             response_model=StandardSuccessResponseV1[ListRankingScore],
             description="Set rank image scores in a batch",
             tags=["image scores"], 
             responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def set_image_rank_score_batch(
    request: Request, 
    batch_scores: ListRankingScore
):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        bulk_operations = []
        response_data = []

        for ranking_score in batch_scores.scores:
            query = {
                "uuid": ranking_score.uuid,
                "image_hash": ranking_score.image_hash,
                "rank_model_id": ranking_score.rank_model_id    
            }

            new_score_data = {
                "uuid": ranking_score.uuid,
                "rank_model_id": ranking_score.rank_model_id,
                "rank_id": ranking_score.rank_id,
                "score": ranking_score.score,
                "sigma_score": ranking_score.sigma_score,
                "image_hash": ranking_score.image_hash,
                "creation_time": datetime.utcnow().isoformat(),
                "image_source": ranking_score.image_source
            }

            update_operation = UpdateOne(
                query,
                {"$set": new_score_data},
                upsert=True
            )
            bulk_operations.append(update_operation)
            response_data.append(new_score_data)

        if bulk_operations:
            request.app.image_rank_scores_collection.bulk_write(bulk_operations)

        return api_response_handler.create_success_response_v1(
            response_data=response_data,
            http_status_code=200  
        )
    
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )


@router.get("/image-scores/scores/get-image-rank-score", 
            description="Get image rank score by hash",
            status_code=200,
            tags=["image scores"],  
            response_model=StandardSuccessResponseV1[ResponseRankingScore],  
            responses=ApiResponseHandlerV1.listErrors([400,422]))
@router.get("/score/image-rank-score-by-hash", 
            description="deprectaed: use /image-scores/scores/get-image-rank-score ",
            status_code=200,
            tags=["deprecated2"],  
            response_model=StandardSuccessResponseV1[RankingScore],  
            responses=ApiResponseHandlerV1.listErrors([400,422]))
def get_image_rank_score_by_hash(
    request: Request, 
    image_hash: str = Query(..., description="The hash of the image to get score for"), 
    rank_model_id: str = Query(..., description="The rank model ID associated with the image score"),
    image_source: str = Query(..., description="The source of the image", regex="^(generated_image|extract_image|external_image)$")
):
    api_response_handler = ApiResponseHandlerV1(request)

    # Adjust the query to include rank_model_id and image_source
    query = {"image_hash": image_hash, "rank_model_id": rank_model_id, "image_source": image_source}
    item = request.app.image_rank_scores_collection.find_one(query)

    if item is None:
        # Return a standardized error response if not found
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="Score for specified rank_model_id and image_hash does not exist.",
            http_status_code=404
        )

    # Remove the auto generated '_id' field before returning
    item.pop('_id', None)

    # Return a standardized success response
    return api_response_handler.create_success_response_v1(
        response_data=item,
        http_status_code=200
    )


@router.get("/score/get-image-rank-scores-by-model-id",
            tags = ['deprecated3'], description= "changed with /image-scores/scores/list-image-rank-scores-by-model-id")
def get_image_rank_scores_by_rank_model_id(request: Request, rank_model_id: int):
    # check if exist
    query = {"rank_model_id": rank_model_id}
    items = request.app.image_rank_scores_collection.find(query).sort("score", -1)
    if items is None:
        return []
    
    score_data = []
    for item in items:
        # remove the auto generated field
        item.pop('_id', None)
        score_data.append(item)

    return score_data

@router.get("/image-scores/scores/list-image-rank-scores-by-model-id",
            description="Get image rank scores by model id. Returns as descending order of scores",
            status_code=200,
            tags=["image scores"],  
            response_model=StandardSuccessResponseV1[ListRankingScore],  
            responses=ApiResponseHandlerV1.listErrors([422]))
def get_image_rank_scores_by_model_id(
    request: Request, 
    rank_model_id: int, 
    image_source: Optional[str] = Query(None, regex="^(generated_image|extract_image|external_image)$")
):
    api_response_handler = ApiResponseHandlerV1(request)
    
    # Check if exist
    query = {"rank_model_id": rank_model_id}
    if image_source:
        query["image_source"] = image_source

    items = list(request.app.image_rank_scores_collection.find(query).sort("score", -1))
    
    score_data = []
    for item in items:
        # Remove the auto generated '_id' field
        item.pop('_id', None)
        score_data.append(item)
    
    # Return a standardized success response with the score data
    return api_response_handler.create_success_response_v1(
        response_data={'scores': score_data},
        http_status_code=200
    )

@router.get("/image-scores/scores/list-image-rank-scores-by-uuid-and-model",
            description="Get image rank scores by uuid and model. Returns as descending order of scores",
            status_code=200,
            tags=["image scores"],  
            response_model=StandardSuccessResponseV1,
            responses=ApiResponseHandlerV1.listErrors([422]))
def get_image_rank_scores_by_uuid_and_model(
    request: Request, 
    uuid: str, 
    rank_model_id: str, 
    image_source: Optional[str] = Query(None, regex="^(generated_image|extract_image|external_image)$")
):
    api_response_handler = ApiResponseHandlerV1.createInstance(request)
    
    # Check if exist
    query = {"uuid": uuid, "rank_model_id": rank_model_id}
    if image_source:
        query["image_source"] = image_source

    try:
        items = list(request.app.image_rank_scores_collection.find(query).sort("score", -1))
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=500,
            error_string=str(e),
            http_status_code=500
        )
    
    score_data = []
    for item in items:
        # Remove the auto generated '_id' field
        item.pop('_id', None)
        score_data.append(item)
    
    # Return a standardized success response with the score data
    return api_response_handler.create_success_response_v1(
        response_data={'scores': score_data},
        http_status_code=200
    )


@router.delete("/image-scores/scores/delete-image-rank-score", 
               description="Delete image rank score by specific hash.",
               status_code=200,
               tags=["image scores"], 
               response_model=StandardSuccessResponseV1[WasPresentResponse],
               responses=ApiResponseHandlerV1.listErrors([422]))
@router.delete("/score/image-rank-score-by-hash", 
               description="deprecated: use /image-scores/scores/delete-image-rank-score",
               status_code=200,
               tags=["deprecated2"], 
               response_model=StandardSuccessResponseV1[WasPresentResponse],
               responses=ApiResponseHandlerV1.listErrors([422]))
def delete_image_rank_score_by_hash(
    request: Request, 
    image_hash: str = Query(..., description="The hash of the image to delete score for"), 
    rank_model_id: str = Query(..., description="The rank model ID associated with the image score"),
    image_source: str = Query(..., description="The source of the image", regex="^(generated_image|extract_image|external_image)$")
):
    api_response_handler = ApiResponseHandlerV1(request)
    
    # Adjust the query to include rank_model_id and image_source
    query = {"image_hash": image_hash, "rank_model_id": rank_model_id, "image_source": image_source}
    res = request.app.image_rank_scores_collection.delete_one(query)
    
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
