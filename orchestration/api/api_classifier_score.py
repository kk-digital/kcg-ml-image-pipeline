from fastapi import Request, APIRouter
from .api_utils import ErrorCode, WasPresentResponse, ApiResponseHandlerV1, StandardSuccessResponseV1
from orchestration.api.mongo_schemas import ClassifierScore
from fastapi.encoders import jsonable_encoder


router = APIRouter()


@router.get("/classifier-score/get-image-classifier-scores-by-tag",
            description="Get the images scores by tag",
            status_code=200,
            tags=["classifier-score"],
            response_model=StandardSuccessResponseV1[ClassifierScore],
            responses=ApiResponseHandlerV1.listErrors([400, 422]))
def get_image_classifier_scores_by_tag(request: Request, tag_id: str, model_id: int, sort: int):
    api_response_handler = ApiResponseHandlerV1(request)

    query = {"tag_id": tag_id, "model_id": model_id}
    items = request.app.image_classifier_scores_collection.find(query).sort("score", sort)

    if not items:
        # If no items found, use ApiResponseHandler to return a standardized error response
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="No scores found for specified model_id and tag_id.",
            http_status_code=400
        )
    
    score_data = []
    for item in items:
        # remove the auto generated '_id' field
        item.pop('_id', None)
        score_data.append(item)
    
    # Return a standardized success response with the score data
    return api_response_handler.create_success_response_v1(
        response_data=score_data,
        http_status_code=200
    )    


@router.get("/classifier-score/get-image-classifier-score-by-hash", 
            description="Get image classifier score by model_id, tag_id and image_hash",
            status_code=200,
            tags=["score"],  
            response_model=StandardSuccessResponseV1[ClassifierScore],  # Specify the expected response model, adjust as needed
            responses=ApiResponseHandlerV1.listErrors([400,422]))
def get_image_classifier_score_by_hash(request: Request, image_hash: str, model_id: int, tag_id: str):
    api_response_handler = ApiResponseHandlerV1(request)

    # check if exists
    query = {"image_hash": image_hash, "model_id": model_id, "tag_id": tag_id}

    item = request.app.image_classifier_scores_collection.find_one(query)

    if item is None:
        # Return a standardized error response if not found
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="Score for specified model_id, tag_id and image_hash does not exist.",
            http_status_code=404
        )

    # Remove the auto generated '_id' field before returning
    item.pop('_id', None)

    # Return a standardized success response
    return api_response_handler.create_success_response_v1(
        response_data=item,
        http_status_code=200
    )


@router.put("/classifier-score/update-image-classifier-score-by-hash", 
            description="put image classfier score by hash",
            status_code=200,
            tags=["put_score_by_hash"],
            response_model=StandardSuccessResponseV1[ClassifierScore],  # Specify the expected response model, adjust as needed
            responses=ApiResponseHandlerV1.listErrors([400,422]))
def update_image_classifier_score_by_hash(request: Request, classifier_score: ClassifierScore):
    print("Updating classifier score", classifier_score)
    api_response_handler = ApiResponseHandlerV1(request, body_data=classifier_score.to_dict())

    # check if exists
    query = {"image_hash": classifier_score.image_hash, "model_id": classifier_score.model_id, "tag_id": classifier_score.tag_id}

    item = request.app.image_classifier_scores_collection.find_one(query)

    if item is None:
        # Return a standardized error response if not found
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="Score for specified model_id, tag_id and image_hash does not exist.",
            http_status_code=404
        )

    # Remove the auto generated '_id' field before returning
    item = request.app.image_classifier_scores_collection.update_one(
            query,
            {
                "$set": {
                    "score": classifier_score.score
                },
            }
        )
    
    if not item:
        updated = True
    else:
        updated = False
    # Return a standardized success response
    return api_response_handler.create_success_response_v1(
        response_data={"update": updated},
        http_status_code=200
    )


@router.post("/classifier-score/set-image-classifier-score", 
             status_code=200,
             description="Set classifier image score",
             tags=["score"],  
             )  # Added 409 for conflict
def set_image_classifier_score(request: Request, classifier_score: ClassifierScore):

    api_response_handler = ApiResponseHandlerV1(request, body_data=classifier_score.to_dict())
    # check if exists
    query = {"image_hash": classifier_score.image_hash,
             "tag_id": classifier_score.tag_id,
             "model_id": classifier_score.model_id}
    
    count = request.app.image_classifier_scores_collection.count_documents(query)
    if count > 0:
        # Using ApiResponseHandler for standardized error response
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="Score for specific model_id, tag_id and image_hash already exists.",
            http_status_code=400
        )

    # Insert the new ranking score
    request.app.image_classifier_scores_collection.insert_one(classifier_score.to_dict())

    # Using ApiResponseHandler for standardized success response
    return api_response_handler.create_success_response_v1(
        response_data=classifier_score.to_dict(),
        http_status_code=200  
    )


@router.delete("/classifier-score/delete-image-classifier-score-by-hash", 
               description="Delete image classifier score by specific hash.",
               status_code=200,
               response_model=StandardSuccessResponseV1[WasPresentResponse],
               responses=ApiResponseHandlerV1.listErrors([422]))
def delete_image_classifier_score_by_hash(request: Request, image_hash: str, model_id: int, tag_id: str):

    api_response_handler = ApiResponseHandlerV1(request)
    
    # Adjust the query to include model_id
    query = {"image_hash": image_hash, "model_id": model_id, "tag_id": tag_id}
    res = request.app.image_classifier_scores_collection.delete_one(query)
    
    was_present = res.deleted_count > 0
    
    # Use ApiResponseHandler to return the standardized response
    return api_response_handler.create_success_response_v1(
        response_data={"wasPresent": was_present},
        http_status_code=200
    )
