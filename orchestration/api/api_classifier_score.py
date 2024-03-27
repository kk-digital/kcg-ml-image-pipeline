from fastapi import Request, APIRouter, Query
from .api_utils import ErrorCode, WasPresentResponse, ApiResponseHandlerV1, StandardSuccessResponseV1
from orchestration.api.mongo_schemas import ClassifierScore, ListClassifierScore
from fastapi.encoders import jsonable_encoder
import uuid
from typing import Optional

router = APIRouter()


@router.get("/classifier-score/get-scores-by-classifier-id-and-tag-id",
            description="Get the images scores by tag",
            status_code=200,
            tags=["classifier-score"],
            response_model=StandardSuccessResponseV1[ClassifierScore],
            responses=ApiResponseHandlerV1.listErrors([400, 422]))
def get_scores_by_classifier_id_and_tag_id(request: Request, 
                                                  tag_id: int, 
                                                 classifier_id: int, 
                                                 sort: int = -1):
    api_response_handler = ApiResponseHandlerV1(request)

    query = {"classifier_id": classifier_id, "tag_id": tag_id}
    items = request.app.image_classifier_scores_collection.find(query).sort("score", sort)

    if not items:
        # If no items found, use ApiResponseHandler to return a standardized error response
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="No scores found for specified tag_id.",
            http_status_code=400
        )
    
    score_data = []
    for item in items:
        # remove the auto generated '_id' field
        item.pop('_id', None)
        score_data.append(item)
        print(item)
    print(len(score_data))
    # Return a standardized success response with the score data
    return api_response_handler.create_success_response_v1(
        response_data=score_data,
        http_status_code=200
    )    


@router.get("/classifier-score/get-image-classifier-score-by-hash", 
            description="Get image classifier score by classifier_id, tag_id and image_hash",
            status_code=200,
            tags=["score"],  
            response_model=StandardSuccessResponseV1[ClassifierScore],  # Specify the expected response model, adjust as needed
            responses=ApiResponseHandlerV1.listErrors([400,422]))
def get_image_classifier_score_by_hash(request: Request, image_hash: str, tag_id: int, classifier_id: int):
    api_response_handler = ApiResponseHandlerV1(request)

    # check if exists
    query = {"image_hash": image_hash, "tag_id": tag_id, "classifier_id": classifier_id}

    item = request.app.image_classifier_scores_collection.find_one(query)

    if item is None:
        # Return a standardized error response if not found
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="Score for specified classifier_id, tag_id and image_hash does not exist.",
            http_status_code=404
        )

    # Remove the auto generated '_id' field before returning
    item.pop('_id', None)

    # Return a standardized success response
    return api_response_handler.create_success_response_v1(
        response_data=item,
        http_status_code=200
    )



@router.get("/classifier-score/get-image-classifier-score-by-uuid", 
            description="Get image classifier score by uuid",
            status_code=200,
            tags=["score"],  
            response_model=StandardSuccessResponseV1[ClassifierScore],  # Specify the expected response model, adjust as needed
            responses=ApiResponseHandlerV1.listErrors([400,422]))
def get_image_classifier_score_by_uuid(request: Request, classifier_score_uuid: str):
    api_response_handler = ApiResponseHandlerV1(request)

    # check if exists
    query = {"uuid": classifier_score_uuid}

    item = request.app.image_classifier_scores_collection.find_one(query)

    if item is None:
        # Return a standardized error response if not found
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="Score for specified uuid does not exist.",
            http_status_code=404
        )

    # Remove the auto generated '_id' field before returning
    item.pop('_id', None)

    # Return a standardized success response
    return api_response_handler.create_success_response_v1(
        response_data=item,
        http_status_code=200
    )


@router.put("/classifier-score/update-image-classifier-score-by-uuid",
            description="update image-classfier-score by uuid",
            status_code=200,
            tags=["put_score_by_hash"],
            response_model=StandardSuccessResponseV1[ClassifierScore],  # Specify the expected response model, adjust as needed
            responses=ApiResponseHandlerV1.listErrors([400,422]))
async def update_image_classifier_score_by_uuid(request: Request, classifier_score: ClassifierScore):
    print("Updating classifier score", classifier_score)
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    query = {"uuid": classifier_score.uuid}

    item = request.app.image_classifier_scores_collection.find_one(query)

    # check if exists
    if item is None:
        # Return a standardized error response if not found
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="Score for specified image_classifier_score uuid does not exist.",
            http_status_code=404
        )

    # Remove the auto generated '_id' field before returning
    item = request.app.image_classifier_scores_collection.update_one(
            query,
            {
                "$set": {
                    "classifier_id": classifier_score.classifier_id,
                    "tag_id": classifier_score.tag_id,
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
             )
async def set_image_classifier_score(request: Request, classifier_score: ClassifierScore):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    # Check if the uuid exists in the completed_jobs_collection
    uuid_exists = request.app.completed_jobs_collection.count_documents({"uuid": classifier_score.uuid}) > 0
    if not uuid_exists:
        # UUID does not exist in completed_jobs_collection
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="The provided UUID does not exist in the completed jobs.",
            http_status_code=404  # Using 404 to indicate the UUID was not found
        )
    
    # check if exists
    query = {"classifier_id": classifier_score.classifier_id,
             "uuid": classifier_score.uuid,
             "tag_id": classifier_score.tag_id}
    
    count = request.app.image_classifier_scores_collection.count_documents(query)
    if count > 0:
        item = request.app.image_classifier_scores_collection.update_one(
        query,
        {
            "$set": {
                "score": classifier_score.score,
                "image_hash": classifier_score.image_hash
            },
        }
        )
    else:
        # Insert the new ranking score
        request.app.image_classifier_scores_collection.insert_one(classifier_score.to_dict())

    # Using ApiResponseHandler for standardized success response
    return api_response_handler.create_success_response_v1(
        response_data=classifier_score.to_dict(),
        http_status_code=200  
    )


@router.delete("/classifier-score/delete-image-classifier-score-by-uuid", 
               description="Delete image classifier score by specific uuid.",
               status_code=200,
               response_model=StandardSuccessResponseV1[WasPresentResponse],
               responses=ApiResponseHandlerV1.listErrors([422]))
def delete_image_classifier_score_by_uuid(
    request: Request,
    classifier_score_uuid: str):

    api_response_handler = ApiResponseHandlerV1(request)
    
    query = {"uuid": classifier_score_uuid}
    res = request.app.image_classifier_scores_collection.delete_one(query)
    
    was_present = res.deleted_count > 0
    
    # Use ApiResponseHandler to return the standardized response
    return api_response_handler.create_success_response_v1(
        response_data={"wasPresent": was_present},
        http_status_code=200
    )


@router.get("/classifier-score/list-by-scores", 
            description="List images by classifier scores",
            tags=["score"],  
            response_model=StandardSuccessResponseV1[ListClassifierScore],  # Adjust the response model as needed
            responses=ApiResponseHandlerV1.listErrors([400, 422]))
async def list_images_by_classifier_scores(
    request: Request,
    classifier_id: Optional[int] = Query(None, description="Filter by classifier ID"),
    min_score: Optional[float] = Query(None, description="Minimum score"),
    max_score: Optional[float] = Query(None, description="Maximum score"),
    limit: int = Query(10, alias="limit")
):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    # Build the query based on provided filters
    query = {}
    if classifier_id is not None:
        query["classifier_id"] = classifier_id
    if min_score is not None and max_score is not None:
        query["score"] = {"$gte": min_score, "$lte": max_score}
    elif min_score is not None:
        query["score"] = {"$gte": min_score}
    elif max_score is not None:
        query["score"] = {"$lte": max_score}

    # Fetch data from MongoDB with a limit
    cursor = request.app.image_classifier_scores_collection.find(query).limit(limit)
    scores_data = list(cursor)

    # Remove _id in response data
    for score in scores_data:
        score.pop('_id', None)

    # Prepare the data for the response
    images_data = ListClassifierScore(images=[ClassifierScore(**doc).to_dict() for doc in scores_data]).dict()

    # Return the fetched data with a success response
    return response_handler.create_success_response_v1(
        response_data=images_data, 
        http_status_code=200
    )




# Updated apis



@router.get("/classifier-scores/get-scores-by-classifier-id-and-tag-id",
            description="Get the images scores by tag",
            status_code=200,
            tags=["classifier-scores"],
            response_model=StandardSuccessResponseV1[ClassifierScore],
            responses=ApiResponseHandlerV1.listErrors([400, 422]))
def get_scores_by_classifier_id_and_tag_id(request: Request, 
                                                  tag_id: int, 
                                                 classifier_id: int, 
                                                 sort: int = -1):
    api_response_handler = ApiResponseHandlerV1(request)

    query = {"classifier_id": classifier_id, "tag_id": tag_id}
    items = request.app.image_classifier_scores_collection.find(query).sort("score", sort)

    if not items:
        # If no items found, use ApiResponseHandler to return a standardized error response
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="No scores found for specified tag_id.",
            http_status_code=400
        )
    
    score_data = []
    for item in items:
        # remove the auto generated '_id' field
        item.pop('_id', None)
        score_data.append(item)
        print(item)
    print(len(score_data))
    # Return a standardized success response with the score data
    return api_response_handler.create_success_response_v1(
        response_data=score_data,
        http_status_code=200
    )    


@router.get("/classifier-scores/get-image-classifier-score-by-hash", 
            description="Get image classifier score by classifier_id, tag_id and image_hash",
            status_code=200,
            tags=["classifier-scores"],  
            response_model=StandardSuccessResponseV1[ClassifierScore],  # Specify the expected response model, adjust as needed
            responses=ApiResponseHandlerV1.listErrors([400,422]))
def get_image_classifier_score_by_hash(request: Request, image_hash: str, tag_id: int, classifier_id: int):
    api_response_handler = ApiResponseHandlerV1(request)

    # check if exists
    query = {"image_hash": image_hash, "tag_id": tag_id, "classifier_id": classifier_id}

    item = request.app.image_classifier_scores_collection.find_one(query)

    if item is None:
        # Return a standardized error response if not found
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="Score for specified classifier_id, tag_id and image_hash does not exist.",
            http_status_code=404
        )

    # Remove the auto generated '_id' field before returning
    item.pop('_id', None)

    # Return a standardized success response
    return api_response_handler.create_success_response_v1(
        response_data=item,
        http_status_code=200
    )



@router.get("/classifier-scores/get-image-classifier-score-by-uuid", 
            description="Get image classifier score by uuid",
            status_code=200,
            tags=["classifier-scores"],  
            response_model=StandardSuccessResponseV1[ClassifierScore],  # Specify the expected response model, adjust as needed
            responses=ApiResponseHandlerV1.listErrors([400,422]))
def get_image_classifier_score_by_uuid(request: Request, classifier_score_uuid: str):
    api_response_handler = ApiResponseHandlerV1(request)

    # check if exists
    query = {"uuid": classifier_score_uuid}

    item = request.app.image_classifier_scores_collection.find_one(query)

    if item is None:
        # Return a standardized error response if not found
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="Score for specified uuid does not exist.",
            http_status_code=404
        )

    # Remove the auto generated '_id' field before returning
    item.pop('_id', None)

    # Return a standardized success response
    return api_response_handler.create_success_response_v1(
        response_data=item,
        http_status_code=200
    )


@router.put("/classifier-scores/update-image-classifier-score-by-uuid",
            description="update image-classfier-score by uuid",
            status_code=200,
            tags=["classifier-scores"],
            response_model=StandardSuccessResponseV1[ClassifierScore],  # Specify the expected response model, adjust as needed
            responses=ApiResponseHandlerV1.listErrors([400,422]))
async def update_image_classifier_score_by_uuid(request: Request, classifier_score: ClassifierScore):
    print("Updating classifier score", classifier_score)
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    query = {"uuid": classifier_score.uuid}

    item = request.app.image_classifier_scores_collection.find_one(query)

    # check if exists
    if item is None:
        # Return a standardized error response if not found
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="Score for specified image_classifier_score uuid does not exist.",
            http_status_code=404
        )

    # Remove the auto generated '_id' field before returning
    item = request.app.image_classifier_scores_collection.update_one(
            query,
            {
                "$set": {
                    "classifier_id": classifier_score.classifier_id,
                    "tag_id": classifier_score.tag_id,
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



@router.post("/classifier-scores/set-image-classifier-score", 
             status_code=200,
             description="Set classifier image score",
             tags=["classifier-scores"],  
             )
async def set_image_classifier_score(request: Request, classifier_score: ClassifierScore):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    # Check if the uuid exists in the completed_jobs_collection
    uuid_exists = request.app.completed_jobs_collection.count_documents({"uuid": classifier_score.uuid}) > 0
    if not uuid_exists:
        # UUID does not exist in completed_jobs_collection
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="The provided UUID does not exist in the completed jobs.",
            http_status_code=404  # Using 404 to indicate the UUID was not found
        )
    
    # check if exists
    query = {"classifier_id": classifier_score.classifier_id,
             "uuid": classifier_score.uuid,
             "tag_id": classifier_score.tag_id}
    
    count = request.app.image_classifier_scores_collection.count_documents(query)
    if count > 0:
        item = request.app.image_classifier_scores_collection.update_one(
        query,
        {
            "$set": {
                "score": classifier_score.score,
                "image_hash": classifier_score.image_hash
            },
        }
        )
    else:
        # Insert the new ranking score
        request.app.image_classifier_scores_collection.insert_one(classifier_score.to_dict())

    # Using ApiResponseHandler for standardized success response
    return api_response_handler.create_success_response_v1(
        response_data=classifier_score.to_dict(),
        http_status_code=200  
    )


@router.delete("/classifier-scores/delete-image-classifier-score-by-uuid", 
               description="Delete image classifier score by specific uuid.",
               status_code=200,
               tags = ["classifier-scores"],
               response_model=StandardSuccessResponseV1[WasPresentResponse],
               responses=ApiResponseHandlerV1.listErrors([422]))
def delete_image_classifier_score_by_uuid(
    request: Request,
    classifier_score_uuid: str):

    api_response_handler = ApiResponseHandlerV1(request)
    
    query = {"uuid": classifier_score_uuid}
    res = request.app.image_classifier_scores_collection.delete_one(query)
    
    was_present = res.deleted_count > 0
    
    # Use ApiResponseHandler to return the standardized response
    return api_response_handler.create_success_response_v1(
        response_data={"wasPresent": was_present},
        http_status_code=200
    )


@router.get("/classifier-scores/list-by-scores", 
            description="List images by classifier scores",
            tags=["classifier-scores"],  
            response_model=StandardSuccessResponseV1[ListClassifierScore],  # Adjust the response model as needed
            responses=ApiResponseHandlerV1.listErrors([400, 422]))
async def list_images_by_classifier_scores(
    request: Request,
    classifier_id: Optional[int] = Query(None, description="Filter by classifier ID"),
    min_score: Optional[float] = Query(None, description="Minimum score"),
    max_score: Optional[float] = Query(None, description="Maximum score"),
    limit: int = Query(10, alias="limit")
):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    # Build the query based on provided filters
    query = {}
    if classifier_id is not None:
        query["classifier_id"] = classifier_id
    if min_score is not None and max_score is not None:
        query["score"] = {"$gte": min_score, "$lte": max_score}
    elif min_score is not None:
        query["score"] = {"$gte": min_score}
    elif max_score is not None:
        query["score"] = {"$lte": max_score}

    # Fetch data from MongoDB with a limit
    cursor = request.app.image_classifier_scores_collection.find(query).limit(limit)
    scores_data = list(cursor)

    # Remove _id in response data
    for score in scores_data:
        score.pop('_id', None)

    # Prepare the data for the response
    images_data = ListClassifierScore(images=[ClassifierScore(**doc).to_dict() for doc in scores_data]).dict()

    # Return the fetched data with a success response
    return response_handler.create_success_response_v1(
        response_data=images_data, 
        http_status_code=200
    )