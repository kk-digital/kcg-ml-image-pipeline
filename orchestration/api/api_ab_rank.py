from datetime import datetime
from fastapi import APIRouter, Request, HTTPException, Query
from typing import List, Dict
from orchestration.api.mongo_schema.ab_ranking_schemas import Rankmodel, RankRequest, RankListResponse, ListImageRank, ImageRank, RankCategory, RankCategoryRequest, RankCategoryListResponse, RankCountResponse, RankedSelection
from .mongo_schemas import Classifier, ABRankImagePairResponse, ABRankImagePairResponse_v1
from typing import Union
from .api_utils import PrettyJSONResponse, validate_date_format, ErrorCode, WasPresentResponse, StandardSuccessResponseV1, ApiResponseHandlerV1
import traceback
from bson import ObjectId
import numpy as np
import bisect



router = APIRouter()


@router.post("/ab-rank/add-new-rank-model", 
             status_code=201,
             tags=["ab-rank"],
             description="Adds a new rank",
             response_model=StandardSuccessResponseV1[Rankmodel],
             responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
async def add_new_rank_model_model(request: Request, rank_model_data: RankRequest):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:

        if rank_model_data.classifier_id is not None:
            classifier_info = request.app.classifier_models_collection.find_one({"classifier_id": rank_model_data.classifier_id})
            if not classifier_info:
                return response_handler.create_error_response_v1(
                    error_code=ErrorCode.INVALID_PARAMS,
                    error_string="The provided classifier ID does not exist.",
                    http_status_code=400
                )
            

        # Check for existing rank_model_category_id
        if rank_model_data.rank_model_category_id is not None:
            existing_category = request.app.rank_model_categories_collection.find_one(
                {"rank_model_category_id": rank_model_data.rank_model_category_id}
            )
            if not existing_category:
                return response_handler.create_error_response_v1(
                    error_code=ErrorCode.INVALID_PARAMS,
                    error_string="rank category not found",
                    http_status_code=400,
                    
                )


        # Generate new rank_model_id
        last_entry = request.app.rank_model_models_collection.find_one({}, sort=[("rank_model_id", -1)])
        new_rank_model_id = last_entry["rank_model_id"] + 1 if last_entry and "rank_model_id" in last_entry else 0

        # Check if the rank model exists by rank_model_string
        existing_rank = request.app.rank_model_models_collection.find_one({"rank_model_string": rank_model_data.rank_model_string})
        if existing_rank:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string="rank model already exists.",
                http_status_code=400,
                
            )

        # Create the new rank object with only the specified fields
        new_rank = {
            "rank_model_id": new_rank_model_id,
            "rank_model_string": rank_model_data.rank_model_string,
            "classifier_id": rank_model_data.classifier_id,
            "rank_model_category_id": rank_model_data.rank_model_category_id,
            "rank_model_description": rank_model_data.rank_model_description,
            "deprecated": rank_model_data.deprecated,
            "user_who_created": rank_model_data.user_who_created,
            "creation_time": datetime.utcnow().isoformat()
        }

        # Insert new rank model into the collection
        inserted_id = request.app.rank_model_models_collection.insert_one(new_rank).inserted_id
        new_rank = request.app.rank_model_models_collection.find_one({"_id": inserted_id})

        new_rank = {k: str(v) if isinstance(v, ObjectId) else v for k, v in new_rank.items()}

        return response_handler.create_success_response_v1(
            response_data = new_rank,
            http_status_code=201,
            
        )

    except Exception as e:

        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500,
            
)



@router.put("/ab-rank/update-rank-model", 
              tags=["ab-rank"],
              status_code=200,
              description="Update rank models",
              response_model=StandardSuccessResponseV1[Rankmodel], 
              responses=ApiResponseHandlerV1.listErrors([400, 404, 422, 500]))
async def update_rank_model_model(request: Request, rank_model_id: int, update_data: RankRequest):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    query = {"rank_model_id": rank_model_id}
    existing_rank = request.app.rank_model_models_collection.find_one(query)

    if not existing_rank:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.ELEMENT_NOT_FOUND, 
            error_string="rank not found.", 
            http_status_code=404,
        )

    if existing_rank.get("deprecated", False):
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS, 
            error_string="Cannot modify a deprecated rank.", 
            http_status_code=400,
        )

    update_fields = {k: v for k, v in update_data.dict().items() if v is not None}

    if update_data.classifier_id:
        classifier_info = request.app.classifier_models_collection.find_one(
            {"classifier_id": update_data.classifier_id}
        )
        if not classifier_info:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string="The provided classifier ID does not exist.",
                http_status_code=400
            )

    request.app.rank_model_models_collection.update_one(query, {"$set": update_fields})

    updated_rank = request.app.rank_model_models_collection.find_one(query)
    updated_rank = {k: str(v) if isinstance(v, ObjectId) else v for k, v in updated_rank.items()}

    return response_handler.create_success_response_v1(
        response_data=updated_rank, 
        http_status_code=200,
    )



@router.post("/ab-rank/add-rank-model-in-selection-datapoints",
             status_code=200,
             tags=["ab-rank"],
             response_model=StandardSuccessResponseV1[RankedSelection],
             responses=ApiResponseHandlerV1.listErrors([404, 422, 500]),
             description="Updates image pair ranking with a new rank model ID")
async def update_rank_model_in_image_pairs(
        request: Request, 
        rank_model_id: int = Query(..., description="The new rank model ID to assign"),
        file_name: str = Query(..., description="The file name to match image pairs")
    ):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        # Assuming this updates the document correctly and you've verified this elsewhere
        result = request.app.image_pair_ranking_collection.update_many(
            {"file_name": file_name},
            {"$set": {"rank_model_id": rank_model_id}}
        )

        if result.modified_count == 0:
            # Handle the case where no documents are updated
            return response_handler.create_error_response_v1(
                error_code="No matching image pairs found.",
                error_string=str(e),
                http_status_code=404,
            )

        # Fetching an example document to use its structure for the response
        # Note: This fetch is for illustrative purposes; adapt as necessary
        document = request.app.image_pair_ranking_collection.find_one({"file_name": file_name})

        if not document:
            return response_handler.create_error_response_v1(
                error_code="Document not found after update.",
                error_string=str(e),
                http_status_code=404,
            )

        # Reconstruct the document with rank_model_id first
        document_with_rank_first = {"rank_model_id": rank_model_id, **document}

        # Since MongoDB returns '_id' that isn't JSON serializable by default, convert to string or remove it
        document_with_rank_first.pop('_id', None)  # Removing '_id' for simplicity

        # Return the modified document as a response
        return response_handler.create_success_response_v1( 
            response_data=document_with_rank_first,
            http_status_code=200)

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500,
        )


@router.delete("/ab-rank/remove-rank-model/{rank_model_id}", 
               response_model=StandardSuccessResponseV1[WasPresentResponse], 
               description="remove rank with rank_model_id", 
               tags=["ab-rank"], 
               status_code=200,
               responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def remove_rank(request: Request, rank_model_id: int ):

    response_handler = ApiResponseHandlerV1(request)

    # Check if the rank exists
    rank_model_query = {"rank_model_id": rank_model_id}
    rank = request.app.rank_model_models_collection.find_one(rank_model_query)
    
    if rank is None:
        # Return standard response with wasPresent: false
        return response_handler.create_success_delete_response_v1(
                                                           False,
                                                           http_status_code=200
                                                           )

    # Check if the rank is used in any images
    image_query = {"rank_model_id": rank_model_id}
    image_with_rank = request.app.image_ranks_collection.find_one(image_query)

    if image_with_rank is not None:
        # Since it's used in images, do not delete but notify the client
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="Cannot remove rank, it is already used in images.",
            http_status_code=400
        )

    # Remove the rank
    request.app.rank_model_models_collection.delete_one(rank_model_query)

    # Return standard response with wasPresent: true
    return response_handler.create_success_delete_response_v1(
                                                       True,
                                                       http_status_code=200
                                                       )

@router.delete("/image-ranks/remove-all",
            description="Remove all documents from the image ranks collection",
            tags=["image-ranks"],
            status_code=200)
async def remove_all_image_ranks(request: Request):
    try:
        delete_result = await request.app.image_ranks_collection.delete_many({})
        return {"status": "success", "deleted_count": delete_result.deleted_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ab-rank/list-rank-models",
            response_model=StandardSuccessResponseV1[RankListResponse],
            description="list rank models",
            tags = ["ab-rank"],
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([500]))
def list_rank_model_models(request: Request):
    response_handler = ApiResponseHandlerV1(request)
    try:
        ranks_cursor = request.app.rank_model_models_collection.find({})

        # Prepare the response list
        response_ranks = []

        for rank in ranks_cursor:
            # Convert MongoDB ObjectID to string if necessary
            rank['_id'] = str(rank['_id'])

            # Find the rank category and determine if it's deprecated
            category = request.app.rank_model_categories_collection.find_one({"rank_model_category_id": rank["rank_model_category_id"]})
            deprecated_rank_model_category = category['deprecated'] if category else False
            
            # Append the 'deprecated_rank_model_category' field to the rank data
            response_rank = {**rank, "deprecated_rank_model_category": deprecated_rank_model_category}
            response_ranks.append(response_rank)

        # Return the modified list of ranks including 'deprecated_rank_model_category'
        return response_handler.create_success_response_v1(
            response_data={"ranks": response_ranks},
            http_status_code=200,
        )

    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Exception Traceback:\n{traceback_str}")
        return response_handler.create_error_response_v1(error_code=ErrorCode.OTHER_ERROR,
                                                         error_string="Internal server error",
                                                         http_status_code=500,
                                                         )

  
    
@router.post("/ab-rank-categories/add-rank-category",
             status_code=201, 
             tags=["ab-rank-categories"], 
             description="Add rank Category",
             response_model=StandardSuccessResponseV1[RankCategory],
             responses=ApiResponseHandlerV1.listErrors([422, 500]))
async def add_new_rank_model_category(request: Request, rank_model_category_data: RankCategoryRequest):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        # Assign new rank_model_category_id
        last_entry = request.app.rank_model_categories_collection.find_one({}, sort=[("rank_model_category_id", -1)])
        new_rank_model_category_id = last_entry["rank_model_category_id"] + 1 if last_entry else 0

        # Prepare rank category document
        rank_model_category_document = rank_model_category_data.dict()
        rank_model_category_document["rank_model_category_id"] = new_rank_model_category_id
        rank_model_category_document["creation_time"] = datetime.utcnow().isoformat()

        # Insert new rank category
        inserted_id = request.app.rank_model_categories_collection.insert_one(rank_model_category_document).inserted_id

        # Retrieve and serialize the new rank category object
        new_rank_model_category = request.app.rank_model_categories_collection.find_one({"_id": inserted_id})
        serialized_rank_model_category = {k: str(v) if isinstance(v, ObjectId) else v for k, v in new_rank_model_category.items()}

        # Adjust order of the keys
        ordered_response = {
            "_id": serialized_rank_model_category.pop("_id"),
            "rank_model_category_id": serialized_rank_model_category.pop("rank_model_category_id"),
            **serialized_rank_model_category
        }
        # Return success response
        return response_handler.create_success_response_v1(
            response_data=ordered_response,
            http_status_code=201,
        )
    except Exception as e:
        # Handle exceptions and return an error response
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )

@router.get("/ab-rank-categories/list-rank-categories", 
            description="list rank categories",
            tags=["ab-rank-categories"],
            status_code=200,
            response_model=StandardSuccessResponseV1[RankCategoryListResponse],
            responses=ApiResponseHandlerV1.listErrors([500]))
def list_rank_model_models(request: Request):
    response_handler = ApiResponseHandlerV1(request)
    try:
        # Query all the rank models
        ranks_cursor = list(request.app.rank_model_categories_collection.find({}))

        # Convert each rank document to rankmodel and then to a dictionary
        for doc in ranks_cursor:
            doc.pop('_id', None)  

        return response_handler.create_success_response_v1(
            response_data={"rank_model_categories": ranks_cursor}, 
            http_status_code=200,
            )

    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Exception Traceback:\n{traceback_str}")
        return response_handler.create_error_response_v1(error_code=ErrorCode.OTHER_ERROR, 
                                                         error_string="Internal server error", 
                                                         http_status_code=500,
                            
                                                         )


@router.put("/ab-rank-categories/update-rank-category", 
              tags=["ab-rank-categories"],
              status_code=200,
              description="Update rank category",
              response_model=StandardSuccessResponseV1[RankCategory],
              responses=ApiResponseHandlerV1.listErrors([400, 404, 422, 500]))
async def update_rank_model_category(
    request: Request, 
    rank_model_category_id: int,
    update_data: RankCategory
):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    query = {"rank_model_category_id": rank_model_category_id}
    existing_category = request.app.rank_model_categories_collection.find_one(query)

    if existing_category is None:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.ELEMENT_NOT_FOUND, 
            error_string="rank category not found.", 
            http_status_code=404,
            
        )

    update_fields = {k: v for k, v in update_data.dict(exclude_unset=True).items() if v is not None}

    if not update_fields:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS, 
            error_string="No fields to update.",
            http_status_code=400,
            
        )

    request.app.rank_model_categories_collection.update_one(query, {"$set": update_fields})

    updated_category = request.app.rank_model_categories_collection.find_one(query)
    updated_category = {k: str(v) if isinstance(v, ObjectId) else v for k, v in updated_category.items()}

    # Adjust order of the keys
    ordered_response = {
        "_id": updated_category.pop("_id"),
        "rank_model_category_id": updated_category.pop("rank_model_category_id"),
        **updated_category
    }

    return response_handler.create_success_response_v1(
                                                       response_data=ordered_response, 
                                                       http_status_code=200,
                                                       )


@router.delete("/ab-rank-categories/remove-rank-category/{rank_model_category_id}", 
               tags=["ab-rank-categories"], 
               description="Remove rank category with rank_model_category_id", 
               status_code=200,
               response_model=StandardSuccessResponseV1[WasPresentResponse],
               responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def delete_rank_model_category(request: Request, rank_model_category_id: int):
    response_handler = ApiResponseHandlerV1(request)

    # Check if the rank category exists
    category_query = {"rank_model_category_id": rank_model_category_id}
    category = request.app.rank_model_categories_collection.find_one(category_query)

    if category is None:
        # Return standard response with wasPresent: false
        return response_handler.create_success_delete_response_v1(
                                                           False,
                                                           http_status_code=200,
                                                           )

    # Check if the rank category is used in any ranks
    rank_model_query = {"rank_model_category_id": rank_model_category_id}
    rank_model_with_category = request.app.rank_model_models_collection.find_one(rank_model_query)

    if rank_model_with_category is not None:
        # Since it's used in ranks, do not delete but notify the client
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="Cannot remove rank category, it is already used in ranks.",
            http_status_code=400,
            
        )

    # Remove the rank category
    request.app.rank_model_categories_collection.delete_one(category_query)

    # Return standard response with wasPresent: true
    return response_handler.create_success_delete_response_v1(
                                                       True,
                                                       http_status_code=200,
                                                       )



@router.put("/ab-rank/update-deprecated-status", 
            tags=["ab-rank"],
            status_code=200,
            description="Update the 'deprecated' status of a rank model.",
            response_model=StandardSuccessResponseV1[Rankmodel],
            responses=ApiResponseHandlerV1.listErrors([400, 404, 422, 500]))
def update_rank_model_deprecated_status(request: Request, rank_model_id: int, deprecated: bool):
    response_handler = ApiResponseHandlerV1(request)

    query = {"rank_model_id": rank_model_id}
    existing_rank = request.app.rank_model_models_collection.find_one(query)

    if existing_rank is None:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.ELEMENT_NOT_FOUND, 
            error_string="rank not found.", 
            http_status_code=404,
        )

    if existing_rank.get("deprecated", False) == deprecated:
        # If the existing 'deprecated' status matches the input, return a message indicating no change
        message = "The 'deprecated' status for rank_model_id {} is already set to {}.".format(rank_model_id, deprecated)
        return response_handler.create_success_response_v1(
            response_data={"message": message}, 
            http_status_code=200,
        )

    # Update the 'deprecated' status of the rank
    request.app.rank_model_models_collection.update_one(query, {"$set": {"deprecated": deprecated}})

    # Retrieve the updated rank to confirm the change
    updated_rank = request.app.rank_model_models_collection.find_one(query)

    # Serialize ObjectId to string if necessary and prepare the response
    updated_rank = {k: str(v) if isinstance(v, ObjectId) else v for k, v in updated_rank.items()}

    # Return the updated rank object with a success message
    return response_handler.create_success_response_v1(
        response_data= updated_rank,
        http_status_code=200,
    )

@router.put("/ab-rank-categories/update-deprecated-status",  
            tags=["ab-rank-categories"],
            status_code=200,
            description="Set the 'deprecated' status of a rank category.",
            response_model=StandardSuccessResponseV1[RankCategory],
            responses=ApiResponseHandlerV1.listErrors([400, 404, 422, 500]))
def update_rank_model_category_deprecated_status(request: Request, rank_model_category_id: int, deprecated: bool):
    response_handler = ApiResponseHandlerV1(request)

    query = {"rank_model_category_id": rank_model_category_id}
    existing_rank_model_category = request.app.rank_model_categories_collection.find_one(query)

    if existing_rank_model_category is None:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.ELEMENT_NOT_FOUND, 
            error_string="rank category not found.", 
            http_status_code=404,
        )

    if existing_rank_model_category.get("deprecated", False) == deprecated:
        # If the existing 'deprecated' status matches the input, return a message indicating no change
        message = "The 'deprecated' status for rank_model_category_id {} is already set to {}.".format(rank_model_category_id, deprecated)
        return response_handler.create_success_response_v1(
            response_data={"message": message}, 
            http_status_code=200,
        )

    # Update the 'deprecated' status of the rank category
    request.app.rank_model_categories_collection.update_one(query, {"$set": {"deprecated": deprecated}})

    # Retrieve the updated rank category to confirm the change
    updated_rank_model_category = request.app.rank_model_categories_collection.find_one(query)

    # Serialize ObjectId to string if necessary and prepare the response
    updated_rank_model_category = {k: str(v) if isinstance(v, ObjectId) else v for k, v in updated_rank_model_category.items()}

    # Return the updated rank category object with a success message
    return response_handler.create_success_response_v1(
        response_data= updated_rank_model_category, 
        http_status_code=200,
    )


@router.get('/rank-training/get-ab-rank-image-pair-v1',
            status_code=200,
            tags=["rank-training"],
            description="Get image pair for ab rank with parameters",
            response_model=StandardSuccessResponseV1[ABRankImagePairResponse_v1],
            responses=ApiResponseHandlerV1.listErrors([400, 500]))
async def get_ab_rank_image_pair_v1(
    request: Request, 
    rank_model_id: int, 
    min_score: float, 
    max_diff: float, 
    sample_size: int = 1000,
    start_date: str = None,
    end_date: str = None
):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        # Validate the provided start_date and end_date
        if start_date:
            validated_start_date = validate_date_format(start_date)
            if validated_start_date is None:
                return response_handler.create_error_response_v1(
                    error_code=ErrorCode.INVALID_PARAMS, 
                    error_string="Invalid start_date format. Expected format: YYYY-MM-DDTHH:MM:SS", 
                    http_status_code=400
                )
        if end_date:
            validated_end_date = validate_date_format(end_date)
            if validated_end_date is None:
                return response_handler.create_error_response_v1(
                    error_code=ErrorCode.INVALID_PARAMS, 
                    error_string="Invalid end_date format. Expected format: YYYY-MM-DDTHH:MM:SS",
                    http_status_code=400
                )

        rank_model = request.app.rank_model_models_collection.find_one({'rank_model_id': rank_model_id})
        if rank_model is None:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="Rank model not found.",
                http_status_code=404
            )
        
        classifier_id = rank_model.get("classifier_id")
        if classifier_id is None:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="Rank model does not have a classifier_id.",
                http_status_code=404
            )
        
        # Build the query for filtering classifier scores
        query = {
            'classifier_id': classifier_id,
            'score': {'$gte': min_score}
        }
        if start_date and end_date:
            query["creation_time"] = {"$gte": validated_start_date, "$lte": validated_end_date}
        elif start_date:
            query["creation_time"] = {"$gte": validated_start_date}
        elif end_date:
            query["creation_time"] = {"$lte": validated_end_date}
        
        while True:
            filtered_classfier_scores = list(request.app.image_classifier_scores_collection.aggregate([
                {'$sample': {
                    'size': sample_size
                }},
                {'$match': query},
                {'$project': {
                    'uuid': 1,
                    'score': 1
                }},
            ]))

            if len(filtered_classfier_scores) < 2:
                continue

            unique_filtered_classifier_scores = []
            uuid_set = set()

            for document in filtered_classfier_scores:
                uuid = document.get('uuid')
                if uuid not in uuid_set:
                    uuid_set.add(uuid)
                    unique_filtered_classifier_scores.append(document)
            filtered_classfier_scores = unique_filtered_classifier_scores

            scores = [classifier_score['score'] for classifier_score in filtered_classfier_scores]
            
            # Sort the scores and filtered_classfier_scores
            sorted_args = np.argsort(scores)
            scores = [scores[i] for i in sorted_args]
            filtered_classfier_scores = [filtered_classfier_scores[i] for i in sorted_args]

            num_filtered_classifier_scores = len(filtered_classfier_scores)

            image_pair_list = []
            
            for i in range(num_filtered_classifier_scores):
                next = bisect.bisect_left(scores, scores[i] + max_diff, i, num_filtered_classifier_scores)

                for j in range(i+1, next):
                    image_pair_list.append((i, j))

            num_image_pair_within_max_diff = len(image_pair_list)

            if num_image_pair_within_max_diff > 0:
                break

        image_pair = image_pair_list[np.random.randint(0, num_image_pair_within_max_diff)]

        image_1 = request.app.completed_jobs_collection.find_one({'uuid': filtered_classfier_scores[image_pair[0]]['uuid']})
        image_2 = request.app.completed_jobs_collection.find_one({'uuid': filtered_classfier_scores[image_pair[1]]['uuid']})
        
        image_1.pop('_id')
        image_2.pop('_id')

        image_1['classifier_score'] = filtered_classfier_scores[image_pair[0]]['score']
        image_2['classifier_score'] = filtered_classfier_scores[image_pair[1]]['score']

        return response_handler.create_success_response_v1(
            response_data={'image_1': image_1,
                            'image_2': image_2,
                            'num_images_above_min_score': num_filtered_classifier_scores, 
                            'num_image_pair_within_max_diff': num_image_pair_within_max_diff},
            http_status_code=200
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=f'Failed to get image pair for ab rank {e}',
            http_status_code=500
        )
    
    

@router.get('/ab-rank/get-ab-rank-image-pair',
            status_code=200,
            tags=["ab-rank"],
            description="Get image pair for ab rank with parameters",
            response_model=StandardSuccessResponseV1[ABRankImagePairResponse],
            responses=ApiResponseHandlerV1.listErrors([400, 500]))
async def get_ab_rank_image_pair(request: Request, rank_model_id:int, min_score:float, max_diff:float, sample_size:int=1000):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        rank_model = request.app.rank_model_models_collection.find_one({'rank_model_id': rank_model_id})
        if rank_model is None:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="Rank model not found.",
                http_status_code=404
            )
        if rank_model.get("classifier_id") is None:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="Rank model does not have a classifier_id.",
                http_status_code=404
            )
        
        classifier_id = rank_model.get("classifier_id")
        # Get all classifier scores for the classifier_id
        # but sample aggregate query select the same columns multiple times so we need to filter
        filtered_classfier_scores = list(request.app.image_classifier_scores_collection.aggregate([
            {'$sample': {
                'size': sample_size
            }}, 
            {'$match': {
                'classifier_id': classifier_id,
                'score': {'$gte': min_score},
            }},
            {
            '$group': {
                '_id': '$uuid',
                'data': {'$first': '$$ROOT'}
            }
            },
            {
                '$replaceRoot': {
                    'newRoot': '$data'
                }
            },
            {'$project': {
                '_id': 0
            }},
        ]))

        scores = [classifier_score['score'] for classifier_score in filtered_classfier_scores]
        
        # Sort the scores and filtered_classfier_scores
        sorted_args = np.argsort(scores)
        scores = [scores[i] for i in sorted_args]
        filtered_classfier_scores = [filtered_classfier_scores[i] for i in sorted_args]

        num_filtered_classifier_scores = len(filtered_classfier_scores)
        if num_filtered_classifier_scores < 2:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="No image pair found.",
                http_status_code=404
            )

        image_uuid_pair_list = []
        
        for i in range(num_filtered_classifier_scores):
            next = bisect.bisect_left(scores, scores[i] + max_diff, i, num_filtered_classifier_scores)

            for j in range(i+1, next):
                image_uuid_pair_list.append((filtered_classfier_scores[i]['uuid'], filtered_classfier_scores[j]['uuid']))

        num_image_pair_within_max_diff = len(image_uuid_pair_list)
        
        if num_image_pair_within_max_diff == 0:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="No image pair found.",
                http_status_code=404
            )
            
        image_uuid_pair = image_uuid_pair_list[np.random.randint(0, num_image_pair_within_max_diff)]

        image_uuids = image_uuid_pair
        images = list(request.app.completed_jobs_collection.aggregate([
            {
                '$match': {
                    'uuid': {'$in': image_uuids}
                }
            },
            {
            '$group': {
                '_id': '$uuid',  
                'data': {'$first': '$$ROOT'} 
            }
            },
            {
                '$replaceRoot': {
                    'newRoot': '$data'
                }
            },
        ]))

        image_pair = []
        image_score_pair = []

        for image in images:
            image_dict = image.copy()  # Create a copy of the document
            del image_dict['_id']
            image_uuid = image_dict['uuid']
            image_score = request.app.image_classifier_scores_collection.find_one({'uuid': image_uuid})['score']
            image_pair.append(image_dict)
            image_score_pair.append(image_score)

        return response_handler.create_success_response_v1(
            response_data={'image_pair': image_pair,
                           'image_score_pair': image_score_pair,
                           'num_images_above_min_score': num_filtered_classifier_scores, 
                           'num_image_pair_within_max_diff': num_image_pair_within_max_diff},
            http_status_code=200
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=f'Failed to get image pair for ab rank {e}',
            http_status_code=500
        )
    
@router.get('/ab-rank/get-ab-rank-image-pair-v1',
            status_code=200,
            tags=["ab-rank"],
            description="Get image pair for ab rank with parameters",
            response_model=StandardSuccessResponseV1[dict],
            responses=ApiResponseHandlerV1.listErrors([400, 500]))
async def get_ab_rank_image_pair(request: Request, rank_model_id: int, min_score: float, max_diff: float, sample_size: int = 1000):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        # Fetch the rank model
        rank_model = request.app.rank_model_models_collection.find_one({'rank_model_id': rank_model_id})
        if rank_model is None:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="Rank model not found.",
                http_status_code=404
            )
        if rank_model.get("classifier_id") is None:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="Rank model does not have a classifier_id.",
                http_status_code=404
            )

        classifier_id = rank_model.get("classifier_id")

        # Fetch classifier scores
        filtered_classifier_scores = list(request.app.image_classifier_scores_collection.aggregate([
            {'$match': {
                'classifier_id': classifier_id,
                'score': {'$gte': min_score},
            }},
            {'$sample': {
                'size': sample_size
            }},
            {'$project': {
                '_id': 0
            }},
        ]))

        if len(filtered_classifier_scores) < 2:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="Not enough images found with the given parameters.",
                http_status_code=404
            )

        scores = [classifier_score['score'] for classifier_score in filtered_classifier_scores]

        # Sort the scores and filtered_classifier_scores
        sorted_indices = np.argsort(scores)
        scores = [scores[i] for i in sorted_indices]
        filtered_classifier_scores = [filtered_classifier_scores[i] for i in sorted_indices]

        image_uuid_pair_list = []

        for i in range(len(filtered_classifier_scores)):
            for j in range(i + 1, len(filtered_classifier_scores)):
                if abs(scores[j] - scores[i]) <= max_diff:
                    image_uuid_pair_list.append((filtered_classifier_scores[i]['uuid'], filtered_classifier_scores[j]['uuid']))
                else:
                    break

        if not image_uuid_pair_list:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="No image pairs found within the given score difference.",
                http_status_code=404
            )

        num_image_pair_within_max_diff = len(image_uuid_pair_list)
        image_uuid_pair = image_uuid_pair_list[np.random.randint(0, num_image_pair_within_max_diff)]

        image_uuids = list(image_uuid_pair)
        images = list(request.app.completed_jobs_collection.find({'uuid': {'$in': image_uuids}}))

        if len(images) != 2:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="Error in fetching image pairs.",
                http_status_code=404
            )

        image_pair = []
        for image in images:
            image_dict = image.copy()  # Create a copy of the document
            del image_dict['_id']
            image_uuid = image_dict['uuid']
            image_score = request.app.image_classifier_scores_collection.find_one({'uuid': image_uuid})['score']
            image_dict['image_classifier_score'] = image_score
            image_pair.append(image_dict)

        return response_handler.create_success_response_v1(
            response_data={
                'image_pair': image_pair,
                'num_images_above_min_score': len(filtered_classifier_scores),
                'num_image_pair_within_max_diff': num_image_pair_within_max_diff
            },
            http_status_code=200
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=f'Failed to get image pair for ab rank: {e}',
            http_status_code=500
        )
