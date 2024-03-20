from datetime import datetime
from fastapi import APIRouter, Request, HTTPException, Query
from typing import List, Dict
from orchestration.api.mongo_schema.tag_schemas import TagDefinition, ImageTag, TagCategory, NewTagRequest, NewTagCategory
from .mongo_schemas import Classifier
from typing import Union
from .api_utils import PrettyJSONResponse, validate_date_format, ApiResponseHandler, ErrorCode, StandardSuccessResponseV1, ApiResponseHandlerV1
import traceback
from bson import ObjectId
from fastapi.encoders import jsonable_encoder



router = APIRouter()

@router.post("/classifier/register-pseudo-tag-classifier", 
             tags=["classifier"],
             description="Adds or updates a classifier model",
             responses=ApiResponseHandlerV1.listErrors([400, 500]))
async def add_update_classifier(request: Request, classifier_data: Classifier):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:

        # Check if the tag_id exists in tag_definitions_collection
        tag_definition_exists = request.app.tag_definitions_collection.find_one({"tag_id": classifier_data.tag_id})
        if not tag_definition_exists:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS, 
                error_string=f"Tag ID {classifier_data.tag_id} not found in tag definitions.",
                http_status_code=400
            )

        existing_classifier = request.app.classifier_models_collection.find_one(
            {"classifier_id": classifier_data.classifier_id}
        )
        
        if existing_classifier:
            new_seq_number = existing_classifier.get("model_sequence_number", 0) + 1
            update_result = request.app.classifier_models_collection.update_one(
                {"classifier_id": classifier_data.classifier_id},
                {"$set": {
                    "classifier_id": classifier_data.classifier_id,
                    "classifier_name": classifier_data.classifier_name,
                    "tag_id": classifier_data.tag_id,
                    "model_sequence_number": new_seq_number,
                    "latest_model": classifier_data.latest_model,
                    "model_path": classifier_data.model_path
                }}
            )
            if update_result.modified_count == 0:
                return response_handler.create_error_response_v1(
                    error_code=ErrorCode.INVALID_PARAMS, 
                    error_string="Classifier update failed.",
                    http_status_code=500
                )
            return response_handler.create_success_response_v1(
                response_data={"message": "Classifier updated successfully.", "classifier_id": classifier_data.classifier_id, "new_model_sequence_number": new_seq_number},
                http_status_code=200
            )
        else:
            classifier_data.model_sequence_number = 0
            request.app.classifier_models_collection.insert_one(classifier_data.dict())
            return response_handler.create_success_response_v1(
                response_data={"message": "New classifier added successfully.", "classifier_id": classifier_data.classifier_id, "model_sequence_number": 1},
                http_status_code=201
            )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=f"Failed to add/update classifier: {str(e)}",
            http_status_code=500
        )
    
    
@router.get("/classifier/list-classifiers", 
            response_model=StandardSuccessResponseV1[List[Classifier]],
            description="list Classifiers",
            tags=["classifier"],
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([500]))
async def list_classifiers(request: Request):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        classifier_cursor = request.app.classifier_models_collection.find({})
        classifier_list = list(classifier_cursor)

        result = []
        for classifier in classifier_list:
            # Convert MongoDB's ObjectId to string if needed, otherwise prepare as is
            classifier['_id'] = str(classifier['_id'])
            result.append(classifier)

        return response_handler.create_success_response_v1(response_data=result, http_status_code=200)

    except Exception as e:
        # Implement appropriate error handling
        print(f"Error: {str(e)}")
        return response_handler.create_error_response_v1(error_code=ErrorCode.OTHER_ERROR, error_string="Internal server error", http_status_code=500)

@router.delete("/classifier/remove-classifier-with-id", 
               tags=["classifier"],
               description="Deletes a classifier model by its classifier_id",
               responses=ApiResponseHandlerV1.listErrors([400, 404, 500]))
async def delete_classifier(request: Request, classifier_id: int ):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    # Attempt to delete the classifier with the specified classifier_id
    delete_result = request.app.classifier_models_collection.delete_one({"classifier_id": classifier_id})

    # Check if a document was deleted
    if delete_result.deleted_count == 0:
        # No document was found with the provided classifier_id, so nothing was deleted
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.ELEMENT_NOT_FOUND,
            error_string=f"No classifier found with classifier_id {classifier_id}.",
            http_status_code=404
        )

    # If a document was successfully deleted
    return response_handler.create_success_response_v1(
        response_data={"message": f"Classifier with classifier_id {classifier_id} was successfully deleted."},
        http_status_code=200
    )