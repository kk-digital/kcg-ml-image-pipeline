from datetime import datetime
from fastapi import APIRouter, Request, HTTPException, Query
from typing import List, Dict
from .mongo_schemas import Classifier, RequestClassifier, ListClassifier
from typing import Union
from .api_utils import PrettyJSONResponse, validate_date_format, ApiResponseHandler, ErrorCode, StandardSuccessResponseV1, ApiResponseHandlerV1, WasPresentResponse
import traceback
from bson import ObjectId
from fastapi.encoders import jsonable_encoder



router = APIRouter()

def get_next_classifier_id_sequence(request: Request):
    # get classifier counter
    counter = request.app.counters_collection.find_one({"_id": "classifiers"})
    # create counter if it doesn't exist already
    if counter is None:
        request.app.counters_collection.insert_one({"_id": "classifiers", "seq":0})
    counter_seq = counter["seq"] if counter else 0 
    counter_seq += 1

    try:
        ret = request.app.counters_collection.update_one(
            {"_id": "classifiers"},
            {"$set": {"seq": counter_seq}})
    except Exception as e:
        raise Exception("Updating of classifier counter failed: {}".format(e))

    return counter_seq

@router.post("/classifier/register-tag-classifier", 
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

        # Check if an existing classifier can be updated
        existing_classifier = request.app.classifier_models_collection.find_one(
            {"tag_id": classifier_data.tag_id, "classifier_name": classifier_data.classifier_name}
        )
        
        if existing_classifier:
            new_seq_number = existing_classifier.get("model_sequence_number", 0) + 1
            classifier_id = existing_classifier.get("classifier_id")
            update_result = request.app.classifier_models_collection.update_one(
                {"classifier_id": classifier_id},
                {"$set": {
                    "classifier_id": classifier_id,
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
                response_data={"message": "Classifier updated successfully.", "classifier_id": classifier_id, "new_model_sequence_number": new_seq_number},
                http_status_code=200
            )
        else:
            classifier_data.model_sequence_number = 1
            classifier_data.classifier_id= get_next_classifier_id_sequence(request)
            request.app.classifier_models_collection.insert_one(classifier_data.to_dict())
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



# Updated apis


@router.post("/pseudotag-classifiers/register-tag-classifier", 
             tags=["pseudotag classifier"],
             description="Adds a new classifier model",
             response_model=StandardSuccessResponseV1[Classifier],
             responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
async def create_classifier(request: Request, request_classifier_data: RequestClassifier):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        # Verify tag_id exists in tag_definitions_collection
        if not request.app.tag_definitions_collection.find_one({"tag_id": request_classifier_data.tag_id}):
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS, 
                error_string=f"Tag ID {request_classifier_data.tag_id} not found in tag definitions.",
                http_status_code=400
            )

        # Calculate internal fields
        new_classifier_id = get_next_classifier_id_sequence(request)
        new_model_sequence_number = 0  
        creation_time = datetime.now().isoformat()

        # Merge user-provided data with calculated values
        full_classifier_data = Classifier(
            classifier_id=new_classifier_id,
            classifier_name=request_classifier_data.classifier_name,
            tag_id=request_classifier_data.tag_id,
            model_sequence_number=new_model_sequence_number,
            latest_model=request_classifier_data.latest_model,
            model_path=request_classifier_data.model_path,
            creation_time=creation_time
        )

        # Insert new classifier into the collection
        request.app.classifier_models_collection.insert_one(full_classifier_data.dict())
        return response_handler.create_success_response_v1(
            response_data=full_classifier_data.dict(),
            http_status_code=201
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=f"Failed to add classifier: {str(e)}",
            http_status_code=500
        )


@router.patch("/pseudotag-classifiers/update-tag-classifier", 
             tags=["pseudotag classifier"],
             response_model=StandardSuccessResponseV1[Classifier],
             description="Updates an existing classifier model",
             responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
async def update_classifier(request: Request, classifier_id: int, update_data: RequestClassifier):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        existing_classifier = request.app.classifier_models_collection.find_one({"classifier_id": classifier_id})

        if not existing_classifier:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS, 
                error_string="Classifier not found.",
                http_status_code=404
            )
        
        # Increment model sequence number by 1
        new_seq_number = existing_classifier.get("model_sequence_number", 0) + 1

        # Prepare update document, excluding unset fields from update_data
        update_doc = {k: v for k, v in update_data.dict(exclude_unset=True).items()}

        # Add incremented model_sequence_number to update document
        update_doc["model_sequence_number"] = new_seq_number

        # Execute update
        update_result = request.app.classifier_models_collection.update_one(
            {"classifier_id": classifier_id},
            {"$set": update_doc}
        )

        if update_result.modified_count == 0:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS, 
                error_string="No changes made to the classifier.",
                http_status_code=400  
            )

        # Fetch updated classifier model to return as response
        updated_classifier = request.app.classifier_models_collection.find_one({"classifier_id": classifier_id})

        # Ensure ObjectId is converted to string if necessary
        updated_classifier['_id'] = str(updated_classifier['_id'])

        return response_handler.create_success_response_v1(
            response_data=updated_classifier,  # Return updated classifier data
            http_status_code=200
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=f"Failed to update classifier: {str(e)}",
            http_status_code=500
        )



    
@router.get("/pseudotag-classifiers/list-classifiers", 
            response_model=StandardSuccessResponseV1[ListClassifier],
            description="list Classifiers",
            tags=["pseudotag classifier"],
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([422, 500]))
async def list_classifiers(request: Request):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        classifier_cursor = request.app.classifier_models_collection.find({})
        classifier_list = list(classifier_cursor)

        result = []
        for classifier in classifier_list:
            # Convert MongoDB's ObjectId to string if needed, otherwise prepare as is
            classifier['_id'] = str(classifier['_id'])
            classifier.pop('_id', None)
            result.append(classifier)

        return response_handler.create_success_response_v1(response_data={"classifiers": result}, http_status_code=200)

    except Exception as e:
        # Implement appropriate error handling
        print(f"Error: {str(e)}")
        return response_handler.create_error_response_v1(error_code=ErrorCode.OTHER_ERROR, error_string="Internal server error", http_status_code=500)


@router.delete("/pseudotag-classifiers/remove-classifier-with-id", 
               tags=["pseudotag classifier"],
               response_model=StandardSuccessResponseV1[WasPresentResponse], 
               description="Deletes a classifier model by its classifier_id",
               responses=ApiResponseHandlerV1.listErrors([400, 404, 422, 500]))
async def delete_classifier(request: Request, classifier_id: int ):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    # Attempt to delete the classifier with the specified classifier_id
    delete_result = request.app.classifier_models_collection.delete_one({"classifier_id": classifier_id})

    # Check if a document was deleted
    if delete_result.deleted_count == 0:
        # No document was found with the provided classifier_id, so nothing was deleted
        return response_handler.create_success_delete_response_v1(
            False,
            http_status_code=200
        )

    # If a document was successfully deleted
    return response_handler.create_success_delete_response_v1(
        True,
        http_status_code=200
    )
