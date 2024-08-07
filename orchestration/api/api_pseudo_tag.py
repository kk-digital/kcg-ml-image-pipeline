from datetime import datetime
from fastapi import APIRouter, Request, HTTPException, Query
from typing import List, Dict
from orchestration.api.mongo_schema.pseudo_tag_schemas import ImagePseudoTagRequest, ImagePseudoTag, ListImagePseudoTag, ImagePseudoTagRequestV1, ListImagePseudoTagScores
from typing import Union
from .api_utils import PrettyJSONResponse, validate_date_format, ApiResponseHandlerV1, ErrorCode, StandardSuccessResponseV1, WasPresentResponse, VectorIndexUpdateRequest, PseudoTagIdResponse, TagCountResponse, ListImageTag
import traceback
from bson import ObjectId
import logging
import time
from typing import Optional



router = APIRouter()


@router.post("/pseudotag/add-pseudo-tag-to-image-v1", 
             tags=["pseudo_tags"], 
             response_model=StandardSuccessResponseV1[ImagePseudoTagRequest],
             responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
async def add_pseudo_tag_to_image_v1(
    request: Request, 
    pseudo_tag: ImagePseudoTagRequest, 
    image_source: str = Query(..., regex="^(generated_image|extract_image|external_image)$")
):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        # Determine the appropriate collection based on image_source
        if image_source == "generated_image":
            collection = request.app.completed_jobs_collection
            projection = {"task_output_file_dict.output_file_hash": 1, "task_type": 1}
        elif image_source == "extract_image":
            collection = request.app.extracts_collection
            projection = {"image_hash": 1}
        elif image_source == "external_image":
            collection = request.app.external_images_collection
            projection = {"image_hash": 1}
        else:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string="Invalid image source provided.",
                http_status_code=422
            )

        # Fetch image_hash from the determined collection
        job_data = collection.find_one({"uuid": pseudo_tag.uuid}, projection)
        if not job_data:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string="The provided UUID does not have an associated image hash.",
                http_status_code=404
            )

        if image_source == "generated_image":
            if 'task_output_file_dict' not in job_data or 'output_file_hash' not in job_data['task_output_file_dict']:
                return response_handler.create_error_response_v1(
                    error_code=ErrorCode.INVALID_PARAMS,
                    error_string="The provided UUID does not have an associated image hash.",
                    http_status_code=404
                )
            image_hash = job_data['task_output_file_dict']['output_file_hash']
            task_type = job_data.get('task_type', None)
        else:
            image_hash = job_data['image_hash']
            task_type = None

        # Fetch tag_id from classifier_models_collection using classifier_id
        classifier_data = request.app.classifier_models_collection.find_one({"classifier_id": pseudo_tag.classifier_id})
        if not classifier_data or 'tag_id' not in classifier_data:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string="The provided classifier ID does not have an associated tag ID.",
                http_status_code=404
            )
        
        tag_id = classifier_data['tag_id']

        # Check for existing pseudo tag
        existing_pseudo_tag = request.app.pseudo_tag_images_collection.find_one({"tag_id": tag_id, "image_hash": image_hash, "image_source": image_source})
        if existing_pseudo_tag:
            # Update the score if pseudo tag already exists
            request.app.pseudo_tag_images_collection.update_one(
                {"_id": existing_pseudo_tag["_id"]}, 
                {"$set": {
                    "classifier_id": pseudo_tag.classifier_id,
                    "score": pseudo_tag.score
                }}
            )

            # Retrieve the updated document to include in the response
            updated_pseudo_tag = request.app.pseudo_tag_images_collection.find_one({"_id": existing_pseudo_tag["_id"]})
            
            # Prepare the document for the response
            updated_pseudo_tag.pop('_id', None)  
            
            return response_handler.create_success_response_v1(
                response_data=updated_pseudo_tag,  # Return the updated object
                http_status_code=200)

        # Add new pseudo tag
        new_pseudo_tag_data = {
            "uuid": pseudo_tag.uuid,
            "task_type": task_type,
            "classifier_id": pseudo_tag.classifier_id,
            "tag_id": tag_id,
            "image_hash": image_hash,
            "score": pseudo_tag.score,
            "creation_time": datetime.utcnow().isoformat(),
            "image_source": image_source
        }
        insert_result = request.app.pseudo_tag_images_collection.insert_one(new_pseudo_tag_data)
        inserted_id = insert_result.inserted_id

        inserted_doc = request.app.pseudo_tag_images_collection.find_one({"_id": inserted_id})

        inserted_doc_dict = dict(inserted_doc)
        inserted_doc_dict.pop('_id', None)  

        return response_handler.create_success_response_v1(
            response_data=inserted_doc_dict,
            http_status_code=200)

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=f"Internal server error: {str(e)}", 
            http_status_code=500)


@router.post("/pseudotag/set-image-pseudotag-score", 
             status_code=200,
             response_model=StandardSuccessResponseV1[ImagePseudoTagRequestV1],
             description="changed with changed with /pseudotag/set-image-pseudotag-score-v1",
             tags=["deprecated3"], 
             responses=ApiResponseHandlerV1.listErrors([404, 422, 500]) 
             )
async def set_image_pseudotag_score(request: Request, pseudo_tag: ImagePseudoTagRequestV1):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:

        # Fetch image_hash from completed_jobs_collection
        job_data = request.app.completed_jobs_collection.find_one({"uuid": pseudo_tag.job_uuid},  {"task_output_file_dict.output_file_hash": 1, "task_type": 1})
        if not job_data or 'task_output_file_dict' not in job_data or 'output_file_hash' not in job_data['task_output_file_dict']:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string="The provided UUID does not have an associated image hash.",
                http_status_code=404
            )
        image_hash = job_data['task_output_file_dict']['output_file_hash']
        task_type = job_data['task_type']

        # Fetch tag_id from classifier_models_collection
        classifier_data = request.app.classifier_models_collection.find_one({"classifier_id": pseudo_tag.classifier_id}, {"tag_id": 1})
        if not classifier_data:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string="The provided classifier ID does not exist.",
                http_status_code=404
            )
        tag_id = classifier_data['tag_id']

        
        query = {
            "uuid": pseudo_tag.job_uuid,
            "tag_id": tag_id
        }

        # Get current UTC time in ISO format
        current_utc_time = datetime.utcnow().isoformat()

        # Initialize new_score_data outside of the if/else block
        new_score_data = {
            "uuid": pseudo_tag.job_uuid,
            "task_type": task_type,
            "classifier_id": pseudo_tag.classifier_id,
            "image_hash": image_hash,
            "tag_id": tag_id,
            "score": pseudo_tag.score,
            "creation_time": current_utc_time
        }

        # Check for existing score and update or insert accordingly
        existing_score = request.app.pseudo_tag_images_collection.find_one(query)
        if existing_score:
            # Update existing score
            request.app.pseudo_tag_images_collection.update_one(query, {"$set": {"classifier_id": pseudo_tag.classifier_id, "score": pseudo_tag.score, "image_hash": image_hash, "creation_time": current_utc_time}})
        else:
            # Insert new score
            insert_result = request.app.pseudo_tag_images_collection.insert_one(new_score_data)
            new_score_data['_id'] = str(insert_result.inserted_id)

        return api_response_handler.create_success_response_v1(
            response_data=new_score_data,
            http_status_code=200  
        )
    
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )

@router.post("/pseudotag/set-image-pseudotag-score-v1", 
             status_code=200,
             response_model=StandardSuccessResponseV1[ImagePseudoTagRequestV1],
             description="Set image pseudotag score",
             tags=["pseudo_tags"], 
             responses=ApiResponseHandlerV1.listErrors([404, 422, 500]) 
             )
async def set_image_pseudotag_score_V1(
    request: Request, 
    pseudo_tag: ImagePseudoTagRequestV1,
    image_source: str = Query(..., regex="^(generated_image|extract_image|external_image)$")
):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        # Determine the appropriate collection based on image_source
        if image_source == "generated_image":
            collection = request.app.completed_jobs_collection
            projection = {"task_output_file_dict.output_file_hash": 1, "task_type": 1}
        elif image_source == "extract_image":
            collection = request.app.extracts_collection
            projection = {"image_hash": 1}
        elif image_source == "external_image":
            collection = request.app.external_images_collection
            projection = {"image_hash": 1}
        else:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string="Invalid image source provided.",
                http_status_code=422
            )

        # Fetch image_hash from the determined collection
        job_data = collection.find_one({"uuid": pseudo_tag.job_uuid}, projection)
        if not job_data:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string="The provided UUID does not have an associated image hash.",
                http_status_code=404
            )

        if image_source == "generated_image":
            if 'task_output_file_dict' not in job_data or 'output_file_hash' not in job_data['task_output_file_dict']:
                return api_response_handler.create_error_response_v1(
                    error_code=ErrorCode.INVALID_PARAMS,
                    error_string="The provided UUID does not have an associated image hash.",
                    http_status_code=404
                )
            image_hash = job_data['task_output_file_dict']['output_file_hash']
            task_type = job_data.get('task_type', None)
        else:
            image_hash = job_data['image_hash']
            task_type = None

        # Fetch tag_id from classifier_models_collection
        classifier_data = request.app.classifier_models_collection.find_one(
            {"classifier_id": pseudo_tag.classifier_id}, 
            {"tag_id": 1}
        )
        if not classifier_data:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string="The provided classifier ID does not exist.",
                http_status_code=404
            )
        tag_id = classifier_data['tag_id']

        query = {
            "uuid": pseudo_tag.job_uuid,
            "tag_id": tag_id,
            "image_source": image_source
        }

        # Get current UTC time in ISO format
        current_utc_time = datetime.utcnow().isoformat()

        # Initialize new_score_data
        new_score_data = {
            "uuid": pseudo_tag.job_uuid,
            "task_type": task_type,
            "classifier_id": pseudo_tag.classifier_id,
            "image_hash": image_hash,
            "tag_id": tag_id,
            "score": pseudo_tag.score,
            "creation_time": current_utc_time,
            "image_source": image_source
        }

        # Check for existing score and update or insert accordingly
        existing_score = request.app.pseudo_tag_images_collection.find_one(query)
        if existing_score:
            # Update existing score
            request.app.pseudo_tag_images_collection.update_one(
                query, 
                {"$set": {
                    "classifier_id": pseudo_tag.classifier_id, 
                    "score": pseudo_tag.score, 
                    "image_hash": image_hash, 
                    "creation_time": current_utc_time,
                    "image_source": image_source
                }}
            )
        else:
            # Insert new score
            insert_result = request.app.pseudo_tag_images_collection.insert_one(new_score_data)
            new_score_data['_id'] = str(insert_result.inserted_id)

        return api_response_handler.create_success_response_v1(
            response_data=new_score_data,
            http_status_code=200  
        )
    
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )







@router.get("/pseudo-tag/count-task-type", 
            response_model=StandardSuccessResponseV1[dict],
            status_code=200,
            tags=["deprecated2"],
            description="Counts the number of documents in the image classifier scores collection that contain the 'task_type' field",
            responses=ApiResponseHandlerV1.listErrors([500]))
async def count_classifier_scores(request: Request):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        # Count documents that include the 'task_type' field
        count = request.app.pseudo_tag_images_collection.count_documents({"task_type": {"$exists": True}})

        return api_response_handler.create_success_response_v1(
            response_data={"count": count},
            http_status_code=200  
        )
    
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )