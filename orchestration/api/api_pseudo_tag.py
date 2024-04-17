from datetime import datetime
from fastapi import APIRouter, Request, HTTPException, Query
from typing import List, Dict
from orchestration.api.mongo_schema.pseudo_tag_schemas import ImagePseudoTagRequest, ImagePseudoTag, ListImagePseudoTag
from typing import Union
from .api_utils import PrettyJSONResponse, validate_date_format, ApiResponseHandlerV1, ErrorCode, StandardSuccessResponseV1, WasPresentResponse, VectorIndexUpdateRequest, PseudoTagIdResponse, TagCountResponse, ListImageTag
import traceback
from bson import ObjectId
import logging
from typing import Optional



router = APIRouter()


@router.get("/pseudotag/get-images-count-by-tag-id", 
            status_code=200,
            tags=["pseudo_tags"], 
            description="Get count of images with a specific pseudo tag",
            response_model=StandardSuccessResponseV1[TagCountResponse],
            responses=ApiResponseHandlerV1.listErrors([400, 422]))
def get_image_count_by_tag(
    request: Request,
    tag_id: int
):
    response_handler = ApiResponseHandlerV1(request)

    # Assuming each image document has an 'tags' array field
    query = {"tag_id": tag_id}
    count = request.app.pseudo_tag_images_collection.count_documents(query)
    
    if count == 0:
        # If no images found with the tag, consider how you want to handle this. 
        # For example, you might still want to return a success response with a count of 0.
        return response_handler.create_success_response_v1(
                                                           response_data={"tag_id": tag_id, "count": 0}, 
                                                           http_status_code=200,
                                                           )

    # Return standard success response with the count
    return response_handler.create_success_response_v1(
                                                       response_data={"tag_id": tag_id, "count": count}, 
                                                       http_status_code=200,
                                                       )



@router.get("/pseudotag/get-images-by-tag-id", 
            tags=["pseudo_tags"], 
            status_code=200,
            description="Get images by tag_id",
            response_model=StandardSuccessResponseV1[ListImagePseudoTag], 
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def get_tagged_images(
    request: Request, 
    tag_id: int,
    start_date: str = None,
    end_date: str = None,
    order: str = Query("desc", description="Order in which the data should be returned. 'asc' for oldest first, 'desc' for newest first")
):
    response_handler = ApiResponseHandlerV1(request)
    try:
        # Validate start_date and end_date
        if start_date:
            validated_start_date = validate_date_format(start_date)
            if validated_start_date is None:
                return response_handler.create_error_response_v1(
                    error_code=ErrorCode.INVALID_PARAMS, 
                    error_string="Invalid start_date format. Expected format: YYYY-MM-DDTHH:MM:SS", 
                    http_status_code=400,
                    
                )
        if end_date:
            validated_end_date = validate_date_format(end_date)
            if validated_end_date is None:
                return response_handler.create_error_response_v1(
                    error_code=ErrorCode.INVALID_PARAMS, 
                    error_string="Invalid end_date format. Expected format: YYYY-MM-DDTHH:MM:SS",
                    http_status_code=400,
                    
                )

        # Build the query
        query = {"tag_id": tag_id}
        if start_date and end_date:
            query["creation_time"] = {"$gte": validated_start_date, "$lte": validated_end_date}
        elif start_date:
            query["creation_time"] = {"$gte": validated_start_date}
        elif end_date:
            query["creation_time"] = {"$lte": validated_end_date}

        # Decide the sort order
        sort_order = -1 if order == "desc" else 1

        # Execute the query
        image_tags_cursor = list(request.app.pseudo_tag_images_collection.find(query).sort("creation_time", sort_order))


        for tag_data in image_tags_cursor:
            tag_data.pop('_id', None)  
   

        # Return the list of images in a standard success response
        return response_handler.create_success_response_v1(
                                                           response_data={"images": image_tags_cursor}, 
                                                           http_status_code=200,
                                                           )

    except Exception as e:
        # Log the exception details here, if necessary
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, error_string=str(e), http_status_code=500
        )


@router.delete("/pseudotag/remove-pseudo-tag-from-image/{tag_id}", 
               status_code=200,
               tags=["pseudo_tags"], 
               description="Remove image pseudotag",
               response_model=StandardSuccessResponseV1[WasPresentResponse],
               responses=ApiResponseHandlerV1.listErrors([400, 422]))
def remove_image_tag(
    request: Request,
    image_hash: str,  
    tag_id: int 
):
    response_handler = ApiResponseHandlerV1(request)

    # The query now checks for the specific tag_id within the array of tags
    query = {"image_hash": image_hash, "tag_id": tag_id}
    result = request.app.pseudo_tag_images_collection.delete_one(query)
    
    # If no document was found and deleted, use response_handler to raise an HTTPException
    if result.deleted_count == 0:
        return response_handler.create_success_response_v1(
            response_data={"wasPresent": False}, 
            http_status_code=200
        )

    # Return standard success response with wasPresent: true using response_handler
    return response_handler.create_success_response_v1(response_data={"wasPresent": True}, http_status_code=200)




@router.post("/pseudotag/add-pseudo-tag-to-image", 
          tags=["pseudo_tags"], 
          response_model=StandardSuccessResponseV1[ImagePseudoTagRequest],
          responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
async def add_pseudo_tag_to_image(request: Request, pseudo_tag: ImagePseudoTagRequest):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        # Fetch image_hash from completed_jobs_collection using uuid
        job_data = request.app.completed_jobs_collection.find_one({"uuid": pseudo_tag.uuid})
        if not job_data or 'task_output_file_dict' not in job_data or 'output_file_hash' not in job_data['task_output_file_dict']:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string="The provided UUID does not have an associated image hash.",
                http_status_code=404)
        image_hash = job_data['task_output_file_dict']['output_file_hash']
        task_type = job_data['task_type']

        # Fetch tag_id from classifier_models_collection using classifier_id
        classifier_data = request.app.classifier_models_collection.find_one({"classifier_id": pseudo_tag.classifier_id})
        if not classifier_data or 'tag_id' not in classifier_data:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string="The provided classifier ID does not have an associated tag ID.",
                http_status_code=404)
        
        tag_id = classifier_data['tag_id']

        # Check for existing pseudo tag
        existing_pseudo_tag = request.app.pseudo_tag_images_collection.find_one({"tag_id": tag_id, "image_hash": image_hash})
        if existing_pseudo_tag:
            # Update the score if pseudo tag already exists
            request.app.pseudo_tag_images_collection.update_one(
            {"_id": existing_pseudo_tag["_id"]}, 
            {
                "$set": {
                    "classifier_id": pseudo_tag.classifier_id,
                    "score": pseudo_tag.score
                }
            }
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
            "creation_time": datetime.utcnow().isoformat()
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

@router.get("/pseudotag/get-pseudo-tag-list-for-image-with-classifier-id", 
            tags=["pseudo_tags"], 
            status_code=200,
            description="list pseudo tags for image",
            response_model=StandardSuccessResponseV1[ListImagePseudoTag],  # Adjust according to your actual response model
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def get_pseudo_tag_list_for_image(request: Request, classifier_id: int):
    response_handler = ApiResponseHandlerV1(request) 
    try:
        # Find all pseudo tags associated with the provided file_hash
        image_tags_cursor = list(request.app.pseudo_tag_images_collection.find({"classifier_id": classifier_id}))

        for image in image_tags_cursor:
                    image.pop('_id', None)  

        # Return the list of pseudo tags for the image
        return response_handler.create_success_response_v1(
            response_data={"images":image_tags_cursor},  
            http_status_code=200,
        )
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )

@router.get("/pseudotag/get-pseudo-tag-list-for-image-with-hash", 
            tags=["pseudo_tags"], 
            status_code=200,
            description="list pseudo tags for image",
            response_model=StandardSuccessResponseV1[ListImagePseudoTag],  # Adjust according to your actual response model
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def get_pseudo_tag_list_for_image(request: Request, file_hash: str):
    response_handler = ApiResponseHandlerV1(request) 
    try:
        # Find all pseudo tags associated with the provided file_hash
        image_tags_cursor = list(request.app.pseudo_tag_images_collection.find({"image_hash": file_hash}))

        for image in image_tags_cursor:
                    image.pop('_id', None)  

        # Return the list of pseudo tags for the image
        return response_handler.create_success_response_v1(
            response_data={"images":image_tags_cursor},  
            http_status_code=200,
        )
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )


@router.get("/pseudotag/get-all-tagged-images", 
            tags=["pseudo_tags"], 
            status_code=200,
            description="Get all tagged images",
            response_model=StandardSuccessResponseV1[ListImagePseudoTag],  
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def get_all_pseudo_tagged_images(request: Request):
    response_handler = ApiResponseHandlerV1(request)

    try:
        # Fetch all documents
        documents = list(request.app.pseudo_tag_images_collection.find({}))
        
        for doc in documents:
            doc.pop('_id', None)  
            

        return response_handler.create_success_response_v1(response_data={"images": documents}, http_status_code=200)
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=f"Internal server error: {str(e)}", 
            http_status_code=500)


@router.post("/pseudotag/batch-update-task-type", 
             response_model=StandardSuccessResponseV1[dict],
             responses=ApiResponseHandlerV1.listErrors([500]))
def batch_update_classifier_scores_with_task_type(request: Request):
    api_response_handler = ApiResponseHandlerV1(request)
    
    try:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()

        # Cursor for iterating over all scores where 'task_type' is not already set
        scores_cursor = request.app.pseudo_tag_images_collection.find({"task_type": {"$exists": False}})

        updated_count = 0
        logger.info("Starting batch update of task types...")
        
        for score in scores_cursor:
            logger.info(f"Processing score with ID: {score['_id']}")

            # Fetch corresponding job using the UUID to get the 'task_type'
            job = request.app.completed_jobs_collection.find_one({"uuid": score["uuid"]}, {"task_type": 1})
            
            if job and 'task_type' in job:
                logger.info(f"Found job with task type: {job['task_type']}")
                
                # Update the score document with the 'task_type'
                update_result = request.app.pseudo_tag_images_collection.update_one(
                    {"_id": score["_id"]},
                    {"$set": {"task_type": job['task_type']}}
                )
                if update_result.modified_count > 0:
                    updated_count += 1
                    logger.info(f"Updated  with new task type: {job['task_type']}")

        logger.info("Completed batch update.")
        return api_response_handler.create_success_response_v1(
            response_data={"updated_count": updated_count},
            http_status_code=200
        )
    
    except Exception as e:
        logger.error(f"Batch update failed: {str(e)}")
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=f"Failed to batch update classifier scores: {str(e)}",
            http_status_code=500
        )


@router.get("/pseudotag/list-images-by-scores", 
            description="List image scores based on classifier",
            tags=["pseudo_tags"],  
            response_model=StandardSuccessResponseV1[ListImagePseudoTag],  
            responses=ApiResponseHandlerV1.listErrors([400, 422]))
async def list_image_scores(
    request: Request,
    classifier_id: Optional[int] = Query(None, description="Filter by classifier ID"),
    min_score: Optional[float] = Query(None, description="Minimum score"),
    max_score: Optional[float] = Query(None, description="Maximum score"),
    limit: int = Query(10, description="Limit on the number of results returned"),
    offset: int = Query(0, description="Offset for pagination"),
    order: str = Query("desc", description="Sort order: 'asc' for ascending, 'desc' for descending"),
    random_sampling: bool = Query(True, description="Enable random sampling")
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

    # Modify behavior based on random_sampling parameter
    if random_sampling:
        # Fetch data without sorting when random_sampling is True
        cursor = request.app.pseudo_tag_images_collection.aggregate([
            {"$match": query},
            {"$sample": {"size": limit}}  # Use the MongoDB $sample operator for random sampling
        ])
    else:
        # Determine sort order and fetch sorted data when random_sampling is False
        sort_order = 1 if order == "asc" else -1
        cursor = request.app.pseudo_tag_images_collection.find(query).sort([("score", sort_order)]).skip(offset).limit(limit)
    
    scores_data = list(cursor)

    # Remove _id in response data
    for score in scores_data:
        score.pop('_id', None)

    # Prepare the data for the response
    images_data = ListImagePseudoTag(images=[ImagePseudoTag(**doc).to_dict() for doc in scores_data]).dict()

    # Return the fetched data with a success response
    return response_handler.create_success_response_v1(
        response_data=images_data, 
        http_status_code=200
    )    