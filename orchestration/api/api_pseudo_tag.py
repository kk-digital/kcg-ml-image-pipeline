from datetime import datetime
from fastapi import APIRouter, Request, HTTPException, Query
from typing import List, Dict
from orchestration.api.mongo_schemas import TagDefinition, ImageTag, TagCategory, NewTagRequest, NewTagCategory
from typing import Union
from .api_utils import PrettyJSONResponse, validate_date_format, ApiResponseHandler, ErrorCode, StandardSuccessResponse, WasPresentResponse, TagsListResponse, VectorIndexUpdateRequest, TagsCategoryListResponse, TagResponse
import traceback
from bson import ObjectId


router = APIRouter()


@router.post("/pseudotags", 
             status_code=201,
             tags=["pseudo_tags"],
             description="Adds a new tag",
             response_model=StandardSuccessResponse[TagDefinition],
             responses=ApiResponseHandler.listErrors([400, 422, 500]))
def add_new_pseudo_tag_definition(request: Request, tag_data: NewTagRequest):
    response_handler = ApiResponseHandler(request)

    try:
        # Check for existing tag_category_id
        if tag_data.tag_category_id is not None:
            existing_category = request.app.pseudo_tag_categories_collection.find_one(
                {"tag_category_id": tag_data.tag_category_id}
            )
            if not existing_category:
                return response_handler.create_error_response(
                    ErrorCode.INVALID_PARAMS,
                    "Tag category not found",
                    400
                )

        # Check for existing tag_vector_index
        if tag_data.tag_vector_index is not None:
            existing_tag_with_index = request.app.pseudo_tag_definitions_collection.find_one(
                {"tag_vector_index": tag_data.tag_vector_index}
            )
            if existing_tag_with_index:
                return response_handler.create_error_response(
                    ErrorCode.INVALID_PARAMS,
                    "Tag vector index already in use.",
                    400
                )

        # Generate new tag_id
        last_entry = request.app.pseudo_tag_definitions_collection.find_one({}, sort=[("tag_id", -1)])
        new_tag_id = last_entry["tag_id"] + 1 if last_entry and "tag_id" in last_entry else 0

        # Check if the tag definition exists by tag_string
        existing_tag = request.app.pseudo_tag_definitions_collection.find_one({"tag_string": tag_data.tag_string})
        if existing_tag:
            return response_handler.create_error_response(
                ErrorCode.INVALID_PARAMS,
                "Tag definition already exists.",
                400
            )

        # Create the new tag object with only the specified fields
        new_tag = {
            "tag_id": new_tag_id,
            "tag_string": tag_data.tag_string,
            "tag_category_id": tag_data.tag_category_id,
            "tag_description": tag_data.tag_description,
            "tag_vector_index": tag_data.tag_vector_index if tag_data.tag_vector_index is not None else -1,
            "deprecated": tag_data.deprecated,
            "user_who_created": tag_data.user_who_created,
            "creation_time": datetime.utcnow().isoformat()
        }

        # Insert new tag definition into the collection
        inserted_id = request.app.pseudo_tag_definitions_collection.insert_one(new_tag).inserted_id
        new_tag = request.app.pseudo_tag_definitions_collection.find_one({"_id": inserted_id})

        new_tag = {k: str(v) if isinstance(v, ObjectId) else v for k, v in new_tag.items()}

        return response_handler.create_success_response(
            new_tag,
            http_status_code=201
        )

    except Exception as e:

        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal server error", 500)
    

@router.get("/tags/id-by-pseudotag-name", 
             status_code=200,
             tags=["pseudo_tags"],
             description="Get tag ID by tag name")
def get_pseudo_tag_id_by_name(request: Request, tag_string: str = Query(..., description="Tag name to fetch ID for")):
    api_handler = ApiResponseHandler(request)

    try:
        # Find the tag with the provided name
        tag = request.app.pseudo_tag_definitions_collection.find_one({"tag_string": tag_string})

        if tag is None:
            return api_handler.create_error_response(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string="Tag not found",
                http_status_code=404
            )

        tag_id = tag.get("tag_id")
        return api_handler.create_success_response(
            response_data={"tag_id": tag_id},
            http_status_code=200
        )

    except Exception as e:
        return api_handler.create_error_response(
            error_code=ErrorCode.OTHER_ERROR,
            error_string="Internal server error",
            http_status_code=500
        )


@router.patch("/pseudotags/{tag_id}", 
              tags=["pseudo_tags"],
              status_code=200,
              description="Update tag definitions",
              response_model=StandardSuccessResponse[TagDefinition], 
              responses=ApiResponseHandler.listErrors([400, 404, 422, 500]))
def update_pseudo_tag_definition(request: Request, tag_id: int, update_data: NewTagRequest):
    response_handler = ApiResponseHandler(request)

    query = {"tag_id": tag_id}
    existing_tag = request.app.pseudo_tag_definitions_collection.find_one(query)

    if existing_tag is None:
        return response_handler.create_error_response(
            ErrorCode.ELEMENT_NOT_FOUND, "Tag not found.", 404
        )

    # Check if the tag is deprecated
    if existing_tag.get("deprecated", False):
        return response_handler.create_error_response(
            ErrorCode.INVALID_PARAMS, "Cannot modify a deprecated tag.", 400
        )

    # Prepare update data
    update_fields = {k: v for k, v in update_data.dict().items() if v is not None}

    if not update_fields:
        return response_handler.create_error_response(
            ErrorCode.INVALID_PARAMS, "No fields to update.", 400
        )

    # Check if tag_vector_index is being updated and if it's already in use
    if 'tag_vector_index' in update_fields:
        index_query = {"tag_vector_index": update_fields['tag_vector_index']}
        existing_tag_with_index = request.app.pseudo_tag_definitions_collection.find_one(index_query)
        if existing_tag_with_index and existing_tag_with_index['tag_id'] != tag_id:
            return response_handler.create_error_response(
                ErrorCode.INVALID_PARAMS, "Tag vector index already in use.", 400
            )

    # Update the tag definition
    request.app.pseudo_tag_definitions_collection.update_one(query, {"$set": update_fields})

    # Retrieve the updated tag
    updated_tag = request.app.pseudo_tag_definitions_collection.find_one(query)

    # Serialize ObjectId to string
    updated_tag = {k: str(v) if isinstance(v, ObjectId) else v for k, v in updated_tag.items()}

    # Return the updated tag object
    return response_handler.create_success_response(updated_tag, 200)


@router.delete("/pseudotags/{tag_id}", 
               response_model=StandardSuccessResponse[WasPresentResponse], 
               description="remove tag with tag_id", 
               tags=["pseudo_tags"], 
               status_code=200,
               responses=ApiResponseHandler.listErrors([400, 422, 500]))
def remove_pseudo_tag(request: Request, tag_id: int):
    response_handler = ApiResponseHandler(request)

    # Check if the tag exists
    tag_query = {"tag_id": tag_id}
    tag = request.app.pseudo_tag_definitions_collection.find_one(tag_query)

    if tag is None:
        # Return standard response with wasPresent: false
        return response_handler.create_success_delete_response({"wasPresent": False})

    # Check if the tag is used in any images
    image_query = {"tags": tag_id}
    image_with_tag = request.app.pseudo_image_tags_collection.find_one(image_query)

    if image_with_tag is not None:
        # Since it's used in images, do not delete but notify the client
        return response_handler.create_error_response(
            ErrorCode.INVALID_PARAMS,
            "Cannot remove tag, it is already used in images.",
            400
        )

    # Remove the tag
    request.app.pseudo_tag_definitions_collection.delete_one(tag_query)

    # Return standard response with wasPresent: true
    return response_handler.create_success_delete_response({"wasPresent": True})


@router.get("/pseudotags", 
            response_model=StandardSuccessResponse[TagsListResponse],
            description="list tags",
            tags=["pseudo_tags"],
            status_code=200,
            responses=ApiResponseHandler.listErrors([500]))
def list_pseudo_tag_definitions(request: Request):
    response_handler = ApiResponseHandler(request)
    try:
        # Query all the tag definitions
        tags_cursor = request.app.pseudo_tag_definitions_collection.find({})

        # Convert each tag document to TagDefinition and then to a dictionary
        result = [TagDefinition(**tag).to_dict() for tag in tags_cursor]

        return response_handler.create_success_response({"tags": result}, http_status_code=200)

    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Exception Traceback:\n{traceback_str}")
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal server error", 500)

@router.put("/pseudotags/{tag_id}/vector-index", 
            tags=["pseudo_tags"], 
            status_code=200,
            description="Set vector index to tag definition",
            response_model=StandardSuccessResponse[VectorIndexUpdateRequest],
            responses=ApiResponseHandler.listErrors([400, 422, 500]))
def set_pseudo_tag_vector_index(request: Request, tag_id: int, update_data: VectorIndexUpdateRequest):
    response_handler = ApiResponseHandler(request)

    # Find the tag definition using the provided tag_id
    query = {"tag_id": tag_id}
    tag = request.app.pseudo_tag_definitions_collection.find_one(query)

    if not tag:
        return response_handler.create_error_response(
            ErrorCode.ELEMENT_NOT_FOUND, "Tag definition not found.", 404
        )

    # Check if any other tag has the same vector index
    existing_tag = request.app.pseudo_tag_definitions_collection.find_one({"tag_vector_index": update_data.vector_index})
    if existing_tag and existing_tag["tag_id"] != tag_id:
        return response_handler.create_error_response(
            ErrorCode.INVALID_PARAMS, "Another tag already has the same vector index.", 400
        )

    # Update the tag vector index
    update_query = {"$set": {"tag_vector_index": update_data.vector_index}}
    request.app.pseudo_tag_definitions_collection.update_one(query, update_query)

    # Optionally, retrieve updated tag data and include it in the response
    updated_tag = request.app.pseudo_tag_definitions_collection.find_one(query)
    return response_handler.create_success_response(
        {"tag_vector_index": updated_tag.get("tag_vector_index", None)}, 200
    )


@router.get("/pseudotags/{tag_id}/vector-index", 
            tags=["pseudo_tags"], 
            status_code=200,
            response_model=StandardSuccessResponse[VectorIndexUpdateRequest],
            responses=ApiResponseHandler.listErrors([400, 422, 500]))
def get_pseudo_tag_vector_index(request: Request, tag_id: int):
    response_handler = ApiResponseHandler(request)

    # Find the tag definition using the provided tag_id
    query = {"tag_id": tag_id}
    tag = request.app.pseudo_tag_definitions_collection.find_one(query)

    if not tag:
        return response_handler.create_error_response(
            ErrorCode.ELEMENT_NOT_FOUND, "Tag not found.", 404
        )

    vector_index = tag.get("tag_vector_index", None)
    return response_handler.create_success_response(
        {"tag_vector_index": vector_index}, 200
    )    

@router.post("/pseudotags/add_tag_to_image", response_model=ImageTag)
def add_pseudo_tag_to_image(request: Request, tag_id: int, file_hash: str, tag_type: int, user_who_created: str):
    date_now = datetime.now().isoformat()
    
    # Check if the tag exists by tag_id
    existing_tag = request.app.pseudo_tag_definitions_collection.find_one({"tag_id": tag_id})
    if not existing_tag:
        raise HTTPException(status_code=400, detail="Tag does not exist!")

    # Get the image from completed_jobs_collection using file_hash
    image = request.app.completed_jobs_collection.find_one({
        'task_output_file_dict.output_file_hash': file_hash
    })

    if not image:
        raise HTTPException(status_code=400, detail="No image found with the given hash")

    file_path = image.get("task_output_file_dict", {}).get("output_file_path", "")

    # Create association between image and tag
    image_tag_data = {
        "tag_id": tag_id,
        "file_path": file_path,  
        "image_hash": file_hash,
        "tag_type": tag_type,
        "user_who_created": user_who_created,
        "creation_time": date_now
    }

    request.app.pseudo_image_tags_collection.insert_one(image_tag_data)

    # Increment the tag count for the image's uuid
    job_uuid = image.get("job_uuid")
    if job_uuid:
        request.app.uuid_tag_count_collection.update_one(
            {"job_uuid": job_uuid},
            {"$inc": {"tag_count": 1}},
            upsert=True
        )

    return image_tag_data

@router.post("/pseudotags/add_tag_to_image-v1", response_model=ImageTag, response_class=PrettyJSONResponse)
def add_pseudo_tag_to_image(request: Request, tag_id: int, file_hash: str, tag_type: int, user_who_created: str):
    response_handler = ApiResponseHandler(request)
    try:
        date_now = datetime.now().isoformat()
    
        existing_tag = request.app.pseudo_tag_definitions_collection.find_one({"tag_id": tag_id})
        if not existing_tag:
            return response_handler.create_error_response(ErrorCode.ELEMENT_NOT_FOUND, "Tag does not exist!", 400)

        image = request.app.completed_jobs_collection.find_one({'task_output_file_dict.output_file_hash': file_hash})
        if not image:
            return response_handler.create_error_response(ErrorCode.ELEMENT_NOT_FOUND, "No image found with the given hash", 400)

        file_path = image.get("task_output_file_dict", {}).get("output_file_path", "")
        
        # Check if the tag is already associated with the image
        existing_image_tag = request.app.pseudo_image_tags_collection.find_one({"tag_id": tag_id, "image_hash": file_hash})
        if existing_image_tag:
            # Return an error response indicating that the tag has already been added to the image
            return response_handler.create_error_response(ErrorCode.INVALID_PARAMS, "This tag has already been added to the image", 400)

        # Add new tag to image
        image_tag_data = {
            "tag_id": tag_id,
            "file_path": file_path,  
            "image_hash": file_hash,
            "tag_type": tag_type,
            "user_who_created": user_who_created,
            "creation_time": date_now,
            "tag_count": 1  # Since this is a new tag for this image, set count to 1
        }
        request.app.pseudo_image_tags_collection.insert_one(image_tag_data)

        return response_handler.create_success_response({"tag_id": tag_id, "file_path": file_path, "image_hash": file_hash, "tag_type": tag_type, "tag_count": 1, "user_who_created": user_who_created, "creation_time": date_now}, http_status_code=200)

    except Exception as e:
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal server error", 500)


@router.delete("/pseudotags/remove_tag_from_image/{tag_id}", 
               status_code=200,
               tags=["pseudo_tags"], 
               description="Remove image tag",
               response_model=StandardSuccessResponse[WasPresentResponse],
               responses=ApiResponseHandler.listErrors([400, 422]))
def remove_image_pseudo_tag(
    request: Request,
    image_hash: str,  
    tag_id: int  # Now as a path parameter
):
    response_handler = ApiResponseHandler(request)

    # The query now checks for the specific tag_id within the array of tags
    query = {"image_hash": image_hash, "tag_id": tag_id}
    result = request.app.pseudo_image_tags_collection.delete_one(query)
    
    # If no document was found and deleted, use response_handler to raise an HTTPException
    if result.deleted_count == 0:
        return response_handler.create_error_response(
            ErrorCode.ELEMENT_NOT_FOUND, 
            "Tag or image hash not found",
            404
        )

    # Return standard success response with wasPresent: true using response_handler
    return response_handler.create_success_response({"wasPresent": True}, 200)


@router.get("/pseudotags/get_tag_list_for_image", response_model=List[TagResponse])
def get_pseudo_tag_list_for_image(request: Request, file_hash: str):
    # Fetch image tags based on image_hash
    image_tags_cursor = request.app.pseudo_image_tags_collection.find({"image_hash": file_hash})
    
    # Process the results
    tags_list = []
    for tag_data in image_tags_cursor:
        tag_definition = request.app.pseudo_tag_definitions_collection.find_one({"tag_id": tag_data["tag_id"]})
        
        if tag_definition:
            # Create a dictionary representing TagDefinition with tag_type
            tag_definition_dict = {
                "tag_id": tag_definition["tag_id"],
                "tag_string": tag_definition["tag_string"],
                "tag_type": tag_data.get("tag_type"),
                "tag_category_id": tag_definition.get("tag_category_id"),
                "tag_description": tag_definition["tag_description"],
                "tag_vector_index": tag_definition.get("tag_vector_index", -1),
                "deprecated": tag_definition.get("deprecated", False),
                "user_who_created": tag_definition["user_who_created"],
                "creation_time": tag_definition.get("creation_time", None)
            }

            tags_list.append(tag_definition_dict)
    
    return tags_list




@router.get("/pseudotags/{tag_id}/images", 
            tags=["pseudo_tags"], 
            status_code=200,
            description="Get images by tag_id",
            response_model=StandardSuccessResponse[ImageTag], 
            responses=ApiResponseHandler.listErrors([400, 422, 500]))
def get_pseudo_tagged_images(
    request: Request, 
    tag_id: int,
    start_date: str = None,
    end_date: str = None,
    order: str = Query("desc", description="Order in which the data should be returned. 'asc' for oldest first, 'desc' for newest first")
):
    response_handler = ApiResponseHandler(request)
    try:
        # Validate start_date and end_date
        if start_date:
            validated_start_date = validate_date_format(start_date)
            if validated_start_date is None:
                return response_handler.create_error_response(
                    ErrorCode.INVALID_PARAMS, "Invalid start_date format. Expected format: YYYY-MM-DDTHH:MM:SS", 400
                )
        if end_date:
            validated_end_date = validate_date_format(end_date)
            if validated_end_date is None:
                return response_handler.create_error_response(
                    ErrorCode.INVALID_PARAMS, "Invalid end_date format. Expected format: YYYY-MM-DDTHH:MM:SS", 400
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
        image_tags_cursor = request.app.pseudo_image_tags_collection.find(query).sort("creation_time", sort_order)

        # Process the results
        image_info_list = []
        for tag_data in image_tags_cursor:
            if "image_hash" in tag_data and "user_who_created" in tag_data and "file_path" in tag_data:
                image_tag = ImageTag(
                    tag_id=int(tag_data["tag_id"]),
                    file_path=tag_data["file_path"], 
                    image_hash=str(tag_data["image_hash"]),
                    tag_type=int(tag_data["tag_type"]),
                    user_who_created=tag_data["user_who_created"],
                    creation_time=tag_data.get("creation_time", None)
                )
                image_info_list.append(image_tag.model_dump())  # Convert to dictionary

        # Return the list of images in a standard success response
        return response_handler.create_success_response({"images": image_info_list}, 200)

    except Exception as e:
        # Log the exception details here, if necessary
        return response_handler.create_error_response(
            ErrorCode.OTHER_ERROR, "Internal Server Error", 500
        )


@router.get("/pseudotags/images", 
            tags=["pseudo_tags"], 
            status_code=200,
            description="Get all tagged images",
            response_model=StandardSuccessResponse[ImageTag], 
            responses=ApiResponseHandler.listErrors([400, 422, 500]))
def get_all_pseudo_tagged_images(request: Request):
    response_handler = ApiResponseHandler(request)

    try:
        # Execute the query to get all tagged images
        image_tags_cursor = request.app.pseudo_image_tags_collection.find({})

        # Process the results
        image_info_list = []
        for tag_data in image_tags_cursor:
            if "image_hash" in tag_data and "user_who_created" in tag_data and "file_path" in tag_data:
                image_tag = ImageTag(
                    tag_id=int(tag_data["tag_id"]),
                    file_path=tag_data["file_path"], 
                    image_hash=str(tag_data["image_hash"]),
                    tag_type=int(tag_data["tag_type"]),
                    user_who_created=tag_data["user_who_created"],
                    creation_time=tag_data.get("creation_time", None)
                )
                image_info_list.append(image_tag.model_dump())  # Convert to dictionary

        # Return the list of images in a standard success response
        return response_handler.create_success_response({"images": image_info_list}, 200)

    except Exception as e:
        # Log the exception details here, if necessary
        return response_handler.create_error_response(
            ErrorCode.OTHER_ERROR, str(e), 500
        )


@router.post("/pseudotag-categories",
             status_code=201, 
             tags=["tag-categories"], 
             description="Add Tag Category",
             response_model=StandardSuccessResponse[TagCategory],
             responses=ApiResponseHandler.listErrors([422, 500]))
def add_pseudo_tag_category(request: Request, tag_category_data: NewTagCategory):
    response_handler = ApiResponseHandler(request)
    try:
        # Assign new tag_category_id
        last_entry = request.app.pseudo_tag_categories_collection.find_one({}, sort=[("tag_category_id", -1)])
        new_tag_category_id = last_entry["tag_category_id"] + 1 if last_entry else 0

        # Prepare tag category document
        tag_category_document = tag_category_data.dict()
        tag_category_document["tag_category_id"] = new_tag_category_id
        tag_category_document["creation_time"] = datetime.utcnow().isoformat()

        # Insert new tag category
        inserted_id = request.app.pseudo_tag_categories_collection.insert_one(tag_category_document).inserted_id

        # Retrieve and serialize the new tag category object
        new_tag_category = request.app.pseudo_tag_categories_collection.find_one({"_id": inserted_id})
        serialized_tag_category = {k: str(v) if isinstance(v, ObjectId) else v for k, v in new_tag_category.items()}

        # Adjust order of the keys
        ordered_response = {
            "_id": serialized_tag_category.pop("_id"),
            "tag_category_id": serialized_tag_category.pop("tag_category_id"),
            **serialized_tag_category
        }

        # Return the ordered tag category in a standard success response
        return response_handler.create_success_response(ordered_response, http_status_code=201)

    except Exception as e:
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal server error", 500)



@router.patch("/pseudotag-categories/{tag_category_id}", 
              tags=["tag-categories"],
              status_code=200,
              description="Update tag category",
              response_model=StandardSuccessResponse[TagCategory],
              responses=ApiResponseHandler.listErrors([400, 404, 422, 500]))
def update_pseudo_tag_category(
    request: Request, 
    tag_category_id: int,
    update_data: NewTagCategory
):
    response_handler = ApiResponseHandler(request)

    query = {"tag_category_id": tag_category_id}
    existing_category = request.app.pseudo_tag_categories_collection.find_one(query)

    if existing_category is None:
        return response_handler.create_error_response(
            ErrorCode.ELEMENT_NOT_FOUND, "Tag category not found.", 404
        )

    update_fields = {k: v for k, v in update_data.dict(exclude_unset=True).items() if v is not None}

    if not update_fields:
        return response_handler.create_error_response(
            ErrorCode.INVALID_PARAMS, "No fields to update.", 400
        )

    request.app.pseudo_tag_categories_collection.update_one(query, {"$set": update_fields})

    updated_category = request.app.pseudo_tag_categories_collection.find_one(query)
    updated_category = {k: str(v) if isinstance(v, ObjectId) else v for k, v in updated_category.items()}

    # Adjust order of the keys
    ordered_response = {
        "_id": updated_category.pop("_id"),
        "tag_category_id": updated_category.pop("tag_category_id"),
        **updated_category
    }

    return response_handler.create_success_response(ordered_response, 200)


@router.delete("/pseudotag-categories/{tag_category_id}", 
               tags=["tag-categories"], 
               description="Remove tag category with tag_category_id", 
               status_code=200,
               response_model=StandardSuccessResponse[WasPresentResponse],
               responses=ApiResponseHandler.listErrors([400, 422, 500]))
def delete_pseudo_tag_category(request: Request, tag_category_id: int):
    response_handler = ApiResponseHandler(request)

    # Check if the tag category exists
    category_query = {"tag_category_id": tag_category_id}
    category = request.app.pseudo_tag_categories_collection.find_one(category_query)

    if category is None:
        # Return standard response with wasPresent: false
        return response_handler.create_success_delete_response({"wasPresent": False})

    # Check if the tag category is used in any tags
    tag_query = {"tag_category_id": tag_category_id}
    tag_with_category = request.app.pseudo_tag_definitions_collection.find_one(tag_query)

    if tag_with_category is not None:
        # Since it's used in tags, do not delete but notify the client
        return response_handler.create_error_response(
            ErrorCode.INVALID_PARAMS,
            "Cannot remove tag category, it is already used in tags.",
            400
        )

    # Remove the tag category
    request.app.pseudo_tag_categories_collection.delete_one(category_query)

    # Return standard response with wasPresent: true
    return response_handler.create_success_delete_response({"wasPresent": True})


@router.get("/pseudotag-categories", 
            tags=["tag-categories"], 
            description="List tag categories",
            status_code=200,
            response_model=StandardSuccessResponse[TagsCategoryListResponse],
            responses=ApiResponseHandler.listErrors([500]))
def list_pseudo_tag_categories(request: Request):
    response_handler = ApiResponseHandler(request)
    try:
        # Query all the tag categories
        categories_cursor = request.app.pseudo_tag_categories_collection.find({})

        # Convert each tag category document to a dictionary
        result = [{k: str(v) if isinstance(v, ObjectId) else v for k, v in category.items()} for category in categories_cursor]

        return response_handler.create_success_response({"categories": result}, http_status_code=200)

    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Exception Traceback:\n{traceback_str}")
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal server error", 500)
