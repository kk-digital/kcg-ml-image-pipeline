from datetime import datetime
from fastapi import APIRouter, Request, HTTPException, Query
from typing import List, Dict
from orchestration.api.mongo_schemas import TagDefinition, ImageTag, TagCategory, NewTagRequest, NewTagCategory
from typing import Union
from .api_utils import PrettyJSONResponse, validate_date_format, ApiResponseHandler, ErrorCode, StandardSuccessResponse, WasPresentResponse, TagsListResponse, VectorIndexUpdateRequest, TagsCategoryListResponse, TagResponse, TagCountResponse
import traceback
from bson import ObjectId


router = APIRouter()


@router.post("/tags/add_new_tag_definition")
def add_new_tag_definition(request: Request, tag_data: TagDefinition):

    # Find the maximum tag_id in the collection
    last_entry = request.app.tag_definitions_collection.find_one({}, sort=[("tag_id", -1)])
    new_tag_id = last_entry["tag_id"] + 1 if last_entry and "tag_id" in last_entry else 0

    # Check if the tag definition exists
    query = {"tag_string": tag_data.tag_string}
    existing_tag = request.app.tag_definitions_collection.find_one(query)

    if existing_tag is not None:
        raise HTTPException(status_code=400, detail="Tag definition already exists.")

    # Add new tag definition
    tag_data.tag_id = new_tag_id
    tag_data.creation_time = datetime.utcnow().isoformat()
    request.app.tag_definitions_collection.insert_one(tag_data.to_dict())
    return {"status": "success", "message": "Tag definition added successfully.", "tag_id": new_tag_id}


@router.post("/tags/new_tag_definition")
def add_new_tag_definition(request: Request, tag_data: TagDefinition):
    response_handler = ApiResponseHandler(request)

    try:
        # Check if the provided tag_category_id exists in the tag_categories_collection
        if tag_data.tag_category_id is not None:
            existing_category = request.app.tag_categories_collection.find_one(
                {"tag_category_id": tag_data.tag_category_id}
            )
            if not existing_category:
                return response_handler.create_error_response(
                    ErrorCode.ELEMENT_NOT_FOUND,
                    "Tag category not found",
                    400
                )

        # Find the maximum tag_id in the collection
        last_entry = request.app.tag_definitions_collection.find_one({}, sort=[("tag_id", -1)])
        new_tag_id = last_entry["tag_id"] + 1 if last_entry and "tag_id" in last_entry else 0

        # Check if the tag definition exists by tag_string
        existing_tag = request.app.tag_definitions_collection.find_one({"tag_string": tag_data.tag_string})
        if existing_tag:
            return response_handler.create_error_response(
                ErrorCode.INVALID_PARAMS,
                "Tag definition already exists.",
                400
            )

        # Assign new tag_id and creation time
        tag_data.tag_id = new_tag_id
        tag_data.creation_time = datetime.utcnow().isoformat()

        # Insert new tag definition and retrieve it
        inserted_id = request.app.tag_definitions_collection.insert_one(tag_data.to_dict()).inserted_id
        new_tag = request.app.tag_definitions_collection.find_one({"_id": inserted_id})
        
        # Prepare the response data with the new tag details
        new_tag_data = {
            "tag_id": new_tag["tag_id"],
            "tag_string": new_tag["tag_string"],
            "tag_category_id": new_tag.get("tag_category_id"),
            "tag_description": new_tag["tag_description"],
            "tag_vector_index": new_tag.get("tag_vector_index", -1),
            "deprecated": new_tag.get("deprecated", False),
            "user_who_created": new_tag["user_who_created"],
            "creation_time": new_tag["creation_time"]
        }

        return response_handler.create_success_response(
            new_tag_data,
            http_status_code=201
        )

    except Exception as e:
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal server error", 500)

@router.post("v1/tags", 
             status_code=201,
             tags=["tags"],
             description="Adds a new tag",
             response_model=StandardSuccessResponse[TagDefinition],
             responses=ApiResponseHandler.listErrors([400, 422, 500]))
@router.post("/tags", 
             status_code=201,
             tags=["deprecated"],
             description="Adds a new tag, DEPRECATED: the name was changed to v1/tags, no other changes were introduced",
             response_model=StandardSuccessResponse[TagDefinition],
             responses=ApiResponseHandler.listErrors([400, 422, 500]))
def add_new_tag_definition(request: Request, tag_data: NewTagRequest):
    response_handler = ApiResponseHandler(request)

    try:
        # Check for existing tag_category_id
        if tag_data.tag_category_id is not None:
            existing_category = request.app.tag_categories_collection.find_one(
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
            existing_tag_with_index = request.app.tag_definitions_collection.find_one(
                {"tag_vector_index": tag_data.tag_vector_index}
            )
            if existing_tag_with_index:
                return response_handler.create_error_response(
                    ErrorCode.INVALID_PARAMS,
                    "Tag vector index already in use.",
                    400
                )

        # Generate new tag_id
        last_entry = request.app.tag_definitions_collection.find_one({}, sort=[("tag_id", -1)])
        new_tag_id = last_entry["tag_id"] + 1 if last_entry and "tag_id" in last_entry else 0

        # Check if the tag definition exists by tag_string
        existing_tag = request.app.tag_definitions_collection.find_one({"tag_string": tag_data.tag_string})
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
        inserted_id = request.app.tag_definitions_collection.insert_one(new_tag).inserted_id
        new_tag = request.app.tag_definitions_collection.find_one({"_id": inserted_id})

        new_tag = {k: str(v) if isinstance(v, ObjectId) else v for k, v in new_tag.items()}

        return response_handler.create_success_response(
            new_tag,
            http_status_code=201
        )

    except Exception as e:

        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal server error", 500)

      

@router.get("/tags/id-by-tag-name", 
             status_code=200,
             tags=["tags"],
             description="Get tag ID by tag name")
def get_tag_id_by_name(request: Request, tag_string: str = Query(..., description="Tag name to fetch ID for")):
    api_handler = ApiResponseHandler(request)

    try:
        # Find the tag with the provided name
        tag = request.app.tag_definitions_collection.find_one({"tag_string": tag_string})

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

@router.put("/tags/update_tag_definition")
def update_tag_definition(request: Request, tag_id: int, update_data: TagDefinition):
    query = {"tag_id": tag_id}
    existing_tag = request.app.tag_definitions_collection.find_one(query)

    if existing_tag is None:
        raise HTTPException(status_code=404, detail="Tag not found.")

    # Prepare update data, excluding 'tag_id' and 'creation_time'
    update_fields = {k: v for k, v in update_data.dict(exclude={'tag_id', 'creation_time'}).items() if v is not None}

    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields to update.")

    # Optionally update
    update_fields["creation_time"] = datetime.utcnow().isoformat()

    # Update the tag definition
    request.app.tag_definitions_collection.update_one(query, {"$set": update_fields})
    return {"status": "success", "message": "Tag definition updated successfully.", "tag_id": tag_id}

@router.patch("/tags/{tag_id}/deprecated", 
              tags=["tags"],
              status_code=200,
              description="Toggle the 'deprecated' status of a tag definition. DEPRECATED: Adjust as needed for naming conventions.",
              response_model=StandardSuccessResponse[TagDefinition],  # Adjust if needed
              responses=ApiResponseHandler.listErrors([400, 404, 422, 500]))
def toggle_tag_deprecated_status(request: Request, tag_id: int):
    response_handler = ApiResponseHandler(request)

    query = {"tag_id": tag_id}
    existing_tag = request.app.tag_definitions_collection.find_one(query)

    if existing_tag is None:
        return response_handler.create_error_response(
            ErrorCode.ELEMENT_NOT_FOUND, "Tag not found.", 404
        )

    # the 'deprecated' status
    new_deprecated_status = not existing_tag.get("deprecated", False)

    # Update the tag definition with the new 'deprecated' status
    request.app.tag_definitions_collection.update_one(query, {"$set": {"deprecated": new_deprecated_status}})

    # Retrieve the updated tag to confirm the change
    updated_tag = request.app.tag_definitions_collection.find_one(query)

    # Serialize ObjectId to string if necessary
    updated_tag = {k: str(v) if isinstance(v, ObjectId) else v for k, v in updated_tag.items()}

    # Return the updated tag object
    return response_handler.create_success_response(updated_tag, 200)

@router.patch("v1/tags/{tag_id}", 
              tags=["tags"],
              status_code=200,
              description="Update tag definitions",
              response_model=StandardSuccessResponse[TagDefinition], 
              responses=ApiResponseHandler.listErrors([400, 404, 422, 500]))
@router.patch("/tags/{tag_id}", 
              tags=["deprecated"],
              status_code=200,
              description="Update tag definitions, DEPRECATED: the name was changed to v1/tags/{tag_id}, no other changes were introduced",
              response_model=StandardSuccessResponse[TagDefinition], 
              responses=ApiResponseHandler.listErrors([400, 404, 422, 500]))
def update_tag_definition(request: Request, tag_id: int, update_data: NewTagRequest):
    response_handler = ApiResponseHandler(request)

    query = {"tag_id": tag_id}
    existing_tag = request.app.tag_definitions_collection.find_one(query)

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
        existing_tag_with_index = request.app.tag_definitions_collection.find_one(index_query)
        if existing_tag_with_index and existing_tag_with_index['tag_id'] != tag_id:
            return response_handler.create_error_response(
                ErrorCode.INVALID_PARAMS, "Tag vector index already in use.", 400
            )

    # Update the tag definition
    request.app.tag_definitions_collection.update_one(query, {"$set": update_fields})

    # Retrieve the updated tag
    updated_tag = request.app.tag_definitions_collection.find_one(query)

    # Serialize ObjectId to string
    updated_tag = {k: str(v) if isinstance(v, ObjectId) else v for k, v in updated_tag.items()}

    # Return the updated tag object
    return response_handler.create_success_response(updated_tag, 200)




@router.delete("/tags/remove_tag")
def remove_test_tag(request: Request, tag_id: int):
    # Check if the tag exists
    tag_query = {"tag_id": tag_id}
    tag = request.app.tag_definitions_collection.find_one(tag_query)

    if tag is None:
        raise HTTPException(status_code=404, detail="Tag not found.")

    # Check if the tag is used in any images
    image_query = {"tags": tag_id}
    image_with_tag = request.app.image_tags_collection.find_one(image_query)

    if image_with_tag is not None:
        raise HTTPException(status_code=400, detail="Cannot remove tag, it is already used in images.")

    # Remove the tag
    request.app.tag_definitions_collection.delete_one(tag_query)
    return {"status": "success", "message": "Test tag removed successfully."}


@router.delete("/tags/{tag_id}", 
               response_model=StandardSuccessResponse[WasPresentResponse], 
               description="remove tag with tag_id", 
               tags=["tags"], 
               status_code=200,
               responses=ApiResponseHandler.listErrors([400, 422, 500]))
def remove_tag(request: Request, tag_id: int):
    response_handler = ApiResponseHandler(request)

    # Check if the tag exists
    tag_query = {"tag_id": tag_id}
    tag = request.app.tag_definitions_collection.find_one(tag_query)

    if tag is None:
        # Return standard response with wasPresent: false
        return response_handler.create_success_delete_response({"wasPresent": False})

    # Check if the tag is used in any images
    image_query = {"tags": tag_id}
    image_with_tag = request.app.image_tags_collection.find_one(image_query)

    if image_with_tag is not None:
        # Since it's used in images, do not delete but notify the client
        return response_handler.create_error_response(
            ErrorCode.INVALID_PARAMS,
            "Cannot remove tag, it is already used in images.",
            400
        )

    # Remove the tag
    request.app.tag_definitions_collection.delete_one(tag_query)

    # Return standard response with wasPresent: true
    return response_handler.create_success_delete_response({"wasPresent": True})


@router.get("/tags/list_tag_definition", response_class=PrettyJSONResponse)
def list_tag_definitions(request: Request):
    # Query all the tag definitions
    tags_cursor = request.app.tag_definitions_collection.find({})
    result = []

    for tag in tags_cursor:
        tag_data = {
            "tag_id": tag["tag_id"],
            "tag_string": tag["tag_string"],
            "tag_category": tag["tag_category"],
            "tag_vector_index": tag.get("tag_vector_index", -1),  # Use default value if tag_vector_index is absent
            "tag_description": tag["tag_description"],
            "user_who_created": tag["user_who_created"],
            "creation_time": tag["creation_time"]
        }
        result.append(tag_data)

    return result

@router.get("/tags/tag_definition", response_class=PrettyJSONResponse)
def list_tag_definitions(request: Request):
    response_handler = ApiResponseHandler(request)
    try:
        # Query all the tag definitions
        tags_cursor = request.app.tag_definitions_collection.find({})
        result = []

        for tag in tags_cursor:
            tag_data = {
                "tag_id": tag["tag_id"],
                "tag_string": tag["tag_string"],
                "tag_category_id": tag["tag_category_id"],
                "tag_vector_index": tag.get("tag_vector_index", -1),  # Use default value if tag_vector_index is absent
                "tag_description": tag["tag_description"],
                "deprecated": tag["deprecated"],  # Corrected typo from 'depracated' to 'deprecated'
                "user_who_created": tag["user_who_created"],
                "creation_time": tag["creation_time"]
            }
            result.append(tag_data)

        return response_handler.create_success_response(result, http_status_code=200)

    except Exception as e:
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal server error", 500)


@router.get("/tags", 
            response_model=StandardSuccessResponse[TagsListResponse],
            description="list tags",
            tags=["tags"],
            status_code=200,
            responses=ApiResponseHandler.listErrors([500]))
def list_tag_definitions(request: Request):
    response_handler = ApiResponseHandler(request)
    try:
        # Query all the tag definitions
        tags_cursor = request.app.tag_definitions_collection.find({})

        # Convert each tag document to TagDefinition and then to a dictionary
        result = [TagDefinition(**tag).to_dict() for tag in tags_cursor]

        return response_handler.create_success_response({"tags": result}, http_status_code=200)

    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Exception Traceback:\n{traceback_str}")
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal server error", 500)

@router.put("/tags/set_tag_vector_index")
def set_tag_vector_index(request: Request, tag_id: int, vector_index: int):
    # Find the tag definition using the provided tag_id
    query = {"tag_id": tag_id}
    tag = request.app.tag_definitions_collection.find_one(query)

    if not tag:
        return {"status": "fail", "message": "Tag definition not found."}
       

    # Check if any other tag has the same vector index
    existing_tag = request.app.tag_definitions_collection.find_one({"tag_vector_index": vector_index})
    if existing_tag and existing_tag["tag_id"] != tag_id:
        raise HTTPException(status_code=400, detail="Another tag already has the same vector index.")

    # Update the tag vector index
    update_query = {"$set": {"tag_vector_index": vector_index}}
    request.app.tag_definitions_collection.update_one(query, update_query)

    return {"status": "success", "message": "Tag vector index updated successfully."}

@router.put("/tags/{tag_id}/vector-index", 
            tags=["tags"], 
            status_code=200,
            description="Set vector index to tag definition",
            response_model=StandardSuccessResponse[VectorIndexUpdateRequest],
            responses=ApiResponseHandler.listErrors([400, 422, 500]))
def set_tag_vector_index(request: Request, tag_id: int, update_data: VectorIndexUpdateRequest):
    response_handler = ApiResponseHandler(request)

    # Find the tag definition using the provided tag_id
    query = {"tag_id": tag_id}
    tag = request.app.tag_definitions_collection.find_one(query)

    if not tag:
        return response_handler.create_error_response(
            ErrorCode.ELEMENT_NOT_FOUND, "Tag definition not found.", 404
        )

    # Check if any other tag has the same vector index
    existing_tag = request.app.tag_definitions_collection.find_one({"tag_vector_index": update_data.vector_index})
    if existing_tag and existing_tag["tag_id"] != tag_id:
        return response_handler.create_error_response(
            ErrorCode.INVALID_PARAMS, "Another tag already has the same vector index.", 400
        )

    # Update the tag vector index
    update_query = {"$set": {"tag_vector_index": update_data.vector_index}}
    request.app.tag_definitions_collection.update_one(query, update_query)

    # Optionally, retrieve updated tag data and include it in the response
    updated_tag = request.app.tag_definitions_collection.find_one(query)
    return response_handler.create_success_response(
        {"tag_vector_index": updated_tag.get("tag_vector_index", None)}, 200
    )


@router.get("/tags/get_tag_vector_index")
def get_tag_vector_index(request: Request, tag_id: int):
    # Find the tag definition using the provided tag_id
    query = {"tag_id": tag_id}
    tag = request.app.tag_definitions_collection.find_one(query)

    if not tag:
        return {"tag_vector_index": None}


    vector_index = tag.get("tag_vector_index", -1)
    return {"tag_vector_index": vector_index}


@router.get("/tags/{tag_id}/vector-index", 
            tags=["tags"], 
            status_code=200,
            response_model=StandardSuccessResponse[VectorIndexUpdateRequest],
            responses=ApiResponseHandler.listErrors([400, 422, 500]))
def get_tag_vector_index(request: Request, tag_id: int):
    response_handler = ApiResponseHandler(request)

    # Find the tag definition using the provided tag_id
    query = {"tag_id": tag_id}
    tag = request.app.tag_definitions_collection.find_one(query)

    if not tag:
        return response_handler.create_error_response(
            ErrorCode.ELEMENT_NOT_FOUND, "Tag not found.", 404
        )

    vector_index = tag.get("tag_vector_index", None)
    return response_handler.create_success_response(
        {"tag_vector_index": vector_index}, 200
    )    

@router.post("/tags/add_tag_to_image", response_model=ImageTag)
def add_tag_to_image(request: Request, tag_id: int, file_hash: str, tag_type: int, user_who_created: str):
    date_now = datetime.now().isoformat()
    
    # Check if the tag exists by tag_id
    existing_tag = request.app.tag_definitions_collection.find_one({"tag_id": tag_id})
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

    request.app.image_tags_collection.insert_one(image_tag_data)

    # Increment the tag count for the image's uuid
    job_uuid = image.get("job_uuid")
    if job_uuid:
        request.app.uuid_tag_count_collection.update_one(
            {"job_uuid": job_uuid},
            {"$inc": {"tag_count": 1}},
            upsert=True
        )

    return image_tag_data

@router.post("/tags/add_tag_to_image-v1", response_model=ImageTag, response_class=PrettyJSONResponse)
def add_tag_to_image(request: Request, tag_id: int, file_hash: str, tag_type: int, user_who_created: str):
    response_handler = ApiResponseHandler(request)
    try:
        date_now = datetime.now().isoformat()
    
        existing_tag = request.app.tag_definitions_collection.find_one({"tag_id": tag_id})
        if not existing_tag:
            return response_handler.create_error_response(ErrorCode.ELEMENT_NOT_FOUND, "Tag does not exist!", 400)

        image = request.app.completed_jobs_collection.find_one({'task_output_file_dict.output_file_hash': file_hash})
        if not image:
            return response_handler.create_error_response(ErrorCode.ELEMENT_NOT_FOUND, "No image found with the given hash", 400)

        file_path = image.get("task_output_file_dict", {}).get("output_file_path", "")
        
        # Check if the tag is already associated with the image
        existing_image_tag = request.app.image_tags_collection.find_one({"tag_id": tag_id, "image_hash": file_hash})
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
        request.app.image_tags_collection.insert_one(image_tag_data)

        return response_handler.create_success_response({"tag_id": tag_id, "file_path": file_path, "image_hash": file_hash, "tag_type": tag_type, "tag_count": 1, "user_who_created": user_who_created, "creation_time": date_now}, http_status_code=200)

    except Exception as e:
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal server error", 500)



@router.delete("/tags/remove_tag_from_image")
def remove_image_tag(
    request: Request,
    image_hash: str,  
    tag_id: int,  
):
    # The query now checks for the specific tag_id within the array of tags
    query = {"image_hash": image_hash, "tag_id": tag_id}
    result = request.app.image_tags_collection.delete_one(query)
    
    # If no document was found and deleted, raise an HTTPException
    if result.deleted_count == 0:
        print("Tag or image hash not found!")
      
    return {"status": "success"}

@router.delete("/tags/remove_tag_from_image/{tag_id}", 
               status_code=200,
               tags=["tags"], 
               description="Remove image tag",
               response_model=StandardSuccessResponse[WasPresentResponse],
               responses=ApiResponseHandler.listErrors([400, 422]))
def remove_image_tag(
    request: Request,
    image_hash: str,  
    tag_id: int  # Now as a path parameter
):
    response_handler = ApiResponseHandler(request)

    # The query now checks for the specific tag_id within the array of tags
    query = {"image_hash": image_hash, "tag_id": tag_id}
    result = request.app.image_tags_collection.delete_one(query)
    
    # If no document was found and deleted, use response_handler to raise an HTTPException
    if result.deleted_count == 0:
        return response_handler.create_error_response(
            ErrorCode.ELEMENT_NOT_FOUND, 
            "Tag or image hash not found",
            404
        )

    # Return standard success response with wasPresent: true using response_handler
    return response_handler.create_success_response({"wasPresent": True}, 200)

@router.delete("/tags/remove_all_tagged_images", 
               status_code=200,
               tags=["tags"], 
               description="Remove all tagged images",
               response_model=StandardSuccessResponse[Dict[str, int]],
               responses=ApiResponseHandler.listErrors([400, 422]))
def remove_all_tagged_images(request: Request):
    response_handler = ApiResponseHandler(request)

    try:
        # Query to match documents that have a 'tag_id' field
        query = {"tag_id": {"$exists": True}}
        result = request.app.image_tags_collection.delete_many(query)

        # If no documents were found and deleted, use response_handler to indicate that
        if result.deleted_count == 0:
            return response_handler.create_error_response(
                ErrorCode.ELEMENT_NOT_FOUND, 
                "No tagged images found",
                404
            )

        # Return standard success response with the count of deleted documents
        return response_handler.create_success_response(
            {"deleted_count": result.deleted_count}, 
            http_status_code=200
        )

    except Exception as e:
        return response_handler.create_error_response(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )

@router.delete("/tags/remove_tag_from_image-v1", response_class=PrettyJSONResponse)
def remove_image_tag(request: Request, image_hash: str, tag_id: int):
    response_handler = ApiResponseHandler(request)
    try:
        existing_image_tag = request.app.image_tags_collection.find_one({
            "tag_id": tag_id, 
            "image_hash": image_hash
        })

        if existing_image_tag:
            # If tag count is already zero, return ELEMENT_NOT_FOUND error
            if existing_image_tag["tag_count"] == 0:
                return response_handler.create_error_response(ErrorCode.ELEMENT_NOT_FOUND, "This image is not tagged with the given tag", 404)

            # Directly delete the tag association
            request.app.image_tags_collection.delete_one({"_id": existing_image_tag["_id"]})
            return response_handler.create_success_response({"wasPresent": True}, http_status_code=200)
        else:
            return response_handler.create_success_response({"wasPresent": False}, http_status_code=200)

    except Exception as e:
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal server error", 500)



@router.get("/tags/get_tag_list_for_image", response_model=List[TagResponse])
def get_tag_list_for_image(request: Request, file_hash: str):
    # Fetch image tags based on image_hash
    image_tags_cursor = request.app.image_tags_collection.find({"image_hash": file_hash})
    
    # Process the results
    tags_list = []
    for tag_data in image_tags_cursor:
        tag_definition = request.app.tag_definitions_collection.find_one({"tag_id": tag_data["tag_id"]})
        
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





@router.get("/tags/get_images_by_tag", response_model=List[ImageTag], response_class=PrettyJSONResponse)
def get_tagged_images(
    request: Request, 
    tag_id: int,
    start_date: str = None,
    end_date: str = None,
    order: str = Query("desc", description="Order in which the data should be returned. 'asc' for oldest first, 'desc' for newest first")
):
    # Build the query
    query = {"tag_id": tag_id}
    if start_date and end_date:
        query["creation_time"] = {"$gte": start_date, "$lte": end_date}
    elif start_date:
        query["creation_time"] = {"$gte": start_date}
    elif end_date:
        query["creation_time"] = {"$lte": end_date}

    # Decide the sort order based on the 'order' parameter
    sort_order = -1 if order == "desc" else 1

    # Execute the query
    image_tags_cursor = request.app.image_tags_collection.find(query).sort("creation_time", sort_order)

    # Process the results
    image_info_list = []
    for tag_data in image_tags_cursor:
        if "image_hash" in tag_data and "user_who_created" in tag_data and "file_path" in tag_data:
            image_info_list.append(ImageTag(
                tag_id=int(tag_data["tag_id"]),
                file_path=tag_data["file_path"], 
                image_hash=str(tag_data["image_hash"]),
                tag_type =int(tag_data["tag_type"]),
                user_who_created=tag_data["user_who_created"],
                creation_time=tag_data.get("creation_time", None)
            ))

    # Return the list of images
    return image_info_list


@router.get("/tags/{tag_id}/images", 
            tags=["tags"], 
            status_code=200,
            description="Get images by tag_id",
            response_model=StandardSuccessResponse[ImageTag], 
            responses=ApiResponseHandler.listErrors([400, 422, 500]))
def get_tagged_images(
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
        image_tags_cursor = request.app.image_tags_collection.find(query).sort("creation_time", sort_order)

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

@router.get("/tags/get_all_tagged_images", response_model=List[ImageTag], response_class=PrettyJSONResponse)
def get_all_tagged_images(request: Request):
    # Fetch all tagged image details
    image_tags_cursor = request.app.image_tags_collection.find({})

    image_info_list = [
        ImageTag(
            tag_id=int(tag_data["tag_id"]),
            file_path=tag_data["file_path"],  
            image_hash=str(tag_data["image_hash"]),
            tag_type=int(tag_data["tag_type"]),
            user_who_created=tag_data["user_who_created"],
            creation_time=tag_data.get("creation_time", None)
        ) 
        for tag_data in image_tags_cursor 
        if tag_data.get("image_hash") and tag_data.get("user_who_created") and tag_data.get("file_path")
    ]


    # If no tagged image details found, raise an exception
    if not image_info_list:
        return []

    return image_info_list


@router.get("/tags/images", 
            tags=["tags"], 
            status_code=200,
            description="Get all tagged images",
            response_model=StandardSuccessResponse[ImageTag], 
            responses=ApiResponseHandler.listErrors([400, 422, 500]))
def get_all_tagged_images(request: Request):
    response_handler = ApiResponseHandler(request)

    try:
        # Execute the query to get all tagged images
        image_tags_cursor = request.app.image_tags_collection.find({})

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


@router.post("/tags/add_tag_category", response_class=PrettyJSONResponse)
def add_tag_category(request: Request, tag_category_data: TagCategory):
    response_handler = ApiResponseHandler(request)
    try:
        # Assign new tag_category_id
        last_entry = request.app.tag_categories_collection.find_one({}, sort=[("tag_category_id", -1)])
        new_tag_category_id = last_entry["tag_category_id"] + 1 if last_entry else 0

        tag_category_data.tag_category_id = new_tag_category_id
        tag_category_data.creation_time = datetime.utcnow().isoformat()

        # Insert new tag category
        request.app.tag_categories_collection.insert_one(tag_category_data.to_dict())

        # Prepare the response data
        response_data = tag_category_data.to_dict()

        return response_handler.create_success_response(response_data, http_status_code=201)

    except Exception as e:
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal server error", 500)


@router.post("/tag-categories",
             status_code=201, 
             tags=["tag-categories"], 
             description="Add Tag Category",
             response_model=StandardSuccessResponse[TagCategory],
             responses=ApiResponseHandler.listErrors([422, 500]))
def add_tag_category(request: Request, tag_category_data: NewTagCategory):
    response_handler = ApiResponseHandler(request)
    try:
        # Assign new tag_category_id
        last_entry = request.app.tag_categories_collection.find_one({}, sort=[("tag_category_id", -1)])
        new_tag_category_id = last_entry["tag_category_id"] + 1 if last_entry else 0

        # Prepare tag category document
        tag_category_document = tag_category_data.dict()
        tag_category_document["tag_category_id"] = new_tag_category_id
        tag_category_document["creation_time"] = datetime.utcnow().isoformat()

        # Insert new tag category
        inserted_id = request.app.tag_categories_collection.insert_one(tag_category_document).inserted_id

        # Retrieve and serialize the new tag category object
        new_tag_category = request.app.tag_categories_collection.find_one({"_id": inserted_id})
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

@router.get("/tags/count/{tag_id}", 
            status_code=200,
            tags=["tags"], 
            description="Get count of images with a specific tag",
            response_model=TagCountResponse,
            responses=ApiResponseHandler.listErrors([400, 422]))
def get_image_count_by_tag(
    request: Request,
    tag_id: int
):
    response_handler = ApiResponseHandler(request)

    # Assuming each image document has an 'tags' array field
    query = {"tag_id": tag_id}
    count = request.app.image_tags_collection.count_documents(query)
    
    if count == 0:
        # If no images found with the tag, consider how you want to handle this. 
        # For example, you might still want to return a success response with a count of 0.
        return response_handler.create_success_response({"tag_id": tag_id, "count": 0}, 200)

    # Return standard success response with the count
    return response_handler.create_success_response({"tag_id": tag_id, "count": count}, 200)

@router.put("/tags/update_tag_category", response_class=PrettyJSONResponse)
def update_tag_category(request: Request, tag_category_id: int, tag_category_update: TagCategory):
    response_handler = ApiResponseHandler(request)
    try:
        # Check if the tag category exists
        existing_category = request.app.tag_categories_collection.find_one({"tag_category_id": tag_category_id})
        if not existing_category:
            return response_handler.create_error_response(ErrorCode.ELEMENT_NOT_FOUND, "Tag category not found", 404)

        # Check if the tag category is deprecated
        if existing_category.get("deprecated", False):
            return response_handler.create_error_response(ErrorCode.INVALID_OPERATION, "Cannot modify a deprecated tag category.", 400)

        # Prepare update data, excluding 'tag_category_id' and 'creation_time'
        update_fields = {k: v for k, v in tag_category_update.dict(exclude={'tag_category_id', 'creation_time'}).items() if v is not None}

        if not update_fields:
            return response_handler.create_error_response(ErrorCode.INVALID_PARAMS, "No fields to update", 400)

        # Update the tag category
        request.app.tag_categories_collection.update_one({"tag_category_id": tag_category_id}, {"$set": update_fields})

        # Fetch the updated tag category
        updated_category = request.app.tag_categories_collection.find_one({"tag_category_id": tag_category_id})
        if updated_category:
            updated_category_data = {
                "tag_category_id": updated_category["tag_category_id"],
                "tag_category_string": updated_category["tag_category_string"],
                "tag_category_description": updated_category["tag_category_description"],
                "deprecated": updated_category.get("deprecated", False),
                "user_who_created": updated_category["user_who_created"],
                "creation_time": updated_category["creation_time"]
            }
            return response_handler.create_success_response(updated_category_data, http_status_code=200)

    except Exception as e:
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal server error", 500)

@router.patch("/tag-categories/{tag_category_id}", 
              tags=["tag-categories"],
              status_code=200,
              description="Update tag category",
              response_model=StandardSuccessResponse[TagCategory],
              responses=ApiResponseHandler.listErrors([400, 404, 422, 500]))
def update_tag_category(
    request: Request, 
    tag_category_id: int,
    update_data: NewTagCategory
):
    response_handler = ApiResponseHandler(request)

    query = {"tag_category_id": tag_category_id}
    existing_category = request.app.tag_categories_collection.find_one(query)

    if existing_category is None:
        return response_handler.create_error_response(
            ErrorCode.ELEMENT_NOT_FOUND, "Tag category not found.", 404
        )

    update_fields = {k: v for k, v in update_data.dict(exclude_unset=True).items() if v is not None}

    if not update_fields:
        return response_handler.create_error_response(
            ErrorCode.INVALID_PARAMS, "No fields to update.", 400
        )

    request.app.tag_categories_collection.update_one(query, {"$set": update_fields})

    updated_category = request.app.tag_categories_collection.find_one(query)
    updated_category = {k: str(v) if isinstance(v, ObjectId) else v for k, v in updated_category.items()}

    # Adjust order of the keys
    ordered_response = {
        "_id": updated_category.pop("_id"),
        "tag_category_id": updated_category.pop("tag_category_id"),
        **updated_category
    }

    return response_handler.create_success_response(ordered_response, 200)


@router.delete("/tags/remove_tag_category", response_class=PrettyJSONResponse)
def remove_tag_category(request: Request, tag_category_id: int):
    response_handler = ApiResponseHandler(request)
    try:
        # Check if the tag category exists
        existing_category = request.app.tag_categories_collection.find_one({"tag_category_id": tag_category_id})
        was_present = False

        if existing_category:
            # Delete the tag category
            request.app.tag_categories_collection.delete_one({"tag_category_id": tag_category_id})
            was_present = True

        # Return success response with the 'wasPresent' flag
        return response_handler.create_success_response({"wasPresent": was_present}, http_status_code=200)

    except Exception as e:
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal server error", 500)


@router.delete("/tag-categories/{tag_category_id}", 
               tags=["tag-categories"], 
               description="Remove tag category with tag_category_id", 
               status_code=200,
               response_model=StandardSuccessResponse[WasPresentResponse],
               responses=ApiResponseHandler.listErrors([400, 422, 500]))
def delete_tag_category(request: Request, tag_category_id: int):
    response_handler = ApiResponseHandler(request)

    # Check if the tag category exists
    category_query = {"tag_category_id": tag_category_id}
    category = request.app.tag_categories_collection.find_one(category_query)

    if category is None:
        # Return standard response with wasPresent: false
        return response_handler.create_success_delete_response({"wasPresent": False})

    # Check if the tag category is used in any tags
    tag_query = {"tag_category_id": tag_category_id}
    tag_with_category = request.app.tag_definitions_collection.find_one(tag_query)

    if tag_with_category is not None:
        # Since it's used in tags, do not delete but notify the client
        return response_handler.create_error_response(
            ErrorCode.INVALID_PARAMS,
            "Cannot remove tag category, it is already used in tags.",
            400
        )

    # Remove the tag category
    request.app.tag_categories_collection.delete_one(category_query)

    # Return standard response with wasPresent: true
    return response_handler.create_success_delete_response({"wasPresent": True})


@router.get("/tags/list_tag_categories", response_class=PrettyJSONResponse)
def list_tag_categories(request: Request):
    response_handler = ApiResponseHandler(request)
    try:
        # Query all the tag categories
        categories_cursor = request.app.tag_categories_collection.find({})
        result = []

        for category in categories_cursor:
            category_data = {
                "tag_category_id": category["tag_category_id"],
                "tag_category_string": category["tag_category_string"],
                "tag_category_description": category["tag_category_description"],
                "deprecated": category.get("deprecated"),
                "user_who_created": category["user_who_created"],
                "creation_time": category["creation_time"]
            }
            result.append(category_data)

        return response_handler.create_success_response(result, http_status_code=200)

    except Exception as e:
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal server error", 500)

@router.get("/tag-categories", 
            tags=["tag-categories"], 
            description="List tag categories",
            status_code=200,
            response_model=StandardSuccessResponse[TagsCategoryListResponse],
            responses=ApiResponseHandler.listErrors([500]))
def list_tag_categories(request: Request):
    response_handler = ApiResponseHandler(request)
    try:
        # Query all the tag categories
        categories_cursor = request.app.tag_categories_collection.find({})

        # Convert each tag category document to a dictionary
        result = [{k: str(v) if isinstance(v, ObjectId) else v for k, v in category.items()} for category in categories_cursor]

        return response_handler.create_success_response({"categories": result}, http_status_code=200)

    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Exception Traceback:\n{traceback_str}")
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal server error", 500)
