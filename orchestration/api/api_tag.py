from datetime import datetime
from fastapi import APIRouter, Request, HTTPException, Query
from typing import List, Dict
from orchestration.api.mongo_schemas import TagDefinition, ImageTag, TagCategory
from typing import Union
from .api_utils import PrettyJSONResponse, ApiResponseHandler, ErrorCode


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


@router.get("/tags/get_tag_vector_index")
def get_tag_vector_index(request: Request, tag_id: int):
    # Find the tag definition using the provided tag_id
    query = {"tag_id": tag_id}
    tag = request.app.tag_definitions_collection.find_one(query)

    if not tag:
        return {"tag_vector_index": None}


    vector_index = tag.get("tag_vector_index", -1)
    return {"tag_vector_index": vector_index}

@router.post("/tags/add_tag_to_image", response_model=ImageTag)
def add_tag_to_image(request: Request, tag_id: int, file_hash: str, user_who_created: str):
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
def add_tag_to_image(request: Request, tag_id: int, file_hash: str, user_who_created: str):
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
            "user_who_created": user_who_created,
            "creation_time": date_now,
            "tag_count": 1  # Since this is a new tag for this image, set count to 1
        }
        request.app.image_tags_collection.insert_one(image_tag_data)

        return response_handler.create_success_response({"tag_id": tag_id, "file_path": file_path, "image_hash": file_hash, "tag_count": 1, "user_who_created": user_who_created, "creation_time": date_now}, http_status_code=200)

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



@router.get("/tags/get_tag_list_for_image", response_model=List[TagDefinition])
def get_tag_list_for_image(request: Request, file_hash: str):
    # Fetch image tags based on image_hash
    image_tags_cursor = request.app.image_tags_collection.find({"image_hash": file_hash})
    
    tag_ids = [tag_data["tag_id"] for tag_data in image_tags_cursor]
    
    # Fetch the actual TagDefinition using tag_ids
    tags_cursor = request.app.tag_definitions_collection.find({"tag_id": {"$in": tag_ids}})
    tags_list = [TagDefinition(**tag_data) for tag_data in tags_cursor]
    
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
                user_who_created=tag_data["user_who_created"],
                creation_time=tag_data.get("creation_time", None)
            ))

    # Return the list of images
    return image_info_list


@router.get("/tags/get_all_tagged_images", response_model=List[ImageTag], response_class=PrettyJSONResponse)
def get_all_tagged_images(request: Request):
    # Fetch all tagged image details
    image_tags_cursor = request.app.image_tags_collection.find({})

    image_info_list = [
        ImageTag(
            tag_id=int(tag_data["tag_id"]),
            file_path=tag_data["file_path"],  
            image_hash=str(tag_data["image_hash"]),
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



@router.put("/tags/update_tag_category", response_class=PrettyJSONResponse)
def update_tag_category(request: Request, tag_category_id: int, tag_category_update: TagCategory):
    response_handler = ApiResponseHandler(request)
    try:
        # Check if the tag category exists
        existing_category = request.app.tag_categories_collection.find_one({"tag_category_id": tag_category_id})
        if not existing_category:
            return response_handler.create_error_response(ErrorCode.ELEMENT_NOT_FOUND, "Tag category not found", 404)

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
