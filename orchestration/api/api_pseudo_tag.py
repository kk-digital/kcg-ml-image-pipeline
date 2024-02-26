from datetime import datetime
from fastapi import APIRouter, Request, HTTPException, Query
from typing import List, Dict
from orchestration.api.mongo_schema.pseudo_tag_schemas import PseudoTagDefinition, ImagePseudoTag, PseudoTagCategory, NewPseudoTagRequest, NewPseudoTagCategory
from typing import Union
from .api_utils import PrettyJSONResponse, validate_date_format, ApiResponseHandlerV1, ErrorCode, StandardSuccessResponseV1, WasPresentResponse, VectorIndexUpdateRequest, TagsCategoryListResponse, TagResponse, PseudoTagIdResponse
import traceback
from bson import ObjectId


router = APIRouter()


@router.post("/add-new-pseudotag", 
             status_code=201,
             tags=["pseudo_tags"],
             description="Adds a new tag",
             response_model=StandardSuccessResponseV1[PseudoTagDefinition],
             responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def add_new_pseudo_tag_definition(request: Request, tag_data: NewPseudoTagRequest):
    response_handler = ApiResponseHandlerV1(request, body_data=tag_data)

    try:
        # Check for existing tag_category_id
        if tag_data.pseudo_tag_category_id is not None:
            existing_category = request.app.pseudo_tag_categories_collection.find_one(
                {"pseudo_tag_category_id": tag_data.pseudo_tag_category_id}
            )
            if not existing_category:
                return response_handler.create_error_response_v1(
                    error_code=ErrorCode.INVALID_PARAMS,
                    error_string="Pseudo Tag category not found",
                    http_status_code=400
                )

        # Check for existing tag_vector_index
        if tag_data.pseudo_tag_vector_index is not None:
            existing_tag_with_index = request.app.pseudo_tag_definitions_collection.find_one(
                {"pseudo_tag_vector_index": tag_data.pseudo_tag_vector_index}
            )
            if existing_tag_with_index:
                return response_handler.create_error_response_v1(
                    error_code=ErrorCode.INVALID_PARAMS,
                    error_string="Pseudo Tag vector index already in use.",
                    http_status_code=400
                )

        # Generate new tag_id
        last_entry = request.app.pseudo_tag_definitions_collection.find_one({}, sort=[("pseudo_tag_id", -1)])
        new_tag_id = last_entry["pseudo_tag_id"] + 1 if last_entry and "pseudo_tag_id" in last_entry else 0

        # Check if the tag definition exists by tag_string
        existing_tag = request.app.pseudo_tag_definitions_collection.find_one({"pseudo_tag_string": tag_data.pseudo_tag_string})
        if existing_tag:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string="pseudo Tag definition already exists.",
                http_status_code=400
            )

        # Create the new tag object with only the specified fields
        new_tag = {
            "pseudo_tag_id": new_tag_id,
            "pseudo_tag_string": tag_data.tag_string,
            "pseudo_tag_category_id": tag_data.tag_category_id,
            "pseudo_tag_description": tag_data.tag_description,
            "pseudo_tag_vector_index": tag_data.tag_vector_index if tag_data.tag_vector_index is not None else -1,
            "deprecated": tag_data.deprecated,
            "user_who_created": tag_data.user_who_created,
            "creation_time": datetime.utcnow().isoformat()
        }

        # Insert new tag definition into the collection
        inserted_id = request.app.pseudo_tag_definitions_collection.insert_one(new_tag).inserted_id
        new_tag = request.app.pseudo_tag_definitions_collection.find_one({"_id": inserted_id})

        new_tag = {k: str(v) if isinstance(v, ObjectId) else v for k, v in new_tag.items()}

        return response_handler.create_success_response_v1(
            response_data=new_tag,
            http_status_code=201
        )

    except Exception as e:

        return response_handler.create_error_response_v1(error_code=ErrorCode.OTHER_ERROR, 
                                                         error_string="Internal server error", 
                                                         http_status_code=500)
    

@router.get("/tags/get-id-by-pseudotag-name", 
             status_code=200,
             tags=["pseudo_tags"],
             response_model=StandardSuccessResponseV1[PseudoTagIdResponse],
             responses=ApiResponseHandlerV1.listErrors([404, 422, 500]),
             description="Get tag ID by tag name")
def get_pseudo_tag_id_by_name(request: Request, pseudo_tag_string: str = Query(..., description="Tag name to fetch ID for")):
    api_handler = ApiResponseHandlerV1(request)

    try:
        # Find the tag with the provided name
        tag = request.app.pseudo_tag_definitions_collection.find_one({"pseudo_tag_string": pseudo_tag_string})

        if tag is None:
            return api_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string="Tag not found",
                http_status_code=404
            )

        pseudo_tag_id = tag.get("pseudo_tag_id")
        return api_handler.create_success_response_v1(
            response_data= pseudo_tag_id,
            http_status_code=200
        )

    except Exception as e:
        return api_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string="Internal server error",
            http_status_code=500
        )


@router.patch("/pseudotags/update-pseudo-tag-definition", 
              tags=["pseudo_tags"],
              status_code=200,
              description="Update pseudo tag definitions",
              response_model=StandardSuccessResponseV1[PseudoTagDefinition], 
              responses=ApiResponseHandlerV1.listErrors([400, 404, 422, 500]))
def update_pseudo_tag_definition(request: Request, pseudo_tag_id: int, update_data: NewPseudoTagRequest):
    response_handler = ApiResponseHandlerV1(request, body_data=update_data)

    query = {"pseudo_tag_id": pseudo_tag_id}
    existing_tag = request.app.pseudo_tag_definitions_collection.find_one(query)

    if existing_tag is None:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.ELEMENT_NOT_FOUND, 
            error_string="Tag not found.", 
            http_status_code=404
        )

    # Check if the tag is deprecated
    if existing_tag.get("deprecated", False):
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS, 
            error_string="Cannot modify a deprecated tag.", 
            http_status_code=400
        )

    # Prepare update data
    update_fields = {k: v for k, v in update_data.dict().items() if v is not None}

    if not update_fields:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS, 
            error_string="No fields to update.", 
            http_status_code=400
        )

    # Check if tag_vector_index is being updated and if it's already in use
    if 'pseudo_tag_vector_index' in update_fields:
        index_query = {"pseudo_tag_vector_index": update_fields['pseudo_tag_vector_index']}
        existing_tag_with_index = request.app.pseudo_tag_definitions_collection.find_one(index_query)
        if existing_tag_with_index and existing_tag_with_index['pseudo_tag_id'] != pseudo_tag_id:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS, 
                error_string="Tag vector index already in use.", 
                http_status_code=400
            )

    # Update the tag definition
    request.app.pseudo_tag_definitions_collection.update_one(query, {"$set": update_fields})

    # Retrieve the updated tag
    updated_tag = request.app.pseudo_tag_definitions_collection.find_one(query)

    # Serialize ObjectId to string
    updated_tag = {k: str(v) if isinstance(v, ObjectId) else v for k, v in updated_tag.items()}

    # Return the updated tag object
    return response_handler.create_success_response_v1(response_data=updated_tag, http_status_code=200)


@router.delete("/pseudotags/remove-pseudo-tag-definition/{tag_id}", 
               response_model=StandardSuccessResponseV1[WasPresentResponse], 
               description="remove pseudo tag with pseudo_tag_id", 
               tags=["pseudo_tags"], 
               status_code=200,
               responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def remove_pseudo_tag(request: Request, pseudo_tag_id: int):
    response_handler = ApiResponseHandlerV1(request)

    # Check if the tag exists
    tag_query = {"pseudo_tag_id": pseudo_tag_id}
    tag = request.app.pseudo_tag_definitions_collection.find_one(tag_query)

    if tag is None:
        # Return standard response with wasPresent: false
       return response_handler.create_success_delete_response_v1(error_code = ErrorCode.SUCCESS,
                                                           error_string= '',
                                                           response_data = {"wasPresent": False},
                                                           http_status_code=200,
                                                           )

    # Check if the tag is used in any images
    image_query = {"pseudo_tag_id": pseudo_tag_id}
    image_with_tag = request.app.pseudo_image_tags_collection.find_one(image_query)

    if image_with_tag is not None:
        # Since it's used in images, do not delete but notify the client
        return response_handler.create_error_response(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="Cannot remove tag, it is already used in images.",
            http_status_code=400
        )

    # Remove the tag
    request.app.pseudo_tag_definitions_collection.delete_one(tag_query)

    # Return standard response with wasPresent: true
    return response_handler.create_success_response_v1(
                                                       response_data = {"wasPresent": True},
                                                       http_status_code=201,
                                                       )


@router.get("/pseudotags/list-pseudo-tag-definitions", 
            response_model=StandardSuccessResponseV1[List[PseudoTagDefinition]],
            description="list pseudo tags",
            tags=["pseudo_tags"],
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([500]))
def list_pseudo_tag_definitions(request: Request):
    response_handler = ApiResponseHandlerV1(request)
    try:
        # Query all the tag definitions
        tags_cursor = request.app.pseudo_tag_definitions_collection.find({})

        # Convert each tag document to TagDefinition and then to a dictionary
        result = [PseudoTagDefinition(**pseudo_tag).to_dict() for pseudo_tag in tags_cursor]

        return response_handler.create_success_response_v1(response_data={"tags": result}, http_status_code=200)

    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Exception Traceback:\n{traceback_str}")
        return response_handler.create_error_response_v1(error_code=ErrorCode.OTHER_ERROR, error_string="Internal server error",http_status_code=500)

@router.put("/pseudotags/set-vector-index", 
            tags=["pseudo_tags"], 
            status_code=200,
            description="Set vector index to pseudo tag definition",
            response_model=StandardSuccessResponseV1[VectorIndexUpdateRequest],
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def set_pseudo_tag_vector_index(request: Request, pseudo_tag_id: int, update_data: VectorIndexUpdateRequest):
    response_handler = ApiResponseHandlerV1(request, body_data=update_data)

    # Find the tag definition using the provided tag_id
    query = {"pseudo_tag_id": pseudo_tag_id}
    tag = request.app.pseudo_tag_definitions_collection.find_one(query)

    if not tag:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.ELEMENT_NOT_FOUND, error_string="pseudo Tag definition not found.", http_status_code=404
        )

    # Check if any other tag has the same vector index
    existing_tag = request.app.pseudo_tag_definitions_collection.find_one({"pseudo_tag_vector_index": update_data.vector_index})
    if existing_tag and existing_tag["pseudo_tag_id"] != pseudo_tag_id:
        return response_handler.create_error_response(
            ErrorCode.INVALID_PARAMS, "Another pseudo tag already has the same vector index.", 400
        )

    # Update the tag vector index
    update_query = {"$set": {"pseudo_tag_vector_index": update_data.vector_index}}
    request.app.pseudo_tag_definitions_collection.update_one(query, update_query)

    # Optionally, retrieve updated tag data and include it in the response
    updated_tag = request.app.pseudo_tag_definitions_collection.find_one(query)
    return response_handler.create_success_response_v1(
        response_data={"pseudo_tag_vector_index": updated_tag.get("pseudo_tag_vector_index", None)}, http_status_code=200
    )


@router.get("/pseudotags/get-vector-index", 
            tags=["pseudo_tags"], 
            status_code=200,
            description="get vector index by pseudotag id",
            response_model=StandardSuccessResponseV1[VectorIndexUpdateRequest],
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def get_pseudo_tag_vector_index(request: Request, pseudo_tag_id: int):
    response_handler = ApiResponseHandlerV1(request)

    # Find the tag definition using the provided tag_id
    query = {"pseudo_tag_id": pseudo_tag_id}
    tag = request.app.pseudo_tag_definitions_collection.find_one(query)

    if not tag:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.ELEMENT_NOT_FOUND, error_string="Tag not found.", http_status_code=404
        )

    vector_index = tag.get("pseudo_tag_vector_index", None)
    return response_handler.create_success_response_v1(
        {"pseudo_tag_vector_index": vector_index}, 200
    )    

@router.post("/pseudotags/add-pseudo-tag-to-image", response_model=ImagePseudoTag, response_class=PrettyJSONResponse)
def add_pseudo_tag_to_image(request: Request, pseudo_tag_id: int, file_hash: str, pseudo_tag_type: int, user_who_created: str):
    response_handler = ApiResponseHandlerV1(request)
    try:
        date_now = datetime.now().isoformat()
    
        existing_tag = request.app.pseudo_tag_definitions_collection.find_one({"pseudo_tag_id": pseudo_tag_id})
        if not existing_tag:
            return response_handler.create_error_response_v1(error_code=ErrorCode.ELEMENT_NOT_FOUND, error_string="Tag does not exist!", http_status_code=400)

        image = request.app.completed_jobs_collection.find_one({'task_output_file_dict.output_file_hash': file_hash})
        if not image:
            return response_handler.create_error_response(error_code=ErrorCode.ELEMENT_NOT_FOUND, error_string="No image found with the given hash", http_status_code=400)

        file_path = image.get("task_output_file_dict", {}).get("output_file_path", "")
        
        # Check if the tag is already associated with the image
        existing_image_tag = request.app.pseudo_image_tags_collection.find_one({"pseudo_tag_id": pseudo_tag_id, "image_hash": file_hash})
        if existing_image_tag:
            # Return an error response indicating that the tag has already been added to the image
            return response_handler.create_error_response_v1(error_code=ErrorCode.INVALID_PARAMS, error_string="This tag has already been added to the image", http_status_code=400)

        # Add new tag to image
        image_pseudo_tag_data = {
            "pseudo_tag_id": pseudo_tag_id,
            "file_path": file_path,  
            "image_hash": file_hash,
            "pseudo_tag_type": pseudo_tag_type,
            "user_who_created": user_who_created,
            "creation_time": date_now,
            "tag_count": 1  # Since this is a new tag for this image, set count to 1
        }
        request.app.pseudo_image_tags_collection.insert_one(image_pseudo_tag_data)

        return response_handler.create_success_response_v1(response_data={"pseudo_tag_id": pseudo_tag_id, "file_path": file_path, "image_hash": file_hash, "pseudo_tag_type": pseudo_tag_type, "tag_count": 1, "user_who_created": user_who_created, "creation_time": date_now}, http_status_code=200)

    except Exception as e:
        return response_handler.create_error_response_v1(error_code=ErrorCode.OTHER_ERROR, error_string="Internal server error", http_status_code=500)


@router.delete("/tags/remove_pseudo_tag_from_image", status_code=200,
                tags=["pseudo_tags"], 
               description="Remove pseudo tag from image",
               response_model=StandardSuccessResponseV1[WasPresentResponse],
               responses=ApiResponseHandlerV1.listErrors([400, 422])) 
def remove_image_pseudo_tag(request: Request, image_hash: str, pseudo_tag_id: int):
    response_handler = ApiResponseHandlerV1(request)
    try:
        existing_image_tag = request.app.pseudo_image_tags_collection.find_one({
            "pseudo_tag_id": pseudo_tag_id, 
            "image_hash": image_hash
        })

        if existing_image_tag:
            # If tag count is already zero, return ELEMENT_NOT_FOUND error
            if existing_image_tag["tag_count"] == 0:
                return response_handler.create_error_response_v1(error_code=ErrorCode.ELEMENT_NOT_FOUND, error_string="This image is not tagged with the given tag", http_status_code=404)

            # Directly delete the tag association
            request.app.pseudo_image_tags_collection.delete_one({"_id": existing_image_tag["_id"]})
            return response_handler.create_success_response_v1(response_data={"wasPresent": True}, http_status_code=200)
        else:
            return response_handler.create_success_response_v1(response_data={"wasPresent": False}, http_status_code=200)

    except Exception as e:
        return response_handler.create_error_response_v1(error_code=ErrorCode.OTHER_ERROR, error_string="Internal server error", http_status_code=500)


@router.get("/pseudotags/get_pseudo_tag_list_for_image", 
            tags=["pseudo_tags"], 
            status_code=200,
            description="list pseudo tags for image",
            response_model=StandardSuccessResponseV1[list[PseudoTagDefinition]],
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def get_pseudo_tag_list_for_image(request: Request, file_hash: str):
    # Fetch image tags based on image_hash
    image_tags_cursor = request.app.pseudo_image_tags_collection.find({"image_hash": file_hash})
    
    # Process the results
    pseudo_tags_list = []
    for tag_data in image_tags_cursor:
        pseudo_tag_definition = request.app.pseudo_tag_definitions_collection.find_one({"pseudo_tag_id": tag_data["pseudo_tag_id"]})
        
        if pseudo_tag_definition:
            # Create a dictionary representing pseudoTagDefinition with tag_type
            pseudo_tag_definition_dict = {
                "pseudo_tag_id": pseudo_tag_definition["pseudo_tag_id"],
                "pseudo_tag_string": pseudo_tag_definition["pseudo_tag_string"],
                "pseudo_tag_type": tag_data.get("pseudo_tag_type"),
                "pseudo_tag_category_id": pseudo_tag_definition.get("pseudo_tag_category_id"),
                "pseudo_tag_description": pseudo_tag_definition["pseudo_tag_description"],
                "pseudo_tag_vector_index": pseudo_tag_definition.get("pseudo_tag_vector_index", -1),
                "deprecated": pseudo_tag_definition.get("deprecated", False),
                "user_who_created": pseudo_tag_definition["user_who_created"],
                "creation_time": pseudo_tag_definition.get("creation_time", None)
            }

            pseudo_tags_list.append(pseudo_tag_definition_dict)
    
    return pseudo_tags_list




@router.get("/pseudotags/get-images-by-tag", 
            tags=["pseudo_tags"], 
            status_code=200,
            description="Get images by pseudo_tag_id",
            response_model=StandardSuccessResponseV1[ImagePseudoTag], 
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def get_pseudo_tagged_images(
    request: Request, 
    pseudo_tag_id: int,
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
                    error_code=ErrorCode.INVALID_PARAMS, error_string="Invalid start_date format. Expected format: YYYY-MM-DDTHH:MM:SS", http_status_code=400
                )
        if end_date:
            validated_end_date = validate_date_format(end_date)
            if validated_end_date is None:
                return response_handler.create_error_response_v1(
                    error_code=ErrorCode.INVALID_PARAMS, error_string="Invalid end_date format. Expected format: YYYY-MM-DDTHH:MM:SS", http_status_code=400
                )

        # Build the query
        query = {"pseudo_tag_id": pseudo_tag_id}
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
                image_tag = ImagePseudoTag(
                    pseudo_tag_id=int(tag_data["pseudo_tag_id"]),
                    file_path=tag_data["file_path"], 
                    image_hash=str(tag_data["image_hash"]),
                    pseudo_tag_type=int(tag_data["pseudo_tag_type"]),
                    user_who_created=tag_data["user_who_created"],
                    creation_time=tag_data.get("creation_time", None)
                )
                image_info_list.append(image_tag.model_dump())  # Convert to dictionary

        # Return the list of images in a standard success response
        return response_handler.create_success_response_v1(response_data={"images": image_info_list}, http_status_code=200)

    except Exception as e:
        # Log the exception details here, if necessary
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, error_string="Internal Server Error", http_status_code=500
        )


@router.get("/pseudotags/get-all-tagged-images", 
            tags=["pseudo_tags"], 
            status_code=200,
            description="Get all tagged images",
            response_model=StandardSuccessResponseV1[ImagePseudoTag], 
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def get_all_pseudo_tagged_images(request: Request):
    response_handler = ApiResponseHandlerV1(request)

    try:
        # Execute the query to get all tagged images
        image_tags_cursor = request.app.pseudo_image_tags_collection.find({})

        # Process the results
        image_info_list = []
        for tag_data in image_tags_cursor:
            if "image_hash" in tag_data and "user_who_created" in tag_data and "file_path" in tag_data:
                image_tag = ImagePseudoTag(
                    pseudo_tag_id=int(tag_data["pseudo_tag_id"]),
                    file_path=tag_data["file_path"], 
                    image_hash=str(tag_data["image_hash"]),
                    pseudo_tag_type=int(tag_data["pseudo_tag_type"]),
                    user_who_created=tag_data["user_who_created"],
                    creation_time=tag_data.get("creation_time", None)
                )
                image_info_list.append(image_tag.model_dump())  # Convert to dictionary

        # Return the list of images in a standard success response
        return response_handler.create_success_response_v1(response_data={"images": image_info_list}, http_status_code=200)

    except Exception as e:
        # Log the exception details here, if necessary
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, error_string=str(e), http_status_code=500
        )


@router.post("/pseudotag-categories/add-pseudo-tag-category",
             status_code=201, 
             tags=["pseudotag-categories"], 
             description="Add pseudo Tag Category",
             response_model=StandardSuccessResponseV1[PseudoTagCategory],
             responses=ApiResponseHandlerV1.listErrors([422, 500]))
def add_pseudo_tag_category(request: Request, tag_category_data: NewPseudoTagCategory):
    response_handler = ApiResponseHandlerV1(request, body_data=tag_category_data)
    try:
        # Assign new tag_category_id
        last_entry = request.app.pseudo_tag_categories_collection.find_one({}, sort=[("pseudo_tag_category_id", -1)])
        new_pseudo_tag_category_id = last_entry["pseudo_tag_category_id"] + 1 if last_entry else 0

        # Prepare tag category document
        pseudo_tag_category_document = tag_category_data.dict()
        pseudo_tag_category_document["pseudo_tag_category_id"] = new_pseudo_tag_category_id
        pseudo_tag_category_document["creation_time"] = datetime.utcnow().isoformat()

        # Insert new tag category
        inserted_id = request.app.pseudo_tag_categories_collection.insert_one(pseudo_tag_category_document).inserted_id

        # Retrieve and serialize the new tag category object
        new_pseudo_tag_category = request.app.pseudo_tag_categories_collection.find_one({"_id": inserted_id})
        serialized_tag_category = {k: str(v) if isinstance(v, ObjectId) else v for k, v in new_pseudo_tag_category.items()}

        # Adjust order of the keys
        ordered_response = {
            "_id": serialized_tag_category.pop("_id"),
            "pseudo_tag_category_id": serialized_tag_category.pop("pseudo_tag_category_id"),
            **serialized_tag_category
        }

        # Return the ordered tag category in a standard success response
        return response_handler.create_success_response_v1(response_data=ordered_response, http_status_code=201)

    except Exception as e:
        return response_handler.create_error_response_v1(error_code=ErrorCode.OTHER_ERROR, error_string="Internal server error", http_status_code=500)



@router.patch("/pseudotag-categories/update-pseudo-tag-category", 
              tags=["pseudotag-categories"],
              status_code=200,
              description="Update pseudo tag category",
              response_model=StandardSuccessResponseV1[PseudoTagCategory],
              responses=ApiResponseHandlerV1.listErrors([400, 404, 422, 500]))
def update_pseudo_tag_category(
    request: Request, 
    pseudo_tag_category_id: int,
    update_data: NewPseudoTagCategory
):
    response_handler = ApiResponseHandlerV1(request, body_data=update_data)

    query = {"pseudo_tag_category_id": pseudo_tag_category_id}
    existing_category = request.app.pseudo_tag_categories_collection.find_one(query)

    if existing_category is None:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.ELEMENT_NOT_FOUND, error_string="Tag category not found.", http_status_code=404
        )

    update_fields = {k: v for k, v in update_data.dict(exclude_unset=True).items() if v is not None}

    if not update_fields:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS, error_string="No fields to update.", http_status_code=400
        )

    request.app.pseudo_tag_categories_collection.update_one(query, {"$set": update_fields})

    updated_category = request.app.pseudo_tag_categories_collection.find_one(query)
    updated_category = {k: str(v) if isinstance(v, ObjectId) else v for k, v in updated_category.items()}

    # Adjust order of the keys
    ordered_response = {
        "_id": updated_category.pop("_id"),
        "pseudo_tag_category_id": updated_category.pop("pseudo_tag_category_id"),
        **updated_category
    }

    return response_handler.create_success_response_v1(response_data=ordered_response, http_status_code=200)


@router.delete("/pseudotag-categories/remove-pseudo-tag-category/{pseudo_tag_category_id}", 
               tags=["pseudotag-categories"], 
               description="Remove pseudo tag category with tag_category_id", 
               status_code=200,
               response_model=StandardSuccessResponseV1[WasPresentResponse],
               responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def delete_pseudo_tag_category(request: Request, pseudo_tag_category_id: int):
    response_handler = ApiResponseHandlerV1(request)

    # Check if the tag category exists
    category_query = {"pseudo_tag_category_id": pseudo_tag_category_id}
    category = request.app.pseudo_tag_categories_collection.find_one(category_query)

    if category is None:
        # Return standard response with wasPresent: false
        return response_handler.create_success_delete_response_v1(error_code = ErrorCode.SUCCESS,
                                                           error_string= '',
                                                           response_data={"wasPresent": False},
                                                           http_status_code=200,
                                                           )

    # Check if the tag category is used in any tags
    tag_query = {"pseudo_tag_category_id": pseudo_tag_category_id}
    tag_with_category = request.app.pseudo_tag_definitions_collection.find_one(tag_query)

    if tag_with_category is not None:
        # Since it's used in tags, do not delete but notify the client
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string="Cannot remove tag category, it is already used in tags.",
            http_status_code=400
        )

    # Remove the tag category
    request.app.pseudo_tag_categories_collection.delete_one(category_query)

    # Return standard response with wasPresent: true
    return response_handler.create_success_delete_response_v1(error_code = ErrorCode.SUCCESS,
                                                           error_string= '',
                                                           response_data={"wasPresent": False},
                                                           http_status_code=200,
                                                           )
@router.get("/pseudotag-categories/list-pseudo-tag-categories", 
            tags=["pseudotag-categories"], 
            description="List pseudo tag categories",
            status_code=200,
            response_model=StandardSuccessResponseV1[List[PseudoTagCategory]],
            responses=ApiResponseHandlerV1.listErrors([500]))
def list_pseudo_tag_categories(request: Request):
    response_handler = ApiResponseHandlerV1(request)
    try:
        # Query all the tag categories
        categories_cursor = request.app.pseudo_tag_categories_collection.find({})

        # Convert each tag category document to a dictionary
        result = [{k: str(v) if isinstance(v, ObjectId) else v for k, v in category.items()} for category in categories_cursor]

        return response_handler.create_success_response_v1(response_data={"categories": result}, http_status_code=200)

    except Exception as e:
        traceback_str = traceback.format_exc()
        return response_handler.create_error_response_v1(error_code=ErrorCode.OTHER_ERROR, error_string="Internal server error", http_status_code=500)


@router.patch("/pseudotags/set-deprecated", 
              tags=["pseudo_tags"],
              status_code=200,
              description="Set the 'deprecated' status of a pseudotag definition to True",
              response_model=StandardSuccessResponseV1[PseudoTagDefinition],  
              responses=ApiResponseHandlerV1.listErrors([400, 404, 422, 500]))
def set_tag_deprecated(request: Request, pseudo_tag_id: int):
    response_handler = ApiResponseHandlerV1(request)

    query = {"pseudo_tag_id": pseudo_tag_id}
    existing_tag = request.app.pseudo_tag_definitions_collection.find_one(query)

    if existing_tag is None:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.ELEMENT_NOT_FOUND, error_string="Tag not found.", http_status_code=404
        )

    # Check if the tag is already deprecated
    if existing_tag.get("deprecated", False):
        # Return a specific message indicating the tag is already deprecated
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS, error_string="This tag is already deprecated.", http_status_code=400
        )

    # Since the tag is not already deprecated, set the 'deprecated' status to True
    request.app.pseudo_tag_definitions_collection.update_one(query, {"$set": {"deprecated": True}})

    # Retrieve the updated tag to confirm the change
    updated_tag = request.app.pseudo_tag_definitions_collection.find_one(query)

    # Serialize ObjectId to string if necessary
    updated_tag = {k: str(v) if isinstance(v, ObjectId) else v for k, v in updated_tag.items()}

    # Return the updated tag object, indicating the deprecation was successful
    return response_handler.create_success_response_v1(response_data=updated_tag, http_status_code=200)



@router.patch("/pseudotag-categories/set-deprecated", 
              tags=["pseudotag-categories"],
              status_code=200,
              description="Set the 'deprecated' status of a tag category to True",
              response_model=StandardSuccessResponseV1[PseudoTagCategory], 
              responses=ApiResponseHandlerV1.listErrors([400, 404, 422, 500]))
def set_tag_category_deprecated(request: Request, pseudo_tag_category_id: int):
    response_handler = ApiResponseHandlerV1(request)

    query = {"pseudo_tag_category_id": pseudo_tag_category_id}
    existing_tag_category = request.app.pseudo_tag_categories_collection.find_one(query)

    if existing_tag_category is None:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.ELEMENT_NOT_FOUND, error_string="Tag category not found.", http_status_code=404
        )

    # Check if the tag category is already deprecated
    if existing_tag_category.get("deprecated", False):
        # Return a specific message indicating the tag category is already deprecated
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS, error_string="This tag category is already deprecated.", http_status_code=400
        )

    # Set the 'deprecated' status to True since it's not already deprecated
    request.app.pseudo_tag_categories_collection.update_one(query, {"$set": {"deprecated": True}})

    # Retrieve the updated tag category to confirm the change
    updated_tag_category = request.app.pseudo_tag_categories_collection.find_one(query)

    # Serialize ObjectId to string if necessary
    updated_tag_category = {k: str(v) if isinstance(v, ObjectId) else v for k, v in updated_tag_category.items()}

    # Return the updated tag category object indicating the deprecation was successful
    return response_handler.create_success_response_v1(response_data=updated_tag_category, http_status_code=200)
