
from fastapi import APIRouter, Body, Request, HTTPException, Query
from typing import Optional
from .api_utils import ApiResponseHandlerV1, StandardSuccessResponseV1, ErrorCode, WasPresentResponse, DeletedCount, validate_date_format, TagListForImages, TagCountResponse
from .mongo_schemas import ExternalImageData, ImageHashRequest, ListExternalImageData, ListImageHashRequest, ExternalImageDataV1
from orchestration.api.mongo_schema.tag_schemas import ExternalImageTag, ListExternalImageTag, ImageTag, ListImageTag
from typing import List
from datetime import datetime, timedelta
from pymongo import UpdateOne
from utility.minio import cmd
import uuid

router = APIRouter()

@router.post("/external-images/add-external-image", 
            description="Add an external image data with a randomly generated UUID by uuid4",
            tags=["external-images"],  
            response_model=StandardSuccessResponseV1[ExternalImageDataV1],  
            responses=ApiResponseHandlerV1.listErrors([404,422, 500])) 
async def add_external_image_data(request: Request, image_data: ExternalImageData):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)


    try:

        objects = cmd.get_list_of_objects(request.app.minio_client, "datasets")
        dataset_path = f'{image_data.dataset}'
        
        if dataset_path not in objects:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string=f"Dataset '{image_data.dataset}' does not exist.",
                http_status_code=422,
            )
    
        # Check if the image data already exists
        existed = request.app.external_images_collection.find_one({
            "image_hash": image_data.image_hash
        })

        if existed:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string="Image data with this hash already exists.",
                http_status_code=422
            )

        # Add a new UUID and upload date to the image data
        image_data_dict = image_data.to_dict()
        image_data_dict['uuid'] = str(uuid.uuid4())
        image_data_dict['upload_date'] = str(datetime.now())
        
        # Insert the new image data into the collection
        request.app.external_images_collection.insert_one(image_data_dict)

        image_data_dict.pop('_id', None)
        
        return api_response_handler.create_success_response_v1(
            response_data=image_data_dict,
            http_status_code=200
        )

    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )



@router.post("/external-images/add-external-image-list", 
            description="Add list of external image data",
            tags=["external-images"],  
            response_model=StandardSuccessResponseV1[ListExternalImageData],  
            responses=ApiResponseHandlerV1.listErrors([422, 500]))
async def add_external_image_data_list(request: Request, image_data_list: List[ExternalImageData]):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        for image_data in image_data_list:
            existed = request.app.external_images_collection.find_one({
                "image_hash": image_data.image_hash
            })

            if existed:
                return api_response_handler.create_error_response_v1(
                    error_code=ErrorCode.INVALID_PARAMS,
                    error_string="Image data with this hash already exists.",
                    http_status_code=422
                )
            else:
                # Add a new UUID and upload date to the image data
                image_data_dict = image_data.to_dict()
                image_data_dict['uuid'] = str(uuid.uuid4())
                image_data_dict['upload_date'] = str(datetime.now())

                # Insert the new image data into the collection
                request.app.external_images_collection.insert_one(image_data_dict)

        # Remove the _id field from the response data
        response_data = [image_data_dict for image_data_dict in image_data_list]
        for data in response_data:
            data.pop('_id', None)

        return api_response_handler.create_success_response_v1(
            response_data={"data": response_data},
            http_status_code=200
        )
    
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )

    
@router.get("/external-images/get-external-image-list", 
            description="changed with /external-images/get-external-image-list-v1",
            tags=["deprecated3"],  
            response_model=StandardSuccessResponseV1[ListExternalImageData],  
            responses=ApiResponseHandlerV1.listErrors([404,422, 500]))
async def get_external_image_data_list(request: Request, image_hash_list: List[str]):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        list_external_images = []
        for image_hash in image_hash_list:

            result = request.app.external_images_collection.find_one({
                "image_hash": image_hash
            }, {"_id": 0})
        
            if result is not None:
                list_external_images.append(result)

        return api_response_handler.create_success_response_v1(
            response_data={"data": list_external_images},
            http_status_code=200  
        )
    
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )
    

@router.post("/external-images/get-external-image-list-v1", 
             description="Get list of external image data by image hash list",
             tags=["external-images"],  
             response_model=StandardSuccessResponseV1[ListExternalImageData],  
             responses=ApiResponseHandlerV1.listErrors([404,422, 500]))
async def get_external_image_data_list(request: Request, body: ListImageHashRequest):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        list_external_images = []
        for image_hash in body.image_hash_list:

            result = request.app.external_images_collection.find_one({
                "image_hash": image_hash
            }, {"_id": 0})
        
            if result is not None:
                list_external_images.append(result)

        return api_response_handler.create_success_response_v1(
            response_data={"data": list_external_images},
            http_status_code=200  
        )
    
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )
        
@router.get("/external-images/get-all-external-image-list", 
            description="Get all external image data. If the 'size' parameter is set, a random sample of that size will be returned.",
            tags=["external-images"],  
            response_model=StandardSuccessResponseV1[List[ExternalImageData]],  
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def get_all_external_image_data_list(request: Request, size: int = None):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        query = {}

        aggregation_pipeline = [{"$match": query}]

        if size:
            aggregation_pipeline.append({"$sample": {"size": size}})

        image_data_list = list(request.app.external_images_collection.aggregate(aggregation_pipeline))

        for image_data in image_data_list:
            image_data.pop('_id', None)  # Remove the auto-generated field

        return api_response_handler.create_success_response_v1(
            response_data={"data": image_data_list},
            http_status_code=200  
        )
    
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )

@router.get("/external-images/get-all-external-image-list-v1", 
            description="Get all external image data for a specific dataset. If the 'size' parameter is set, a random sample of that size will be returned.",
            tags=["external-images"],  
            response_model=StandardSuccessResponseV1[List[ExternalImageData]],  
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def get_all_external_image_data_list(request: Request, dataset: str, size: int = None):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        query = {"dataset": dataset}

        aggregation_pipeline = [{"$match": query}]

        if size:
            aggregation_pipeline.append({"$sample": {"size": size}})

        image_data_list = list(request.app.external_images_collection.aggregate(aggregation_pipeline))

        for image_data in image_data_list:
            image_data.pop('_id', None)  # Remove the auto-generated field

        return api_response_handler.create_success_response_v1(
            response_data={"data": image_data_list},
            http_status_code=200  
        )
    
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )




@router.delete("/external-images/delete-external-image", 
            description="Delete an external image data",
            tags=["external-images"],  
            response_model=StandardSuccessResponseV1[WasPresentResponse],  
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def delete_external_image_data(request: Request, image_hash: str):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        result = request.app.external_images_collection.delete_one({
            "image_hash": image_hash
        })
        
        if result.deleted_count == 0:
            return api_response_handler.create_success_delete_response_v1(
                response_data=False, 
                http_status_code=200
            )
        
        return api_response_handler.create_success_delete_response_v1(
                response_data=True, 
                http_status_code=200
            )
    
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )

    

@router.delete("/external-images/delete-external-image-list", 
            description="Delete an external image data list",
            tags=["external-images"],  
            response_model=StandardSuccessResponseV1[DeletedCount],  
            responses=ApiResponseHandlerV1.listErrors([404,422, 500]))
async def delete_external_image_data_list(request: Request, image_hash_list:List[str] ):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        deleted_count = 0
        for image_hash in image_hash_list:
            result = request.app.external_images_collection.delete_one({
                "image_hash": image_hash
            })
            
            if result.deleted_count > 0:
                deleted_count += 1
            
        return api_response_handler.create_success_response_v1(
            response_data={'deleted_count': deleted_count},
            http_status_code=200  
        )
    
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )
    

@router.put("/external-images/add-task-attributes-v1",
              description="Add or update the task attributes of an external image, No old attibute will be deleted, this function only adds and overwrites",
              tags=["external-images"],  
              response_model=StandardSuccessResponseV1[ExternalImageDataV1],  
              responses=ApiResponseHandlerV1.listErrors([404,422, 500]))    
@router.patch("/external-images/add-task-attributes",
              description="changed with /external-images/add-task-attributes-v1",
              tags=["deprecated3"],  
              response_model=StandardSuccessResponseV1[ListExternalImageData],  
              responses=ApiResponseHandlerV1.listErrors([404,422, 500]))
async def add_task_attributes(request: Request, image_hash: str, data_dict: dict = Body(...)):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
     
    try: 
        image = request.app.external_images_collection.find_one(
            {"image_hash": image_hash},
        )

        if image:
            task_attributs_dict= image['task_attributes_dict']
            for key, value in data_dict.items():
                task_attributs_dict[key]= value

            image = request.app.external_images_collection.find_one_and_update(
                {"image_hash": image_hash},
                {"$set": {"task_attributes_dict": task_attributs_dict}},
                return_document=True
            )

            image.pop('_id', None)

            return api_response_handler.create_success_response_v1(
                response_data={"data": image},
                http_status_code=200  
            )       

        else:
            return api_response_handler.create_error_response_v1(
                    error_code=ErrorCode.INVALID_PARAMS, 
                    error_string="There is no external image data with image hash: {}".format(image_hash), 
                    http_status_code=400)
    
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )



@router.post("/external-images/update-uuids",
             status_code=200,
             tags=["external-images"],  
             response_model=StandardSuccessResponseV1,  
             responses=ApiResponseHandlerV1.listErrors([404, 422]))
def update_external_images(request: Request):
    api_response_handler = ApiResponseHandlerV1(request)

    # Fetch all items from the external_images_collection
    items = list(request.app.external_images_collection.find())

    updated_count = 0
    for item in items:
        # Generate a new UUID
        new_uuid = str(uuid.uuid4())

        # Construct the updated document with uuid before image_hash
        updated_item = {"uuid": new_uuid, **item}
        updated_item.pop('_id', None)  # Remove the '_id' field to avoid duplication issues

        # Perform the update operation
        result = request.app.external_images_collection.update_one(
            {"_id": item["_id"]},
            {"$set": updated_item}
        )
        if result.modified_count > 0:
            updated_count += 1

    if updated_count == 0:
        raise HTTPException(status_code=404, detail="No items updated")

    # Return a standardized success response with the update result
    return api_response_handler.create_success_response_v1(
        response_data={'updated_count': updated_count},
        http_status_code=200
    ) 


@router.post("/external-images/add-tag-to-external-image",
             status_code=201,
             tags=["external-images"],  
             response_model=StandardSuccessResponseV1[ImageTag], 
             responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def add_tag_to_image(request: Request, tag_id: int, image_hash: str, tag_type: int, user_who_created: str):
    response_handler = ApiResponseHandlerV1(request)
    try:
        date_now = datetime.now().isoformat()
    
        existing_tag = request.app.tag_definitions_collection.find_one({"tag_id": tag_id})
        if not existing_tag:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND, 
                error_string="Tag does not exist!", 
                http_status_code=400
            )

        image = request.app.external_images_collection.find_one({'image_hash': image_hash})
        if not image:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND, 
                error_string="No image found with the given hash", 
                http_status_code=400
            )

        file_path = image.get("file_path", "")
        
        # Check if the tag is already associated with the image
        existing_image_tag = request.app.image_tags_collection.find_one({
            "tag_id": tag_id, 
            "image_hash": image_hash, 
            "image_source": "external-image"
        })
        if existing_image_tag:
            # Return a success response indicating that the tag has already been added to the image
            return response_handler.create_success_response_v1(
                response_data=existing_image_tag, 
                http_status_code=200
            )

        # Add new tag to image
        image_tag_data = {
            "tag_id": tag_id,
            "file_path": file_path,  
            "image_hash": image_hash,
            "tag_type": tag_type,
            "image_source": "external-image",
            "user_who_created": user_who_created,
            "tag_count": 1,  # Since this is a new tag for this image, set count to 1
            "creation_time": date_now
        }
        request.app.image_tags_collection.insert_one(image_tag_data)

        return response_handler.create_success_response_v1(
            response_data=image_tag_data, 
            http_status_code=200
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string="Internal server error", 
            http_status_code=500
        )



@router.delete("/external-images/remove-tag-from-external-image",
               status_code=200,
               tags=["external-images"],
               response_model=StandardSuccessResponseV1[WasPresentResponse],
               responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def remove_tag_from_image(request: Request, tag_id: int, image_hash: str):
    response_handler = ApiResponseHandlerV1(request)
    try:
        # Check if the tag is associated with the image with the specific image_source
        existing_image_tag = request.app.image_tags_collection.find_one({
            "tag_id": tag_id, 
            "image_hash": image_hash, 
            "image_source": "external-image"
        })
        if not existing_image_tag:
            return response_handler.create_success_delete_response_v1(
                response_data={"wasPresent": False},
                http_status_code=200
            )

        # Remove the tag
        request.app.image_tags_collection.delete_one({
            "tag_id": tag_id, 
            "image_hash": image_hash, 
            "image_source": "external-image"
        })

        return response_handler.create_success_delete_response_v1(
            response_data={"wasPresent": True},
            http_status_code=200
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string="Internal server error", 
            http_status_code=500
        )



@router.get("/external-images/get-images-by-tag-id", 
            tags=["external-images"], 
            status_code=200,
            description="Get external images by tag_id",
            response_model=StandardSuccessResponseV1[ListExternalImageTag], 
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def get_external_images_by_tag_id(
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
        query = {"tag_id": tag_id, "image_source": "external-image"}
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
        return response_handler.create_success_response_v1(
            response_data={"images": image_info_list}, 
            http_status_code=200,
        )

    except Exception as e:
        # Log the exception details here, if necessary
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, error_string="Internal Server Error", http_status_code=500
        )



@router.get("/external-images/get-tag-list-for-image", 
            response_model=StandardSuccessResponseV1[TagListForImages], 
            description="Get tag list for image",
            tags=["external-images"],
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([400, 404, 422, 500]))
def get_tag_list_for_external_image(request: Request, file_hash: str):
    response_handler = ApiResponseHandlerV1(request)
    try:
        # Fetch image tags based on image_hash
        image_tags_cursor = request.app.image_tags_collection.find({"image_hash": file_hash, "image_source": "external-image"})
        
        # Process the results
        tags_list = []
        for tag_data in image_tags_cursor:
            # Find the tag definition
            tag_definition = request.app.tag_definitions_collection.find_one({"tag_id": tag_data["tag_id"]})
            if tag_definition:
                # Find the tag category and determine if it's deprecated
                category = request.app.tag_categories_collection.find_one({"tag_category_id": tag_definition.get("tag_category_id")})
                deprecated_tag_category = category['deprecated'] if category else False
                
                # Create a dictionary representing TagDefinition with tag_type and deprecated_tag_category
                tag_definition_dict = {
                    "tag_id": tag_definition["tag_id"],
                    "tag_string": tag_definition["tag_string"],
                    "tag_type": tag_data.get("tag_type"),
                    "tag_category_id": tag_definition.get("tag_category_id"),
                    "tag_description": tag_definition["tag_description"],
                    "tag_vector_index": tag_definition.get("tag_vector_index", -1),
                    "deprecated": tag_definition.get("deprecated", False),
                    "deprecated_tag_category": deprecated_tag_category,
                    "user_who_created": tag_definition["user_who_created"],
                    "creation_time": tag_definition.get("creation_time", None)
                }

                tags_list.append(tag_definition_dict)
        
        # Return the list of tags including 'deprecated_tag_category'
        return response_handler.create_success_response_v1(
            response_data={"tags": tags_list},
            http_status_code=200,
        )
    except Exception as e:
        # Optional: Log the exception details here
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )



@router.get("/external-images/get-images-count-by-tag-id",
            status_code=200,
            tags=["external-images"],
            description="Get count of external images with a specific tag",
            response_model=StandardSuccessResponseV1[TagCountResponse],
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def get_images_count_by_tag_id(request: Request, tag_id: int):
    response_handler = ApiResponseHandlerV1(request)
    try :
        # Build the query to include the image_source as "external-image"
        query = {"tag_id": tag_id, "image_source": "external-image"}
        count = request.app.image_tags_collection.count_documents(query)

        # Return the count even if it is zero
        return response_handler.create_success_response_v1(
            response_data={"tag_id": tag_id, "count": count},
            http_status_code=200
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string="Internal server error",
            http_status_code=500
        )


@router.get("/external-images/list-images-v1",
            status_code=200,
            tags=["external-images"],
            response_model=StandardSuccessResponseV1[List[ExternalImageData]],
            description="List external images with optional filtering and pagination",
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def list_external_images_v1(
    request: Request,
    limit: int = Query(20, description="Limit on the number of results returned"),
    offset: int = Query(0, description="Offset for the results to be returned"),
    start_date: Optional[str] = Query(None, description="Start date for filtering results (YYYY-MM-DDTHH:MM:SS)"),
    end_date: Optional[str] = Query(None, description="End date for filtering results (YYYY-MM-DDTHH:MM:SS)"),
    order: str = Query("desc", description="Order in which the data should be returned. 'asc' for oldest first, 'desc' for newest first"),
    time_interval: Optional[int] = Query(None, description="Time interval in minutes or hours"),
    time_unit: str = Query("minutes", description="Time unit, either 'minutes' or 'hours'")
):
    response_handler = ApiResponseHandlerV1(request)

    try:
        # Calculate the time threshold based on the current time and the specified interval
        if time_interval is not None:
            current_time = datetime.utcnow()
            if time_unit == "minutes":
                threshold_time = current_time - timedelta(minutes=time_interval)
            elif time_unit == "hours":
                threshold_time = current_time - timedelta(hours=time_interval)
            else:
                raise HTTPException(status_code=400, detail="Invalid time unit. Use 'minutes' or 'hours'.")

            # Convert threshold_time to a string in ISO format
            threshold_time_str = threshold_time.isoformat(timespec='milliseconds') 
        else:
            threshold_time_str = None

        # Validate start_date and end_date
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

        # Build the query
        query = {}
        if start_date and end_date:
            query["creation_time"] = {"$gte": validated_start_date, "$lte": validated_end_date}
        elif start_date:
            query["creation_time"] = {"$gte": validated_start_date}
        elif end_date:
            query["creation_time"] = {"$lte": validated_end_date}
        elif threshold_time_str:
            query["creation_time"] = {"$gte": threshold_time_str}

        # Decide the sort order
        sort_order = -1 if order == "desc" else 1

        # Query the external_images_collection using the constructed query
        images_cursor = request.app.external_images_collection.find(query).sort("creation_time", sort_order).skip(offset).limit(limit)

        # Collect the metadata for the images that match the query
        images_metadata = []
        for image in images_cursor:
            image.pop('_id', None)  # Remove the auto-generated field
            images_metadata.append(image)

        return response_handler.create_success_response_v1(
            response_data=images_metadata,
            http_status_code=200
        )
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )
        
@router.get("/external-images/get-unique-datasets", 
            description="Get all unique dataset names in the external images collection.",
            tags=["external-images"],  
            response_model=StandardSuccessResponseV1[str],  
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def get_unique_datasets(request: Request):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        # Use aggregation pipeline to get unique dataset names
        aggregation_pipeline = [
            {"$group": {"_id": "$dataset"}},
            {"$project": {"_id": 0, "dataset": "$_id"}}
        ]

        datasets_cursor = request.app.external_images_collection.aggregate(aggregation_pipeline)
        datasets = [doc["dataset"] for doc in datasets_cursor]

        return api_response_handler.create_success_response_v1(
            response_data={"datasets": datasets},
            http_status_code=200  
        )
    
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )        