
from fastapi import APIRouter, Request,  Query
from .api_utils import ApiResponseHandlerV1, StandardSuccessResponseV1, ErrorCode, WasPresentResponse, TagCountResponse, validate_date_format, TagListForImages
from .mongo_schemas import ExtractImageData, ListExtractImageData
from orchestration.api.mongo_schema.tag_schemas import ListExternalImageTag, ImageTag
from datetime import datetime
import uuid

router = APIRouter()

@router.get("/extracts/get-current-data-batch-sequential-id", 
            description="Get the sequential id for numpy file batches stored for a dataset",
            tags=["extracts"])
async def get_current_data_batch_sequential_id(request: Request, dataset: str):

    # get batch counter
    counter = request.app.extract_data_batch_sequential_id.find_one({"dataset": dataset})
    # create counter if it doesn't exist already
    if counter is None:
        # insert the new counter
        insert_result= request.app.extract_data_batch_sequential_id.insert_one({"dataset": dataset, "sequence_number": 0, "complete": True})
        # Retrieve the inserted counter using the inserted_id
        counter = request.app.extract_data_batch_sequential_id.find_one({'_id': insert_result.inserted_id})
    
    # remove _id field
    counter.pop("_id")

    return counter

@router.get("/extracts/get-next-data-batch-sequential-id", 
            description="Increment the sequential id for numpy file batches stored for a dataset",
            tags=["extracts"])
async def get_next_data_batch_sequential_id(request: Request, dataset: str, complete: bool):

    # get batch counter
    counter = request.app.extract_data_batch_sequential_id.find_one({"dataset": dataset})
    # create counter if it doesn't exist already
    if counter is None:
        request.app.extract_data_batch_sequential_id.insert_one({"dataset": dataset})

    # get current last batch count
    counter_seq = counter["sequence_number"] if counter else 0
    counter_seq += 1
    try:
        ret = request.app.counters_collection.update_one(
            {"dataset": dataset},
            {"$set": 
                {
                    "sequence_number": counter_seq,
                    "complete": complete
                }})
    except Exception as e:
        raise Exception("Updating of classifier counter failed: {}".format(e))

    return counter_seq

@router.post("/extracts/add-extracted-image", 
            description="Add an extracted image data",
            tags=["extracts"],  
            response_model=StandardSuccessResponseV1[ListExtractImageData],  
            responses=ApiResponseHandlerV1.listErrors([404,422, 500]))
async def add_extract(request: Request, image_data: ExtractImageData):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:

        image_data.uuid = str(uuid.uuid4())

        existed = request.app.extracts_collection.find_one({
            "image_hash": image_data.image_hash
        })

        if existed is None:
            image_data.upload_date = str(datetime.now())
            request.app.extracts_collection.insert_one(image_data.to_dict())
        else:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string="An image with this hash already exists",
                http_status_code=400
            )
        
        return api_response_handler.create_success_response_v1(
            response_data={"data": image_data.to_dict()},
            http_status_code=200  
        )
    
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )

@router.get("/extracts/get-all-extracts-list", 
            description="Get all extracted images. If 'dataset' parameter is set, it only returns images from that dataset, and if the 'size' parameter is set, a random sample of that size will be returned.",
            tags=["extracts"],  
            response_model=StandardSuccessResponseV1[ListExtractImageData],  
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def get_all_extracts_list(request: Request, dataset: str=None, size: int = None):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        query={}
        if dataset:
            query['dataset']= dataset

        aggregation_pipeline = [{"$match": query}]

        if size:
            aggregation_pipeline.append({"$sample": {"size": size}})

        image_data_list = list(request.app.extracts_collection.aggregate(aggregation_pipeline))

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
    
@router.delete("/extracts/delete-extract", 
            description="Delete an extracted image",
            tags=["extracts"],  
            response_model=StandardSuccessResponseV1[WasPresentResponse],  
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def delete_extract_image_data(request: Request, image_hash: str):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        result = request.app.extracts_collection.delete_one({
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

@router.post("/extracts/add-tag-to-extract",
             status_code=201,
             tags=["extracts"],  
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

        image = request.app.extracts_collection.find_one({'image_hash': image_hash})
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
            "image_source": "extracts"
        })
        if existing_image_tag:
            # Remove the '_id' field before returning the response
            existing_image_tag.pop('_id', None)
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
            "image_source": "extracts",
            "user_who_created": user_who_created,
            "tag_count": 1,  # Since this is a new tag for this image, set count to 1
            "creation_time": date_now
        }
        result = request.app.image_tags_collection.insert_one(image_tag_data)
        
        # Add the generated _id to the image_tag_data
        image_tag_data['_id'] = str(result.inserted_id)

        # Remove the '_id' field before returning the response
        image_tag_data.pop('_id', None)

        return response_handler.create_success_response_v1(
            response_data=image_tag_data, 
            http_status_code=200
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e), 
            http_status_code=500
        )
    


@router.delete("/extracts/remove-tag-from-extract",
               status_code=200,
               tags=["extracts"],
               response_model=StandardSuccessResponseV1[WasPresentResponse],
               responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def remove_tag_from_image(request: Request, tag_id: int, image_hash: str):
    response_handler = ApiResponseHandlerV1(request)
    try:
        # Check if the tag is associated with the image with the specific image_source
        existing_image_tag = request.app.image_tags_collection.find_one({
            "tag_id": tag_id, 
            "image_hash": image_hash, 
            "image_source": "extracts"
        })
        if not existing_image_tag:
            return response_handler.create_success_delete_response_v1(
                False,
                http_status_code=200
            )

        # Remove the tag
        request.app.image_tags_collection.delete_one({
            "tag_id": tag_id, 
            "image_hash": image_hash, 
            "image_source": "extracts"
        })

        return response_handler.create_success_delete_response_v1(
            True,
            http_status_code=200
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e), 
            http_status_code=500
        )
    


@router.get("/extracts/get-images-by-tag-id", 
            tags=["extracts"], 
            status_code=200,
            description="Get extract images by tag_id",
            response_model=StandardSuccessResponseV1[ListExternalImageTag], 
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def get_extracts_by_tag_id(
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
        query = {"tag_id": tag_id, "image_source": "extracts"}
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



@router.get("/extracts/get-tag-list-for-image", 
            response_model=StandardSuccessResponseV1[TagListForImages], 
            description="Get tag list for image",
            tags=["extracts"],
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([400, 404, 422, 500]))
def get_tag_list_for_extract_image(request: Request, file_hash: str):
    response_handler = ApiResponseHandlerV1(request)
    try:
        # Fetch image tags based on image_hash
        image_tags_cursor = request.app.image_tags_collection.find({"image_hash": file_hash, "image_source": "extracts"})
        
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



@router.get("/extracts/get-images-count-by-tag-id",
            status_code=200,
            tags=["extracts"],
            description="Get count of extract images with a specific tag",
            response_model=StandardSuccessResponseV1[TagCountResponse],
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def get_images_count_by_tag_id(request: Request, tag_id: int):
    response_handler = ApiResponseHandlerV1(request)
    try :
        # Build the query to include the image_source as "extracts"
        query = {"tag_id": tag_id, "image_source": "extracts"}
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
    