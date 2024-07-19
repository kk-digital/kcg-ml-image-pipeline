
from fastapi import APIRouter, Request,  Query, HTTPException, status

from utility.path import separate_bucket_and_file_path
from .mongo_schemas import ExtractImageData, ListExtractImageData, Dataset, ListExtractImageDataV1, ListDataset , ListExtractImageDataWithScore, ExtractImageDataV1
from pymongo import ReturnDocument
from .api_utils import ApiResponseHandlerV1, StandardSuccessResponseV1, ErrorCode, WasPresentResponse, TagCountResponse, get_minio_file_path, get_next_external_dataset_seq_id, update_external_dataset_seq_id, validate_date_format, TagListForImages, TagListForImagesV1,PrettyJSONResponse
from orchestration.api.mongo_schema.tag_schemas import ListExternalImageTag, ImageTag
from datetime import datetime
from typing import Optional
import uuid
from typing import List
from datetime import datetime, timedelta
from .api_clip import http_clip_server_get_cosine_similarity_list


router = APIRouter()

extracts = "extract_image"

@router.get("/extracts/get-current-data-batch-sequential-id", 
            description="Get the sequential id for file batches stored for a dataset",
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
        # insert the new counter
        insert_result= request.app.extract_data_batch_sequential_id.insert_one({"dataset": dataset, "sequence_number": 0, "complete": True})
        # Retrieve the inserted counter using the inserted_id
        counter = request.app.extract_data_batch_sequential_id.find_one({'_id': insert_result.inserted_id})

    # get current last batch count
    counter_seq = counter["sequence_number"] if counter else 0
    counter_seq += 1

    try:
        counter = request.app.extract_data_batch_sequential_id.find_one_and_update(
            {"dataset": dataset},
            {"$set": 
                {
                    "sequence_number": counter_seq,
                    "complete": complete
                }
            },
            return_document=ReturnDocument.AFTER
            )
    except Exception as e:
        raise Exception("Updating of classifier counter failed: {}".format(e))

    # remove _id field
    counter.pop("_id")

    return counter

@router.delete("/extracts/delete-dataset-batch-sequential-id", 
               response_model=StandardSuccessResponseV1[WasPresentResponse], 
               description="remove the batch sequential id for a dataset", 
               tags=["extracts"], 
               status_code=200,
               responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
async def remove_current_data_batch_sequential_id(request: Request, dataset: str ):

    response_handler = ApiResponseHandlerV1(request)

    # Check if the rank exists
    query = {"dataset": dataset}
    sequential_id = request.app.extract_data_batch_sequential_id.find_one(query)
    
    if sequential_id is None:
        # Return standard response with wasPresent: false
        return response_handler.create_success_delete_response_v1(
                                                           False,
                                                           http_status_code=200
                                                           )

    # Remove the sequential id
    request.app.extract_data_batch_sequential_id.delete_one(query)

    # Return standard response with wasPresent: true
    return response_handler.create_success_response_v1(
                                                       response_data={"wasPresent": True},
                                                       http_status_code=200
                                                       )

@router.post("/extracts/add-extracted-image", 
            description="changed with /extracts/add-extracted-image-v1 ",
            tags=["deprecated3"],  
            response_model=StandardSuccessResponseV1[ListExtractImageData],  
            responses=ApiResponseHandlerV1.listErrors([404,422, 500]))
async def add_extract(request: Request, image_data: ExtractImageData):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        
        dataset_result = request.app.extract_datasets_collection.find_one({"dataset_name": image_data.dataset})
        if not dataset_result:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND, 
                error_string=f"{image_data.dataset} dataset does not exist",
                http_status_code=422
            )

        image_data.uuid = str(uuid.uuid4())

        existed = request.app.extracts_collection.find_one({
            "image_hash": image_data.image_hash
        })

        if existed is None:
            image_data.upload_date = str(datetime.now())
            # set minio path using sequential id
            next_seq_id = get_next_external_dataset_seq_id(request, bucket="extracts", dataset=image_data.dataset)
            image_data.file_path = get_minio_file_path(next_seq_id,
                                                    "extracts",    
                                                    image_data.dataset, 
                                                    'jpg')
            
            request.app.extracts_collection.insert_one(image_data.to_dict())
        else:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string="An image with this hash already exists",
                http_status_code=400
            )
        
        # update sequential id
        update_external_dataset_seq_id(request=request, bucket="extracts", dataset=image_data.dataset, seq_id=next_seq_id) 

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
    

@router.post("/extracts/add-extracted-image-v1", 
            description="Add an extracted image data",
            tags=["extracts"],  
            response_model=StandardSuccessResponseV1[ExtractImageData],  
            responses=ApiResponseHandlerV1.listErrors([404,422, 500]))
async def add_extract(request: Request, image_data: ExtractImageDataV1):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        dataset_result = request.app.extract_datasets_collection.find_one({"dataset_name": image_data.dataset})
        if not dataset_result:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND, 
                error_string=f"{image_data.dataset} dataset does not exist",
                http_status_code=422
            )

        existed = request.app.extracts_collection.find_one({
            "image_hash": image_data.image_hash
        })

        if existed is None:
            image_data_dict = image_data.to_dict()
            image_data_dict['uuid'] = str(uuid.uuid4())
            image_data_dict['upload_date'] = str(datetime.now())
            next_seq_id = get_next_external_dataset_seq_id(request, bucket="extracts", dataset=image_data.dataset)
            image_data_dict['file_path'] = get_minio_file_path(next_seq_id, "extracts", image_data.dataset, 'jpg')

            
            request.app.extracts_collection.insert_one(image_data_dict)
            image_data_dict.pop('_id', None)
            
        else:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string="An image with this hash already exists",
                http_status_code=400
            )
        
        update_external_dataset_seq_id(request=request, bucket="extracts", dataset=image_data.dataset, seq_id=next_seq_id) 

        return api_response_handler.create_success_response_v1(
            response_data= image_data_dict,
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

@router.delete("/extracts/delete-extract-dataset", 
            description="Delete all the extracted images in a dataset",
            tags=["extracts"],  
            response_model=StandardSuccessResponseV1[WasPresentResponse],  
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def delete_extract_dataset_data(request: Request, dataset: str):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        result = request.app.extracts_collection.delete_many({
            "dataset": dataset
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
             tags=["deprecated3"],  
             description="changed with /tags/add-tag-to-image-v2",
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
            "image_source": extracts
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
            "image_source": extracts,
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
               tags=["deprecated3"],
               description="changed with /tags/remove-tag-from-image-v1/{tag_id}",
               response_model=StandardSuccessResponseV1[WasPresentResponse],
               responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def remove_tag_from_image(request: Request, tag_id: int, image_hash: str):
    response_handler = ApiResponseHandlerV1(request)
    try:
        # Check if the tag is associated with the image with the specific image_source
        existing_image_tag = request.app.image_tags_collection.find_one({
            "tag_id": tag_id, 
            "image_hash": image_hash, 
            "image_source": extracts
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
            "image_source": extracts
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
            tags=["deprecated3"], 
            status_code=200,
            description="changed with /tags/get-images-by-tag-id-v1",
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
        query = {"tag_id": tag_id, "image_source": extracts}
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
            description="changed with /tags/get-tag-list-for-image-v2",
            tags=["deprecated3"],
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([400, 404, 422, 500]))
def get_tag_list_for_extract_image(request: Request, file_hash: str):
    response_handler = ApiResponseHandlerV1(request)
    try:
        # Fetch image tags based on image_hash
        image_tags_cursor = request.app.image_tags_collection.find({"image_hash": file_hash, "image_source": extracts})
        
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
            tags=["deprecated3"],
            description="changed with tags/get-images-count-by-tag-id-v1",
            response_model=StandardSuccessResponseV1[TagCountResponse],
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def get_images_count_by_tag_id(request: Request, tag_id: int):
    response_handler = ApiResponseHandlerV1(request)
    try :
        # Build the query to include the image_source as "extracts"
        query = {"tag_id": tag_id, "image_source": extracts}
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
    

@router.post("/extract-images/add-new-dataset",
            description="Add new dataset in MongoDB",
            tags=["extracts"],
            response_model=StandardSuccessResponseV1[Dataset],  
            responses=ApiResponseHandlerV1.listErrors([400, 422]))
async def add_new_dataset(request: Request, dataset: Dataset):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    if request.app.extract_datasets_collection.find_one({"dataset_name": dataset.dataset_name}):
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string='Dataset already exists',
            http_status_code=400
        )

    # Find the current highest dataset_id
    highest_dataset = request.app.extract_datasets_collection.find_one(
        sort=[("dataset_id", -1)]
    )
    next_dataset_id = (highest_dataset["dataset_id"] + 1) if highest_dataset else 0

    # Add the dataset_id to the dataset
    dataset_dict = dataset.to_dict()
    dataset_dict["dataset_id"] = next_dataset_id

    # Insert the new dataset with dataset_id
    request.app.extract_datasets_collection.insert_one(dataset_dict)

    return response_handler.create_success_response_v1(
        response_data={"dataset_name": dataset.dataset_name, "dataset_id": next_dataset_id}, 
        http_status_code=200
    )
 

@router.get("/extract-images/list-datasets",
            description="list datasets from mongodb",
            tags=["extracts"],
            response_model=StandardSuccessResponseV1[ListDataset],  
            responses=ApiResponseHandlerV1.listErrors([422]))
async def list_datasets(request: Request):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    datasets = list(request.app.extract_datasets_collection.find({}))
    for dataset in datasets:
        dataset.pop('_id', None)

    return response_handler.create_success_response_v1(
                response_data={'datasets': datasets}, 
                http_status_code=200
            )  

@router.delete("/extract-images/remove-dataset",
               description="Remove dataset",
               tags=["extracts"],
               response_model=StandardSuccessResponseV1[WasPresentResponse],  
               responses=ApiResponseHandlerV1.listErrors([422]))
async def remove_dataset(request: Request, dataset: str = Query(...)):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    # Check if the dataset contains any objects (images) 
    image_count = request.app.extract_datasets_collection.count_documents({"dataset_name": dataset, "images": {"$exists": True, "$ne": []}})
    if image_count > 0:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string=f"Dataset '{dataset}' contains images and cannot be deleted.",
            http_status_code=422
        )

    # Attempt to delete the dataset
    dataset_result = request.app.extract_datasets_collection.delete_one({"dataset_name": dataset})

    # Check if the dataset was present and deleted
    was_present = dataset_result.deleted_count > 0

    # Using the check to determine which response to send
    if was_present:
        # If the dataset was deleted, return True
        return response_handler.create_success_delete_response_v1(
            True, 
            http_status_code=200
        )
    else:
        # If the dataset was not found, return False
        return response_handler.create_success_delete_response_v1(
            False, 
            http_status_code=200
        )


@router.get("/extract-images/list-images",
            status_code=200,
            tags=["extracts"],
            response_model=StandardSuccessResponseV1[ListExtractImageDataV1],
            description="List extracts images with optional filtering and pagination",
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
async def list_extract_images_v1(
    request: Request,
    dataset: Optional[str] = Query(None, description="Dataset to filter the results by"),
    limit: int = Query(20, description="Limit on the number of results returned"),
    offset: int = Query(0, description="Offset for the results to be returned"),
    start_date: Optional[str] = Query(None, description="Start date for filtering results (YYYY-MM-DDTHH:MM:SS)"),
    end_date: Optional[str] = Query(None, description="End date for filtering results (YYYY-MM-DDTHH:MM:SS)"),
    order: str = Query("desc", description="Order in which the data should be returned. 'asc' for oldest first, 'desc' for newest first"),
    time_interval: Optional[int] = Query(None, description="Time interval in minutes or hours"),
    time_unit: str = Query("minutes", description="Time unit, either 'minutes' or 'hours'")
):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        # Calculate the time threshold based on the current time and the specified interval
        if time_interval is not None:
            current_time = datetime.utcnow()
            if time_unit == "minutes":
                threshold_time = current_time - timedelta(minutes=time_interval)
            elif time_unit == "hours":
                threshold_time = current_time - timedelta(hours=time_interval)
            else:
                return response_handler.create_error_response_v1(
                    error_code=ErrorCode.INVALID_PARAMS,
                    error_string="Invalid time unit. Use 'minutes' or 'hours'.",
                    http_status_code=400)

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
            query["upload_date"] = {"$gte": validated_start_date, "$lte": validated_end_date}
        elif start_date:
            query["upload_date"] = {"$gte": validated_start_date}
        elif end_date:
            query["upload_date"] = {"$lte": validated_end_date}
        elif threshold_time_str:
            query["upload_date"] = {"$gte": threshold_time_str}

        # Add dataset filter if specified
        if dataset:
            query["dataset"] = dataset

        # Decide the sort order
        sort_order = -1 if order == "desc" else 1

        # Query the external_images_collection using the constructed query
        images_cursor = request.app.extracts_collection.find(query).sort("upload_date", sort_order).skip(offset).limit(limit)

        # Collect the metadata for the images that match the query
        images_metadata = []
        for image in images_cursor:
            image.pop('_id', None)  # Remove the auto-generated field
            images_metadata.append(image)

        return response_handler.create_success_response_v1(
            response_data={"images": images_metadata},
            http_status_code=200
        )
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )
    
@router.get("/extract-images/get-image-details-by-hash/{image_hash}", 
            response_model=StandardSuccessResponseV1[ExtractImageData],
            status_code=200,
            tags=["extracts"],
            description="Retrieves the details of an extract image by image hash. It returns the full data by default, but it can return only some properties by listing them using the 'fields' param",
            responses=ApiResponseHandlerV1.listErrors([404,422, 500]))
async def get_image_details_by_hash(request: Request, image_hash: str, fields: List[str] = Query(None)):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    
    # Create a projection for the MongoDB query
    projection = {field: 1 for field in fields} if fields else {}
    projection['_id'] = 0  # Exclude the _id field

    # Find the image by hash
    image_data = request.app.extracts_collection.find_one({"image_hash": image_hash}, projection)
    if image_data:
        return response_handler.create_success_response_v1(response_data=image_data, http_status_code=200)
    else:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.ELEMENT_NOT_FOUND, 
            error_string="Image not found",
            http_status_code=404
        )    

@router.get("/extract-images/get-image-details-by-hashes", 
            response_model=StandardSuccessResponseV1[ListExtractImageDataV1],
            status_code=200,
            tags=["extracts"],
            description="Retrieves the details of extract images by image hashes. It returns the full data by default, but it can return only some properties by listing them using the 'fields' param",
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def get_image_details_by_hashes(request: Request, image_hashes: List[str] = Query(...), fields: List[str] = Query(None)):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    
    # Create a projection for the MongoDB query
    projection = {field: 1 for field in fields} if fields else {}
    projection['_id'] = 0  # Exclude the _id field

    # Use the $in operator to find all matching documents in one query
    image_data_list = list(request.app.extracts_collection.find({"image_hash": {"$in": image_hashes}}, projection))

    # Return the data found in the success response
    return response_handler.create_success_response_v1(response_data={"images":image_data_list}, http_status_code=200)    

@router.get("/extract-images/get-random-images-with-clip-search",
            tags=["extracts"],
            description="Gets as many random extract images as set in the size param, scores each image with CLIP according to the value of the 'phrase' param and then returns the list sorted by the similarity score. NOTE: before using this endpoint, make sure to register the phrase using the '/clip/add-phrase' endpoint.",
            response_model=StandardSuccessResponseV1[ListExtractImageDataWithScore],
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
async def get_random_external_image_similarity(
    request: Request,
    phrase: str = Query(..., description="Phrase to compare similarity with"),
    dataset: Optional[str] = Query(None, description="Dataset to filter images"),
    similarity_threshold: float = Query(0, description="Minimum similarity threshold"),
    start_date: Optional[str] = Query(None, description="Start date for filtering results (YYYY-MM-DDTHH:MM:SS)"),
    end_date: Optional[str] = Query(None, description="End date for filtering results (YYYY-MM-DDTHH:MM:SS)"),
    size: int = Query(..., description="Number of random images to return")
):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        query = {}

        if dataset:
            query['dataset'] = dataset

        if start_date and end_date:
            query['upload_date'] = {'$gte': start_date, '$lte': end_date}
        elif start_date:
            query['upload_date'] = {'$gte': start_date}
        elif end_date:
            query['upload_date'] = {'$lte': end_date}

        aggregation_pipeline = [{"$match": query}]
        if size:
            aggregation_pipeline.append({"$sample": {"size": size}})

        images = list(request.app.extracts_collection.aggregate(aggregation_pipeline))

        image_path_list = []
        for image in images:
            image.pop('_id', None)  # Remove the auto-generated field
            bucket_name, file_path= separate_bucket_and_file_path(image['file_path'])
            image_path_list.append(file_path)

        similarity_score_list = http_clip_server_get_cosine_similarity_list("extracts" ,image_path_list, phrase)
        print(similarity_score_list)

        if similarity_score_list is None or 'similarity_list' not in similarity_score_list:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.OTHER_ERROR,
                error_string=str(e),
                http_status_code=500
            )

        similarity_score_list = similarity_score_list['similarity_list']

        if len(images) != len(similarity_score_list):
            return response_handler.create_success_response_v1(response_data={"images": []}, http_status_code=200)

        filtered_images = []
        for i in range(len(images)):
            image_similarity_score = similarity_score_list[i]
            image = images[i]

            if image_similarity_score >= similarity_threshold:
                image["similarity_score"] = image_similarity_score
                filtered_images.append(image)

        return response_handler.create_success_response_v1(response_data={"images": filtered_images}, http_status_code=200)

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )

@router.post("/extract-images/get-tag-list-for-multiple-extract-images", 
             response_model=StandardSuccessResponseV1[TagListForImagesV1], 
             description="changed with /tags/get-tag-list-for-multiple-images-v1",
             tags=["deprecated3"],
             status_code=200,
             responses=ApiResponseHandlerV1.listErrors([400, 404, 422, 500]))
async def get_tag_list_for_multiple_images(request: Request, file_hashes: List[str]):
    response_handler = ApiResponseHandlerV1(request)
    try:
        all_tags_list = []
        
        for file_hash in file_hashes:
            # Fetch image tags based on image_hash
            image_tags_cursor = request.app.image_tags_collection.find({"image_hash": file_hash, "image_source": extracts})
            
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

            all_tags_list.append({"file_hash": file_hash, "tags": tags_list})
        
        # Return the list of tag lists for each image
        return response_handler.create_success_response_v1(
            response_data={"images": all_tags_list},
            http_status_code=200,
        )
    except Exception as e:
        # Optional: Log the exception details here
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )        
@router.get("/extract-images/get_random_image_by_classifier_score", response_class=PrettyJSONResponse)
def get_random_image_date_range(
    request: Request,
    rank_id: int = None,
    start_date: str = None,
    end_date: str = None,
    min_score: float = 0.6,
    size: int = None,
):
    query = {}

    if start_date and end_date:
        query['task_creation_time'] = {'$gte': start_date, '$lte': end_date}
    elif start_date:
        query['task_creation_time'] = {'$gte': start_date}
    elif end_date:
        query['task_creation_time'] = {'$lte': end_date}

    # If rank_id is provided, adjust the query to consider classifier scores
    if rank_id is not None:
        # Get rank data
        rank = request.app.rank_model_models_collection.find_one({'rank_model_id': rank_id})
        if rank is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Rank model with this id doesn't exist")

        # Get the relevance classifier model id
        classifier_id = rank["classifier_id"]
        if classifier_id is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="This Rank has no relevance classifier model assigned to it")

        classifier_query = {'classifier_id': classifier_id}
        if min_score is not None:
            classifier_query['score'] = {'$gte': min_score}
            
        # Fetch image hashes from classifier_scores collection that match the criteria
        classifier_scores = request.app.image_classifier_scores_collection.find(classifier_query)
        if classifier_scores is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="The relevance classifier model has no scores.")
        image_hashes = [score['image_hash'] for score in classifier_scores]

        # Break down the image hashes into smaller batches
        BATCH_SIZE = 1000  # Adjust batch size as needed
        all_documents = []

        for i in range(0, len(image_hashes), BATCH_SIZE):
            batch_image_hashes = image_hashes[i:i+BATCH_SIZE]
            batch_query = query.copy()
            batch_query['task_output_file_dict.output_file_hash'] = {'$in': batch_image_hashes}

            aggregation_pipeline = [{"$match": batch_query}]
            if size:
                aggregation_pipeline.append({"$sample": {"size": size}})
            
            batch_documents = request.app.extracts_collection.aggregate(aggregation_pipeline)
            all_documents.extend(list(batch_documents))

        documents = all_documents
    else:
        aggregation_pipeline = [{"$match": query}]
        if size:
            aggregation_pipeline.append({"$sample": {"size": size}})

        documents = request.app.extracts_collection.aggregate(aggregation_pipeline)
        documents = list(documents)

    for document in documents:
        document.pop('_id', None)  # Remove the auto-generated field

    return documents
