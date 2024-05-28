
from fastapi import APIRouter, Body, Request, HTTPException
from .api_utils import ApiResponseHandlerV1, StandardSuccessResponseV1, ErrorCode, WasPresentResponse, DeletedCount
from .mongo_schemas import ExternalImageData, ImageHashRequest, ListExternalImageData, ListImageHashRequest
from orchestration.api.mongo_schema.tag_schemas import ExternalImageTag, ListExternalImageTag
from typing import List
from datetime import datetime
from pymongo import UpdateOne
import uuid

router = APIRouter()

@router.post("/external-images/add-external-image", 
            description="Add an external image data",
            tags=["external-images"],  
            response_model=StandardSuccessResponseV1[ListExternalImageData],  
            responses=ApiResponseHandlerV1.listErrors([404,422, 500]))
async def add_external_image_data(request: Request, image_data: ExternalImageData):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:

        image_data.uuid = str(uuid.uuid4())

        existed = request.app.external_images_collection.find_one({
            "image_hash": image_data.image_hash
        })

        if existed is None:
            image_data.upload_date = str(datetime.now())
            request.app.external_images_collection.insert_one(image_data.to_dict())
        else:
            request.app.external_images_collection.update_one({
                "image_hash": image_data.image_hash
            }, {
                "$set": {
                    "upload_date": str(datetime.now()),
                    "dataset": image_data.dataset,
                    "image_resolution": image_data.image_resolution.to_dict(),
                    "image_format": image_data.image_format,
                    "file_path": image_data.file_path,
                    "source_image_dict": image_data.source_image_dict,
                    "task_attributes_dict": image_data.task_attributes_dict,
                    "uuid": image_data.uuid
                }
            })
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

@router.post("/external-images/add-external-image-list", 
            description="Add list of external image data",
            tags=["external-images"],  
            response_model=StandardSuccessResponseV1[ListExternalImageData],  
            responses=ApiResponseHandlerV1.listErrors([422, 500]))
async def add_external_image_data_list(request: Request, image_data_list: List[ExternalImageData]):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:

        image_data.uuid = str(uuid.uuid4())

        for image_data in image_data_list:
            
            existed = request.app.external_images_collection.find_one({
                "image_hash": image_data.image_hash
            })

            if existed is None:
                image_data.upload_date = str(datetime.now())
                request.app.external_images_collection.insert_one(image_data.to_dict())
            else:
                request.app.external_images_collection.update_one({
                    "image_hash": image_data.image_hash
                }, {
                    "$set": {
                        "upload_date": str(datetime.now()),
                        "dataset": image_data.dataset,
                        "image_resolution": image_data.image_resolution.to_dict(),
                        "image_format": image_data.image_format,
                        "file_path": image_data.file_path,
                        "source_image_dict": image_data.source_image_dict,
                        "task_attributes_dict": image_data.task_attributes_dict,
                        "uuid": image_data.uuid
                    }
                })

        return api_response_handler.create_success_response_v1(
            response_data={"data": [image_data.to_dict() for image_data in image_data_list]},
            http_status_code=200  
        )
    
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )
    
@router.get("/external-images/get-external-image-list", 
            description="get list of external image data by image hash list",
            tags=["external-images"],  
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
            response_model=StandardSuccessResponseV1[ListExternalImageData],  
            responses=ApiResponseHandlerV1.listErrors([404,422, 500]))
async def get_all_external_image_data_list(request: Request, 
                                           dataset: str=None,
                                           size: int=None):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        query={}
        if dataset:
            query['dataset']= dataset
        
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
            return api_response_handler.create_success_response_v1(
                response_data={"wasPresent": False}, 
                http_status_code=200
            )
        
        return api_response_handler.create_success_response_v1(
            response_data={"wasPresent": True}, 
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
              response_model=StandardSuccessResponseV1[ListExternalImageData],  
              responses=ApiResponseHandlerV1.listErrors([404,422, 500]))    
@router.patch("/external-images/add-task-attributes",
              description="Add or update the task attributes of an external image",
              tags=["external-images"],  
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

@router.get("/external-images/list-images",
            status_code=200,
            tags=["external-images"],  
            response_model=StandardSuccessResponseV1[List],  
            responses=ApiResponseHandlerV1.listErrors([404, 422]))
def get_external_images(request: Request):
    api_response_handler = ApiResponseHandlerV1(request)
    
    # Fetch all items from the collection, sorted by score in descending order
    items = list(request.app.external_images_collection.find())
    
    images = []
    for item in items:
        # Remove the auto-generated '_id' field
        item.pop('_id', None)
        images.append(item)
    
    # Return a standardized success response with the images data
    return api_response_handler.create_success_response_v1(
        response_data={'images': images},
        http_status_code=200
    )


@router.post("/external-images/add-tag-to-external-image",
             status_code=201,
             tags=["external-images"],  
             response_model=StandardSuccessResponseV1[ExternalImageTag], 
             responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def add_tag_to_image(request: Request, tag_id: int, image_hash: str, tag_type: int, user_who_created: str):
    response_handler = ApiResponseHandlerV1(request)
    try:
        date_now = datetime.now().isoformat()
    
        existing_tag = request.app.tag_definitions_collection.find_one({"tag_id": tag_id})
        if not existing_tag:
            return response_handler.create_error_response_v1(error_code=ErrorCode.ELEMENT_NOT_FOUND, error_string="Tag does not exist!", http_status_code=400)

        image = request.app.external_images_collection.find_one({'image_hash': image_hash})
        if not image:
            return response_handler.create_error_response_v1(error_code=ErrorCode.ELEMENT_NOT_FOUND, error_string="No image found with the given hash", http_status_code=400)

        file_path = image.get("file_path", "")
        
        # Check if the tag is already associated with the image
        existing_image_tag = request.app.image_tags_collection.find_one({"tag_id": tag_id, "image_hash": image_hash})
        if existing_image_tag:
            # Return an error response indicating that the tag has already been added to the image
            return response_handler.create_error_response_v1(error_code=ErrorCode.INVALID_PARAMS, error_string="This tag has already been added to the image", http_status_code=400)

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

        return response_handler.create_success_response_v1(response_data={"tag_id": tag_id, "file_path": file_path, "image_hash": image_hash, "tag_type": tag_type, "image_source": "external-image", "tag_count": 1, "user_who_created": user_who_created, "creation_time": date_now}, http_status_code=200)

    except Exception as e:
        return response_handler.create_error_response_v1(error_code=ErrorCode.OTHER_ERROR, error_string="Internal server error", http_status_code=500)


@router.delete("/external-images/remove-tag-from-external-image",
               status_code=200,
               tags=["external-images"],
               response_model=StandardSuccessResponseV1,
               responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def remove_tag_from_image(request: Request, tag_id: int, image_hash: str):
    response_handler = ApiResponseHandlerV1(request)
    try:
        # Check if the tag is associated with the image
        existing_image_tag = request.app.image_tags_collection.find_one({"tag_id": tag_id, "image_hash": image_hash})
        if not existing_image_tag:
            return response_handler.create_success_delete_response_v1(
            False,
            http_status_code=200
        )

        # Remove the tag
        request.app.image_tags_collection.delete_one({"tag_id": tag_id, "image_hash": image_hash})

        return response_handler.create_success_delete_response_v1(
            True,
            http_status_code=200
        )

    except Exception as e:
        return response_handler.create_error_response_v1(error_code=ErrorCode.OTHER_ERROR, error_string="Internal server error", http_status_code=500)


@router.get("/external-images/images-by-tag",
            status_code=200,
            tags=["external-images"],
            response_model=StandardSuccessResponseV1[ListExternalImageTag],
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def get_images_by_tag(request: Request, tag_id: int):
    response_handler = ApiResponseHandlerV1(request)
    try:
        images = list(request.app.image_tags_collection.find({"tag_id": tag_id}))
        
        if not images:
            return response_handler.create_error_response_v1(error_code=ErrorCode.ELEMENT_NOT_FOUND, error_string="No images found for the given tag", http_status_code=400)

        return response_handler.create_success_response_v1(response_data=images, http_status_code=200)

    except Exception as e:
        return response_handler.create_error_response_v1(error_code=ErrorCode.OTHER_ERROR, error_string="Internal server error", http_status_code=500)


@router.get("/external-images/tags-by-image",
            status_code=200,
            tags=["external-images"],
            response_model=StandardSuccessResponseV1[List[int]],
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def get_tags_by_image(request: Request, image_hash: str):
    response_handler = ApiResponseHandlerV1(request)
    try:
        tags = list(request.app.image_tags_collection.find({"image_hash": image_hash}, {"tag_id": 1, "_id": 0}))

        if not tags:
            return response_handler.create_error_response_v1(error_code=ErrorCode.ELEMENT_NOT_FOUND, error_string="No tags found for the given image", http_status_code=400)

        tag_ids = [tag["tag_id"] for tag in tags]

        return response_handler.create_success_response_v1(response_data=tag_ids, http_status_code=200)

    except Exception as e:
        return response_handler.create_error_response_v1(error_code=ErrorCode.OTHER_ERROR, error_string="Internal server error", http_status_code=500)


@router.get("/external-images/count-by-tag",
            status_code=200,
            tags=["external-images"],
            response_model=StandardSuccessResponseV1[int],
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def get_count_by_tag(request: Request, tag_id: int):
    response_handler = ApiResponseHandlerV1(request)
    try:
        count = request.app.image_tags_collection.count_documents({"tag_id": tag_id})

        if count == 0:
            return response_handler.create_error_response_v1(error_code=ErrorCode.ELEMENT_NOT_FOUND, error_string="No images found for the given tag", http_status_code=400)

        return response_handler.create_success_response_v1(response_data=count, http_status_code=200)

    except Exception as e:
        return response_handler.create_error_response_v1(error_code=ErrorCode.OTHER_ERROR, error_string="Internal server error", http_status_code=500)
