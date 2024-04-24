
from fastapi import APIRouter, Request
from .api_utils import ApiResponseHandlerV1, StandardSuccessResponseV1, ErrorCode
from .mongo_schemas import ExternalImageData, ImageHashRequest
from typing import List
from datetime import datetime
router = APIRouter()

@router.post("/external-images/add-external-image", 
            description="Add an external image data",
            tags=["external-images"],  
            response_model=StandardSuccessResponseV1[ExternalImageData],  
            responses=ApiResponseHandlerV1.listErrors([404, 500]))
async def add_external_image_data(request: Request, image_data: ExternalImageData):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:

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
                    "task_attributes_dict": image_data.task_attributes_dict
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
            response_model=StandardSuccessResponseV1[List[ExternalImageData]],  
            responses=ApiResponseHandlerV1.listErrors([500]))
async def add_external_image_data_list(request: Request, image_data_list: List[ExternalImageData]):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
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
                        "task_attributes_dict": image_data.task_attributes_dict
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
            response_model=StandardSuccessResponseV1[List[ExternalImageData]],  
            responses=ApiResponseHandlerV1.listErrors([404, 500]))
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
    
@router.get("/external-images/get-all-external-image-list", 
            description="get all external image data",
            tags=["external-images"],  
            response_model=StandardSuccessResponseV1[List[ExternalImageData]],  
            responses=ApiResponseHandlerV1.listErrors([404, 500]))
async def get_all_external_image_data_list(request: Request):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:

        all_image_data_list = list(request.app.external_images_collection.find_one({}, {"_id": 0}))
    
        list_external_images = []
        for image_data in all_image_data_list:
            list_external_images.append(image_data)

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


@router.delete("/external-images/delete-external-image", 
            description="Delete an external image data",
            tags=["external-images"],  
            response_model=StandardSuccessResponseV1[int],  
            responses=ApiResponseHandlerV1.listErrors([404, 500]))
async def delete_external_image_data(request: Request, image_hash:str ):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        was_present = False
        result = request.app.external_images_collection.delete_one({
            "image_hash": image_hash
        })
        
        if result.deleted_count == 0:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS, 
                error_string="There is no external image data with image hash: {}".format(image_hash), 
                http_status_code=400)
        else:
            was_present = True
        
        return api_response_handler.create_success_delete_response_v1(
            was_present,
            200  
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
            response_model=StandardSuccessResponseV1[int],  
            responses=ApiResponseHandlerV1.listErrors([404, 500]))
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