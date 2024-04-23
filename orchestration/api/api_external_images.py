
from fastapi import APIRouter, Request
from .api_utils import ApiResponseHandlerV1, StandardSuccessResponseV1, ErrorCode
from .mongo_schemas import ExternalImageData
from typing import List

router = APIRouter()

@router.post("/external-images/add-external-image", 
            description="Add an external image data",
            tags=["external-images"],  
            response_model=StandardSuccessResponseV1[ExternalImageData],  
            responses=ApiResponseHandlerV1.listErrors([404, 500]))
async def add_external_image_data(request: Request, image_data: ExternalImageData):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        image_hash = image_data.source_image_dict.get("image_hash")

        if image_hash is not None:
            existed = request.app.external_images_collection.find_one({
                "source_image_dict.image_hash": image_hash
            })

            if existed is None:
                request.app.external_images_collection.insert_one(image_data.to_dict())
            else:
                request.app.external_images_collection.update_one({
                    "source_image_dict.image_hash": image_hash
                }, {
                    "$set": {
                        "source_image_dict": image_data.source_image_dict,
                        "task_attributes_dict": image_data.task_attributes_dict
                    }
                })
            return api_response_handler.create_success_response_v1(
                response_data={"data": image_data.to_dict()},
                http_status_code=200  
            )
        else:
            return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS, 
            error_string="There is no image hash",
            http_status_code=500
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
            responses=ApiResponseHandlerV1.listErrors([404, 500]))
async def add_external_image_data_list(request: Request, image_data_list: List[ExternalImageData]):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    print(1)
    try:
        for image_data in image_data_list:
            print(2)
            image_hash = image_data.source_image_dict.get("image_hash")
            print(3)
            if image_hash is not None:
                existed = request.app.external_images_collection.find_one({
                    "source_image_dict.image_hash": image_hash
                })
                if existed is None:
                    request.app.external_images_collection.insert_one(image_data.to_dict())
                else:
                    request.app.external_images_collection.update_one({
                        "source_image_dict.image_hash": image_hash
                    }, {
                        "$set": {
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
    
@router.delete("/external-images/delete-external-image", 
            description="Delete an external image data",
            tags=["external-images"],  
            response_model=StandardSuccessResponseV1[int],  
            responses=ApiResponseHandlerV1.listErrors([404, 500]))
async def delete_external_image_data(request: Request, image_data: ExternalImageData):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        deleted = False
        image_hash = image_data.source_image_dict.get("image_hash")
        if image_hash is not None:
            request.app.external_images_collection.delete_one({
                "source_image_dict.image_hash": image_hash
            })
            deleted = True

        return api_response_handler.create_success_response_v1(
            response_data={"deleted": deleted},
            http_status_code=200  
        )
    
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )


@router.delete("/external-images/delete-external-image-list-by-image-hash-list", 
            description="Delete a list of external image data",
            tags=["external-images"],  
            response_model=StandardSuccessResponseV1[int],  
            responses=ApiResponseHandlerV1.listErrors([404, 500]))
async def delete_external_image_data_list(request: Request, image_data_list: List[ExternalImageData]):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        deleted_count = 0
        for image_data in image_data_list:
            image_hash = image_data.source_image_dict.get("image_hash")
            if image_hash is not None:
                request.app.external_images_collection.delete_one({
                    "source_image_dict.image_hash": image_hash
                })
                deleted_count += 1
        return api_response_handler.create_success_response_v1(
            response_data={"deleted_count": deleted_count},
            http_status_code=200
        )
    
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )



@router.delete("/external-images/delete-external-image-by-image-hash", 
            description="Delete an external image data",
            tags=["external-images"],  
            response_model=StandardSuccessResponseV1[int],  
            responses=ApiResponseHandlerV1.listErrors([404, 500]))
async def delete_external_image_data_by_image_hash(request: Request, image_hash: str):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        deleted = False
        request.app.external_images_collection.delete_one({
            "source_image_dict.image_hash": image_hash
        })
        deleted = True

        return api_response_handler.create_success_response_v1(
            response_data={"deleted": deleted},
            http_status_code=200  
        )
    
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )