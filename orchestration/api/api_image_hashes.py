import bson.int64
from fastapi import APIRouter, Request, Response
from .api_utils import PrettyJSONResponse, validate_date_format, ApiResponseHandler, ErrorCode, StandardSuccessResponseV1, ApiResponseHandlerV1, WasPresentResponse
from .mongo_schemas import ImageHashRequest, ImageHash, ListImageHash, GlobalId
from decimal import Decimal
from typing import List
from bson.decimal128 import Decimal128
router = APIRouter()

next_image_global_id = 0

@router.post("/image-hashes/add-image-hash-v1",
             tags=["image-hashes"], 
             description= "add image hash",
             response_model=StandardSuccessResponseV1[ImageHash],
             responses=ApiResponseHandlerV1.listErrors([422,500]))
@router.post("/image-hashes/add_image_hash",
             tags=["deprecated3"], 
             description= "changed with /image-hashes/add-image-hash-v1 ",
             response_model=StandardSuccessResponseV1[ImageHash],
             responses=ApiResponseHandlerV1.listErrors([422,500]))
async def add_image_hash(request: Request, image_hash_request: ImageHashRequest):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    
    try:
        image_hash = ImageHash(
            image_global_id=request.app.max_image_global_id + 1,
            image_hash=image_hash_request.image_hash)
        
        existed_image_hash = request.app.image_hashes_collection.find_one({"image_hash": image_hash_request.image_hash})
        
        if existed_image_hash is None:
            request.app.image_hashes_collection.insert_one(image_hash.to_dict_for_mongodb())
            request.app.max_image_global_id += 1
        else:
            image_hash.image_global_id = existed_image_hash["image_global_id"]

        return response_handler.create_success_response_v1(
            response_data=image_hash.to_dict(),
            http_status_code=200
        )
    
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=f"Failed to add image hash: {str(e)}",
            http_status_code=500
        )
    

@router.get("/image-hashes/update_all_image_hashes",
             tags=["image-hashes"], 
             description="Updates the image_hashes_collection collection so that it includes all the hashes from the completed_jobs_collection collection",
             response_model=StandardSuccessResponseV1[List[ImageHash]],
             responses=ApiResponseHandlerV1.listErrors([500]))
async def update_all_image_hashes(request: Request):

    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        # fetch all image hash and add the image hash with image_global_id
        completed_jobs = list(request.app.completed_jobs_collection.find(
            {}, 
            {"task_output_file_dict.output_file_hash": 1, "task_type": 1}))
        for job_data in completed_jobs:
            
            if not job_data or 'task_output_file_dict' not in job_data or 'output_file_hash' not in job_data['task_output_file_dict']:
                continue

            await add_image_hash(request, ImageHashRequest(image_hash=job_data["task_output_file_dict"]['output_file_hash']))

        # get image hashes with image_global_id
        image_hashes = list(request.app.image_hashes_collection.find(
            {},
            {"_id": 0}
        ))

        image_hashes = [ImageHash(**doc).to_dict() for doc in image_hashes]

        # Return the fetched data with a success response
        return response_handler.create_success_response_v1(
            response_data=image_hashes, 
            http_status_code=200
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=f"Failed to adding all image hases of completed jobs with unique global id: {str(e)}",
            http_status_code=500
        )
    

@router.put("/image-hashes/update-all-image-hashes-v1",
             tags=["image-hashes"], 
             description="Updates the image_hashes_collection collection so that it includes all the hashes from the completed_jobs_collection collection",
             response_model=StandardSuccessResponseV1[ListImageHash],
             responses=ApiResponseHandlerV1.listErrors([500]))
async def update_all_image_hashes_v1(request: Request):

    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        # fetch all image hash and add the image hash with image_global_id
        completed_jobs = list(request.app.completed_jobs_collection.find(
            {}, 
            {"task_output_file_dict.output_file_hash": 1, "task_type": 1}))
        for job_data in completed_jobs:
            
            if not job_data or 'task_output_file_dict' not in job_data or 'output_file_hash' not in job_data['task_output_file_dict']:
                continue

            await add_image_hash(request, ImageHashRequest(image_hash=job_data["task_output_file_dict"]['output_file_hash']))

        # get image hashes with image_global_id
        image_hashes = list(request.app.image_hashes_collection.find(
            {},
            {"_id": 0}
        ))

        image_hashes = [ImageHash(**doc).to_dict() for doc in image_hashes]

        # Return the fetched data with a success response
        return response_handler.create_success_response_v1(
            response_data={"data": image_hashes}, 
            http_status_code=200
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=f"Failed to adding all image hases of completed jobs with unique global id: {str(e)}",
            http_status_code=500
        )    

@router.get("/image-hashes/get-all-image-hashes",
            tags=["image-hashes"],
            description="get all image hash tags",
            response_model=StandardSuccessResponseV1[ImageHash],
            responses=ApiResponseHandlerV1.listErrors([500]))
async def get_all_image_hashes_with_global_id(request: Request):

    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        image_hashes_data = list(request.app.image_hashes_collection.find(
            {}, 
            { "_id": 0 }
        ))

        image_hashes_data = [ImageHash(**doc).to_dict() for doc in image_hashes_data]        

        # Return the fetched data with a success response
        return response_handler.create_success_response_v1(
            response_data=image_hashes_data, 
            http_status_code=200
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=f"Failed to get all image hases with global id: {str(e)}",
            http_status_code=500
        )
    
@router.get("/image-hashes/get-all-image-hashes-v1",
            tags=["image-hashes"],
            description="Gets all image hashes",
            response_model=StandardSuccessResponseV1[ListImageHash],
            responses=ApiResponseHandlerV1.listErrors([500]))
async def get_all_image_hashes_with_global_id_v1(request: Request):

    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        image_hashes_data = list(request.app.image_hashes_collection.find(
            {}, 
            { "_id": 0 }
        ))

        image_hashes_data = [ImageHash(**doc).to_dict() for doc in image_hashes_data]        

        # Return the fetched data with a success response
        return response_handler.create_success_response_v1(
            response_data={"data": image_hashes_data}, 
            http_status_code=200
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=f"Failed to get all image hases with global id: {str(e)}",
            http_status_code=500
        )    

    
@router.get("/image-hashes/get-image-hash-by-global-id",
            tags=["image-hashes"],
            description="get image hash with image global id which is unique and type is int64",
            response_model=StandardSuccessResponseV1[str],
            responses=ApiResponseHandlerV1.listErrors([422,500]))
async def get_image_hash_by_global_id(request: Request, image_global_id: int):

    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        
        image_hash_data = request.app.image_hashes_collection.find_one({
            "image_global_id": image_global_id
        }, {
            "image_hash": 1
        })

        # Return the fetched data with a success response
        return response_handler.create_success_response_v1(
            response_data=image_hash_data["image_hash"], 
            http_status_code=200
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=f"Failed to get image hash by global id: {str(e)}",
            http_status_code=500
        )
    
@router.get("/image-hashes/get-image-hash-by-global-id-v1",
            tags=["image-hashes"],
            description="get image hash with image global id which is unique and type is int64",
            response_model=StandardSuccessResponseV1[ImageHashRequest],
            responses=ApiResponseHandlerV1.listErrors([422,500]))
async def get_image_hash_by_global_id_v1(request: Request, image_global_id: int):

    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        
        image_hash_data = request.app.image_hashes_collection.find_one({
            "image_global_id": image_global_id
        }, {
            "image_hash": 1
        })

        # Return the fetched data with a success response
        return response_handler.create_success_response_v1(
            response_data={"image_hash": image_hash_data["image_hash"]}, 
            http_status_code=200
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=f"Failed to get image hash by global id: {str(e)}",
            http_status_code=500
        )    

@router.get("/image-hashes/get-image-global-id-by-image-hash",
            tags = ["image-hashes"],
            description="get image global id with image hash  which is unique and type is int64",
            response_model=StandardSuccessResponseV1[str],
            responses=ApiResponseHandlerV1.listErrors([422,500]))
async def get_image_hash_by_global_id(request: Request, image_hash: str):

    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        
        image_hash_data = request.app.image_hashes_collection.find_one({
            "image_hash": image_hash
        }, {
            "image_global_id": 1
        })

        # Return the fetched data with a success response
        return response_handler.create_success_response_v1(
            response_data=image_hash_data["image_global_id"], 
            http_status_code=200
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=f"Failed to get image global id by image hash: {str(e)}",
            http_status_code=500
        )
    

@router.get("/image-hashes/get-image-global-id-by-image-hash-v1",
            tags = ["image-hashes"],
            description="get image global id with image hash  which is unique and type is int64",
            response_model=StandardSuccessResponseV1[GlobalId],
            responses=ApiResponseHandlerV1.listErrors([422,500]))
async def get_image_hash_by_global_id_v1(request: Request, image_hash: str):

    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        
        image_hash_data = request.app.image_hashes_collection.find_one({
            "image_hash": image_hash
        }, {
            "image_global_id": 1
        })

        # Return the fetched data with a success response
        return response_handler.create_success_response_v1(
            response_data={"image_global_id": image_hash_data["image_global_id"]}, 
            http_status_code=200
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=f"Failed to get image global id by image hash: {str(e)}",
            http_status_code=500
        )    
