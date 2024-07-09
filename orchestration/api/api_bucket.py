from datetime import datetime
from fastapi import APIRouter, Request, HTTPException, Query
from typing import List, Dict
from .mongo_schemas import Classifier
from typing import Union
from orchestration.api.mongo_schema.bucket_schemas import Bucket, ResponseBucket, ListResponseBucket
from .api_utils import validate_date_format, ErrorCode, WasPresentResponse, StandardSuccessResponseV1, ApiResponseHandlerV1
from bson import ObjectId

router = APIRouter()


@router.post("/buckets/add-new-bucket",
          description="Add a new bucket in MongoDB",
          tags=["buckets"], 
          response_model=StandardSuccessResponseV1[ResponseBucket],  
          responses=ApiResponseHandlerV1.listErrors([400,422]))
async def add_new_bucket(request: Request, bucket: Bucket):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    if request.app.buckets_collection.find_one({"bucket_name": bucket.bucket_name}):
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.INVALID_PARAMS,
            error_string='Bucket already exists',
            http_status_code=400
        )    

    # Get the next bucket_id
    last_bucket = request.app.buckets_collection.find_one(sort=[("bucket_id", -1)])
    next_bucket_id = (last_bucket["bucket_id"] if last_bucket else 0) + 1

    # Create the new bucket document
    new_bucket = {
        "bucket_id": next_bucket_id,
        "bucket_name": bucket.bucket_name,
        "creation_time": datetime.utcnow().isoformat()
    }

    # Insert the new bucket document into the collection
    request.app.buckets_collection.insert_one(new_bucket)
    
    # Remove '_id' from the response data
    new_bucket.pop('_id', None)

    return response_handler.create_success_response_v1(
        response_data=new_bucket,
        http_status_code=200
    )

@router.get("/buckets/list-buckets", 
         status_code=200, 
         description="List all buckets",
         tags=["buckets"],
         response_model=StandardSuccessResponseV1[ListResponseBucket])
async def list_buckets(request: Request):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    # Retrieve all buckets
    buckets = list(request.app.buckets_collection.find({}, {"_id": 0}))

    return response_handler.create_success_response_v1(
        response_data={"buckets": buckets},
        http_status_code=200
    )


@router.delete("/buckets/remove-bucket",
               description="Remove bucket in MongoDB",
               tags=["buckets"],
               response_model=StandardSuccessResponseV1[WasPresentResponse],  
               responses=ApiResponseHandlerV1.listErrors([422]))
async def remove_bucket(request: Request, bucket_name: str = Query(...)):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    # Attempt to delete the bucket
    bucket_result = request.app.buckets_collection.delete_one({"bucket_name": bucket_name})

    # Check if the bucket was present and deleted
    was_present = bucket_result.deleted_count > 0

    # Using the check to determine which response to send
    if was_present:
        # If the bucket was deleted, return True
        return response_handler.create_success_delete_response_v1(
            True, 
            http_status_code=200
        )
    else:
        # If the bucket was not deleted, return False
        return response_handler.create_success_delete_response_v1(
            False, 
            http_status_code=200
        )
