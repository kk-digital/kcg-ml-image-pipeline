from fastapi import Request, HTTPException, APIRouter, Response, Query, status, File, UploadFile
from datetime import datetime, timedelta
from typing import Optional
import pymongo
from utility.minio import cmd
from utility.path import separate_bucket_and_file_path
from .mongo_schemas import Task, ImageMetadata, UUIDImageMetadata, ListTask
from .api_utils import PrettyJSONResponse, StandardSuccessResponseV1, ApiResponseHandlerV1, UrlResponse, ErrorCode
from .api_ranking import get_image_rank_use_count
import os
from .api_utils import find_or_create_next_folder_and_index
from orchestration.api.mongo_schema.all_images_schemas import AllImagesResponse, ListAllImagesResponse
import io
from typing import List
from PIL import Image


router = APIRouter()

@router.get("/all-images/list", 
            description= "list images according dataset_id and bucket_id",
            tags=["all-images"],
            response_model=StandardSuccessResponseV1[ListAllImagesResponse])
async def list_all_images(
    request: Request,
    bucket_id: int = Query(..., description="Bucket ID"),
    dataset_id: int = Query(..., description="Dataset ID"),
    size: Optional[int] = Query(None, description="Number of results to return")
):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        query = {
            "bucket_id": bucket_id,
            "dataset_id": dataset_id
        }

        cursor = request.app.all_image_collection.find(query).limit(size if size else 0)
        images = list(cursor)

        for image in images:
            image.pop("_id", None)  # Remove the MongoDB ObjectId

        return response_handler.create_success_response_v1(
            response_data==images,
            http_status_code=200
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )