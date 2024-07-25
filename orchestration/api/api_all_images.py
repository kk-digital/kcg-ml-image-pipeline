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
import time

def datetime_to_unix_int32(dt_str):
    formats = ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f"]
    for fmt in formats:
        try:
            dt = datetime.datetime.strptime(dt_str, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"time data '{dt_str}' does not match any known format")
    
    unix_time = int(time.mktime(dt.timetuple()))
    return unix_time & 0xFFFFFFFF

router = APIRouter()

@router.get("/all-images/list",
            description="list images according dataset_id and bucket_id",
            tags=["all-images"],
            response_model=StandardSuccessResponseV1)
async def list_all_images(
    request: Request,
    bucket_ids: Optional[List[int]] = Query(None, description="Bucket IDs"),
    dataset_ids: Optional[List[int]] = Query(None, description="Dataset IDs"),
    limit: int = Query(20, description="Limit on the number of results returned"),
    offset: int = Query(0, description="Offset for the results to be returned"),
    order: str = Query("desc", description="Order in which the data should be returned. 'asc' for oldest first, 'desc' for newest first"),
    start_date: Optional[str] = Query(None, description="Start date for filtering results"),
    end_date: Optional[str] = Query(None, description="End date for filtering results"),
    time_interval: Optional[int] = Query(None, description="Time interval in minutes or hours"),
    time_unit: str = Query("minutes", description="Time unit, either 'minutes' or 'hours")
):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        query = {}

        # Add the OR conditions for buckets and datasets
        if bucket_ids or dataset_ids:
            query_conditions = []
            if bucket_ids:
                query_conditions.append({"bucket_id": {"$in": bucket_ids}})
            if dataset_ids:
                query_conditions.append({"dataset_id": {"$in": dataset_ids}})
            if query_conditions:
                query = {"$or": query_conditions}

        # Add date filters to the query
        date_query = {}
        if start_date:
            date_query['$gte'] = datetime_to_unix_int32(start_date)
        if end_date:
            date_query['$lte'] = datetime_to_unix_int32(end_date)

        # Calculate the time threshold based on the current time and the specified interval
        if time_interval is not None:
            current_time = datetime.utcnow()
            if time_unit == "minutes":
                threshold_time = current_time - timedelta(minutes=time_interval)
            elif time_unit == "hours":
                threshold_time = current_time - timedelta(hours=time_interval)
            else:
                raise HTTPException(status_code=400, detail="Invalid time unit. Use 'minutes' or 'hours'.")
            date_query['$gte'] = datetime_to_unix_int32(threshold_time.isoformat(timespec='milliseconds'))
        
        if date_query:
            query['date'] = date_query

        # Decide the sort order based on the 'order' parameter
        sort_order = -1 if order == "desc" else 1

        # Query the collection with pagination and sorting
        cursor = request.app.all_image_collection.find(query).sort('date', sort_order).skip(offset).limit(limit)
        images = list(cursor)

        for image in images:
            image.pop("_id", None)  # Remove the MongoDB ObjectId

        return response_handler.create_success_response_v1(
            response_data=images,
            http_status_code=200
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )