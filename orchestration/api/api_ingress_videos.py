from fastapi import APIRouter, Request
from .api_utils import ApiResponseHandlerV1, \
    StandardSuccessResponseV1, ErrorCode, WasPresentResponse, DeletedCount

# import schema
from .mongo_schemas import VideoMetaData, ListVideoMetaData

# import time library
from datetime import datetime

# import typing
from typing import List

router = APIRouter()

api_prefix = '/ingress-videos'

@router.post(f'{api_prefix}/add',
            description='Add ingress video',
            response_model=StandardSuccessResponseV1[VideoMetaData],
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def add_video(request: Request, video_meta_data: VideoMetaData):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        # check if video already exists
        video_id = video_meta_data.video_id
        existed = request.app.ingress_video_collection.find_one({
            'video_id': video_id
        })

        if existed is None:
            video_meta_data.upload_date = str(datetime.now())
            request.app.ingress_video_collection.insert_one(video_meta_data.to_dict())
        else:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string="Video data with this video-id already exists.",
                http_status_code=422
            )

        return api_response_handler.create_success_response_v1(
            response_data=video_meta_data.to_dict(),
            http_status_code=200
        )
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )
    
@router.post(f'{api_prefix}/add-list',
            description='Add video list',
            response_model=StandardSuccessResponseV1[ListVideoMetaData],
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def add_video_list(request: Request, video_meta_data_list: List[VideoMetaData]):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        for video_meta_data in video_meta_data_list:
            # check if video already exists
            video_file_hash = video_meta_data.file_hash
            existed = request.app.ingress_video_collection.find_one({
                'video_id': video_meta_data.video_id
            })
            
            if existed is not None:
                continue
                
            video_meta_data.upload_date = str(datetime.now())
            request.app.ingress_video_collection.insert_one(video_meta_data.to_dict())

        return api_response_handler.create_success_response_v1(
            response_data={'data': [video_meta_data.to_dict() \
                                    for video_meta_data in video_meta_data_list]},
            http_status_code=200
        )
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )


@router.get(f'{api_prefix}/get',
            description='Get video data by video hash',
            response_model=StandardSuccessResponseV1[VideoMetaData],
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def get_video_by_hash(request: Request, video_hash: str):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        video_data = dict(request.app.ingress_video_collection.find_one({
            'video_id': video_hash
        }, {'_id': 0}))
        
        video_data = dict(video_data)

        return api_response_handler.create_success_response_v1(
            response_data=video_data,
            http_status_code=200
        )
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )


@router.get(f'{api_prefix}/get-list',
            description='Get list of video by video hash list',
            response_model=StandardSuccessResponseV1[ListVideoMetaData],
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def get_video_list_by_hash_list(request: Request, video_hash_list: List[str]):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        pipeline = [
            {
                '$match': {
                    'file_hash': {
                        '$in': video_hash_list
                    }
                }
            },
            {
                '$project': {
                    '_id': 0
                }
            }
        ]
        video_list = list(request.app.ingress_video_collection.aggregate(pipeline))

        return api_response_handler.create_success_response_v1(
            response_data={'data': video_list},
            http_status_code=200
        )
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )
    
@router.get(f'{api_prefix}/list',
            description='Get get all video data',
            response_model=StandardSuccessResponseV1[ListVideoMetaData],
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def get_video_list(request: Request):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        video_list = list(request.app.ingress_video_collection.find({}, {'_id': 0}))

        return api_response_handler.create_success_response_v1(
            response_data={'data': video_list},
            http_status_code=200
        )
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )
        
@router.get(f'{api_prefix}/unprocessed-list',
            description='Get get all video data',
            response_model=StandardSuccessResponseV1[ListVideoMetaData],
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def get_unprocessed_video_list(request: Request):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        video_list = list(request.app.ingress_video_collection.find({"processed": False}, {'_id': 0}))

        return api_response_handler.create_success_response_v1(
            response_data={'data': video_list},
            http_status_code=200
        )
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )

@router.delete(f'{api_prefix}/delete',
            description='Delete ingress video with video hash',
            response_model=StandardSuccessResponseV1[WasPresentResponse],
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def delete_video(request: Request, video_hash: str):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        result = request.app.ingress_video_collection.delete_one({
            'video_id': video_hash
        })

        if result.deleted_count == 0:
            return api_response_handler.create_success_delete_response_v1(
                wasPresent=False,
                http_status_code=404
            )
        
        return api_response_handler.create_success_delete_response_v1(
            wasPresent=True,
            http_status_code=200
        )
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )
    
@router.delete(f'{api_prefix}/delete-list',
               description='Delete ingress video list',
               response_model=StandardSuccessResponseV1[DeletedCount],
               responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def delete_video_list(request: Request, video_hash_list: List[str]):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        deleted_count = 0
        for video_hash in video_hash_list:
            result = request.app.ingress_video_collection.delete_one({
                'file_hash': video_hash
            })

            if result.deleted_count != 0:
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
        

@router.put("/ingress-videos/update",
            description="Update a video game",
            tags=["video-game", "video", "game"],
            response_model=StandardSuccessResponseV1[VideoMetaData],
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def update(request: Request, video_metadata: VideoMetaData):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    
    try:
        existed = request.app.ingress_video_collection.find_one(
            {"video_id": video_metadata.video_id}
        )
        
        if not existed:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string=f"Video game data with {video_metadata.video_id} not found",
                http_status_code=422
            )
        
        request.app.ingress_video_collection.update_one(
            {"video_id": video_metadata.video_id}, 
            {"$set": video_metadata.to_dict()}
        )
        
        return api_response_handler.create_success_response_v1(
            response_data=video_metadata.to_dict(),
            http_status_code=200
        )
        
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )