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

@router.post(f'{api_prefix}/add-ingress-video',
             description='Add ingress video',
             tags=['ingress-videos'],
             response_model=StandardSuccessResponseV1[VideoMetaData],
             responses=ApiResponseHandlerV1.listErrors([422, 500]))
async def add_ingress_video(request: Request, video_meta_data: VideoMetaData):
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
    
@router.post(f'{api_prefix}/add-ingress-video-list',
             description='Add ingress videos',
             tags=['ingress-videos'],
             response_model=StandardSuccessResponseV1[ListVideoMetaData],
             responses=ApiResponseHandlerV1.listErrors([422, 500]))
async def add_ingress_video_list(request: Request, video_meta_data_list: List[VideoMetaData]):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        added_video_list = []
        for video_meta_data in video_meta_data_list:
            # check if video already exists
            existed = request.app.ingress_video_collection.find_one({
                'video_id': video_meta_data.video_id
            })
            
            if existed is not None:
                continue
            
            video_meta_data.upload_date = str(datetime.now())
            request.app.ingress_video_collection.insert_one(video_meta_data.to_dict())
            added_video_list.append(video_meta_data.to_dict())

        return api_response_handler.create_success_response_v1(
            response_data={'data': added_video_list},
            http_status_code=200
        )
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )


@router.get(f'{api_prefix}/get-ingress-video-by-video-id',
            description='Get ingress video by video id',
            tags=['ingress-videos'],
            response_model=StandardSuccessResponseV1[VideoMetaData],
            responses=ApiResponseHandlerV1.listErrors([404, 500]))
async def get_ingress_video_by_video_id(request: Request, video_id: str):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        video_data = request.app.ingress_video_collection.find_one({
            'video_id': video_id
        }, {'_id': 0})

        if video_data is None:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="Video data with this video id is not existed.",
                http_status_code=404
            )
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


@router.get(f'{api_prefix}/get-ingress-videos-by-video-id-list',
            description='Get ingress videos whose video id is in the video id list',
            tags=['ingress-videos'],
            response_model=StandardSuccessResponseV1[ListVideoMetaData],
            responses=ApiResponseHandlerV1.listErrors([422, 500]))
async def get_ingress_videos_by_video_id_list(request: Request, video_id_list: List[str]):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        pipeline = [
            {
                '$match': {
                    'video_id': {
                        '$in': video_id_list
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
    
@router.get(f'{api_prefix}/get-all-ingress-videos',
            description='Get all ingress videos',
            tags=['ingress-videos'],
            response_model=StandardSuccessResponseV1[ListVideoMetaData],
            responses=ApiResponseHandlerV1.listErrors([500]))
async def get_all_ingress_videos(request: Request):
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
            description='Get get all unprocessed ingress video',
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

@router.delete(f'{api_prefix}/delete-ingress-video-by-video-id',
            description='Delete ingress video with video id',
            tags=['ingress-videos'],
            response_model=StandardSuccessResponseV1[WasPresentResponse],
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def delete_video(request: Request, video_id: str):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        result = request.app.ingress_video_collection.delete_one({
            'video_id': video_id
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
    
@router.delete(f'{api_prefix}/delete-ingress-videos-by-video-id-list',
               description='Delete an Ingress video whose video id is in the list of video ids for deletion.',
               tags=['ingress-videos'],
               response_model=StandardSuccessResponseV1[DeletedCount],
               responses=ApiResponseHandlerV1.listErrors([422, 500]))
async def delete_ingress_videos_by_video_id_list(request: Request, video_id_list: List[str]):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        deleted_count = 0
        for video_id in video_id_list:
            result = request.app.ingress_video_collection.delete_one({
                'video_id': video_id
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
        

@router.put(f"{api_prefix}/update-ingress-video",
            description="Update a video game",
            tags=['ingress-videos'],
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
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string=f"Video game data with {video_metadata.video_id} not found",
                http_status_code=404
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