from fastapi import APIRouter, Request
from .api_utils import ApiResponseHandlerV1, \
    StandardSuccessResponseV1, ErrorCode, WasPresentResponse, DeletedCount

# import schema
from .mongo_schemas import VideoMetaData, ListVideoMetaData

# import time library
from datetime import datetime

# import typing
from typing import List
from .api_utils import get_next_seq_id, update_seq_id, get_minio_file_path

router = APIRouter()

api_prefix = '/ingress-video'

@router.post(f'{api_prefix}/add-video',
             description='Add video for extracting external images',
             response_model=StandardSuccessResponseV1[VideoMetaData],
             responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def add_video(request: Request, video_meta_data: VideoMetaData):
    # TODO: add minio path into video meta data:
    # for this, need to add dataset name
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)


    
    try:
        # check if video already exists
        video_file_hash = video_meta_data.file_hash
        existed = request.app.ingress_video_collection.find_one({
            'file_hash': video_file_hash
        })

        video_meta_data.upload_date = str(datetime.now())
        if existed is None:
            # TODO: add dataset so get sequential id for specific dataset
            next_seq_id = get_next_seq_id(request, bucket="ingress-video", dataset=video_meta_data.dataset)
            video_meta_data.file_path = get_minio_file_path(next_seq_id,
                                                            video_meta_data.dataset, 
                                                            video_meta_data.file_type)

            request.app.ingress_video_collection.insert_one(video_meta_data.to_dict())
            update_seq_id(request=request, bucket="ingress-video", dataset=video_meta_data.dataset, seq_id=next_seq_id)

        else:
            video_meta_data.file_path = existed['file_path']
            request.app.ingress_video_collection.update_one({
                'file_hash': video_file_hash
            }, {
                '$set': video_meta_data.to_dict()
            })


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
    
@router.post(f'{api_prefix}/add-video-list',
             description='Add video for extracting external image list',
             response_model=StandardSuccessResponseV1[ListVideoMetaData],
             responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def add_video_list(request: Request, video_meta_data_list: List[VideoMetaData]):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:

        for video_meta_data in video_meta_data_list:
            # add minio video path
            
            # check if video already exists
            video_file_hash = video_meta_data.file_hash
            existed = request.app.ingress_video_collection.find_one({
                'file_hash': video_file_hash
            })

            video_meta_data.upload_date = str(datetime.now())
            if existed is None:
                next_seq_id = get_next_seq_id(request, bucket="ingress-video", dataset=video_meta_data.dataset)
                video_meta_data.file_path = get_minio_file_path(next_seq_id, 
                                                        video_meta_data.dataset, 
                                                        video_meta_data.file_type)
                request.app.ingress_video_collection.insert_one(video_meta_data.to_dict())
                update_seq_id(request=request, bucket="ingress-video", dataset=video_meta_data.dataset, seq_id=next_seq_id)

            else:
                video_meta_data.file_path = existed['file_path']
                request.app.ingress_video_collection.update_one({
                    'file_hash': video_file_hash
                }, {
                    '$set': video_meta_data.to_dict()
                })

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


@router.get(f'{api_prefix}/get-video',
            description='Get video data by video hash',
            response_model=StandardSuccessResponseV1[VideoMetaData],
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def get_video_list(request: Request, video_hash: str):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        video_data = request.app.ingress_video_collection.find_one({
            'file_hash': video_hash
        })

        video_data.pop('_id')
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


@router.get(f'{api_prefix}/get-video-list',
            description='Get list of video by video hash list',
            response_model=StandardSuccessResponseV1[ListVideoMetaData],
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def get_video_list(request: Request, video_hash_list: List[str]):
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
    
@router.get(f'{api_prefix}/get-all-video',
            description='Get list of video by video hash list',
            response_model=StandardSuccessResponseV1[ListVideoMetaData],
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def get_video_list(request: Request):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        pipeline = [
            {
                '$match': {}
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

@router.delete(f'{api_prefix}/delete-video',
               description='Delete ingress video',
               response_model=StandardSuccessResponseV1[WasPresentResponse],
               responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def delete_video(request: Request, video_hash: str):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        result = request.app.ingress_video_collection.delete_one({
            'file_hash': video_hash
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
    
@router.delete(f'{api_prefix}/delete-video-list',
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