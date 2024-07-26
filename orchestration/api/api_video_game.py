from fastapi import APIRouter, Body, Request, HTTPException, Query, status
from typing import Optional
from .api_utils import ApiResponseHandlerV1, StandardSuccessResponseV1, ErrorCode
from .mongo_schemas import VideoGame, ListVideoGame
router = APIRouter()
video_game = "video-game"

@router.get("/video-games/list",
            description="List all the games",
            tags=["video-game", "video", "game"],
            response_model=StandardSuccessResponseV1[ListVideoGame],
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def list_games(request: Request):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    
    try:
        video_games = list(request.app.video_game_collection.find({}, {"_id": 0}))
        print(video_games)
        return api_response_handler.create_success_response_v1(
            response_data={ "games": video_games },
            http_status_code=200
        )
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )

@router.get("/video-games/get",
            description="Get a video game with game_id",
            tags=["video-game", "video", "game"],
            response_model=StandardSuccessResponseV1[VideoGame],
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def add_game(request: Request, game_id: str):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    
    try:
        video_game = request.app.video_game_collection.find_one({"game_id": game_id}, {"_id": 0})
        
        if video_game:
            return api_response_handler.create_success_response_v1(
                response_data=video_game,
                http_status_code=200
            )
        else:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND, 
                error_string=f"Video game with game_id {game_id} not found",
                http_status_code=422
            )
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )

@router.post("/video-games/add",
            description="Add a video game",
            tags=["video-game", "video", "game"],
            response_model=StandardSuccessResponseV1[VideoGame],
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def add_game(request: Request, video_game: VideoGame):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    
    try:
        
        existed = request.app.video_game_collection.find_one(
            {"game_id": video_game.game_id}
        )
        
        if existed:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string="Video game data with this hash already exists.",
                http_status_code=422
            )
        
        request.app.video_game_collection.insert_one(video_game.to_dict())
        
        return api_response_handler.create_success_response_v1(
            response_data=video_game.to_dict(),
            http_status_code=200
        )
        
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )

@router.put("/video-games/update",
            description="Update a video game",
            tags=["video-game", "video", "game"],
            response_model=StandardSuccessResponseV1[VideoGame],
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def add_game(request: Request, video_game: VideoGame):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    
    try:
        existed = request.app.video_game_collection.find_one(
            {"game_id": video_game.game_id}, 
            {"_id": 0}
        )
        
        if not existed:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string="Video game data with {} not found".format({video_game.game_id}),
                http_status_code=422
            )
        
        request.app.video_game_collection.update_one(
            {"game_id": video_game.game_id}, 
            {"$set": video_game.to_dict()}
        )
        
        return api_response_handler.create_success_response_v1(
            response_data=video_game.to_dict(),
            http_status_code=200
        )
        
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )


@router.delete("/video-games/delete",
            description="Update a video game",
            tags=["video-game", "video", "game"],
            response_model=StandardSuccessResponseV1[VideoGame],
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def add_game(request: Request, game_id: str):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    
    try:
        result = request.app.video_game_collection.delete_one({
            "game_id": game_id
        })
        
        if result.deleted_count == 0:
            return api_response_handler.create_success_delete_response_v1(
                wasPresent=False,
                http_status_code=200
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