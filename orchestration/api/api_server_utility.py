from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from .api_utils import PrettyJSONResponse, ApiResponseHandlerV1, StandardSuccessResponseV1, ErrorCode, WasPresentResponse
from pymongo import MongoClient


router = APIRouter()

@router.get("/ping-server", response_class=JSONResponse, tags = ["deprecated3"], description = "changed with /ping ")
async def ping_server():
    headers = {
        "content-type": "application/json; charset=utf-8",
    }
    return JSONResponse(content="Pong!", headers=headers, status_code=200)


@router.get("/ping", 
            response_model=StandardSuccessResponseV1[None],
            tags = ['utility'],
            description="dummy endpoint to test connection with the server",
            responses=ApiResponseHandlerV1.listErrors([422, 500]))
def ping(request: Request):
    response_handler = ApiResponseHandlerV1(request)
    # Simply return None for data, indicating a successful ping with no additional data
    return response_handler.create_success_response_v1(
        response_data=None,  
        http_status_code=200
    )
