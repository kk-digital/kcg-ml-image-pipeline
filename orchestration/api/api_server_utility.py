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

# Replace with your MongoDB connection string
client = MongoClient("mongodb://192.168.3.1:32017/")
db = client["orchestration-job-db"]

@router.get("/database-size")
async def get_database_size():
    try:
        database_stats = db.command("dbstats")
        return {
            "database_size": database_stats["storageSize"],
            "data_size": database_stats["dataSize"],
            "index_size": database_stats["indexSize"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/collection-sizes")
async def get_collection_sizes():
    try:
        collection_sizes = {}
        for collection_name in db.list_collection_names():
            collection_stats = db.command("collstats", collection_name)
            collection_sizes[collection_name] = {
                "size": collection_stats["size"],
                "storage_size": collection_stats["storageSize"],
                "total_index_size": collection_stats["totalIndexSize"]
            }
        return collection_sizes
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
