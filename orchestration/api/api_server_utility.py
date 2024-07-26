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

def bytes_to_gb(size_in_bytes):
    return size_in_bytes / (1024 ** 3)

@router.get("/database-used-size")
async def get_database_used_size(request: Request):
    try:
        response_handler = await ApiResponseHandlerV1.createInstance(request)
        database_stats = db.command("dbstats")
        data_size_gb = bytes_to_gb(database_stats["dataSize"])
        index_size_gb = bytes_to_gb(database_stats["indexSize"])
        total_used_size_gb = data_size_gb + index_size_gb
        
        result = {
            "dataSize": total_used_size_gb, 
            "indexSize": index_size_gb
        }
        return response_handler.create_success_response_v1(response_data=result, http_status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/collection-sizes")
async def get_collection_sizes(request: Request):
    try:
        response_handler = await ApiResponseHandlerV1.createInstance(request)
        collection_sizes = {}
        for collection_name in db.list_collection_names():
            collection_stats = db.command("collstats", collection_name)
            collection_sizes[collection_name] = {
                "size_gb": bytes_to_gb(collection_stats["size"]),
                "storage_size_gb": bytes_to_gb(collection_stats["storageSize"]),
                "total_index_size_gb": bytes_to_gb(collection_stats["totalIndexSize"])
            }
        return response_handler.create_success_response_v1(response_data=collection_sizes, http_status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))