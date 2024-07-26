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

def format_size_gb(size_in_bytes):
    size_in_gb = size_in_bytes / (1024 ** 3)
    return f"{size_in_gb:.5f} GB"


@router.get("/database-sizes")
async def get_database_sizes(request: Request):
    try:
        response_handler = await ApiResponseHandlerV1.createInstance(request)
        database_stats = db.command("dbstats")
        total_used_size_gb = format_size_gb(database_stats["dataSize"] + database_stats["indexSize"])

        result = {
            "total_used_size_gb": total_used_size_gb,
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
            used_size_gb = format_size_gb(collection_stats["size"] + collection_stats["totalIndexSize"])
            avg_obj_size_bytes = collection_stats["avgObjSize"]
            num_objects = collection_stats["count"]
            # Min and max size are not directly available, need to compute if needed
            min_obj_size_bytes = collection_stats.get("minObjSize", 0)
            max_obj_size_bytes = collection_stats.get("maxObjSize", 0)

            # If min and max object size are not available in stats, they need to be computed manually (could be heavy)
            if "minObjSize" not in collection_stats or "maxObjSize" not in collection_stats:
                sizes = [len(str(doc).encode('utf-8')) for doc in db[collection_name].find()]
                if sizes:
                    min_obj_size_bytes = min(sizes)
                    max_obj_size_bytes = max(sizes)
                else:
                    min_obj_size_bytes = 0
                    max_obj_size_bytes = 0

            collection_sizes[collection_name] = {
                "used_size_gb": used_size_gb,
                "number_of_objects": num_objects,
                "average_object_size_bytes": avg_obj_size_bytes,
                "min_object_size_bytes": min_obj_size_bytes,
                "max_object_size_bytes": max_obj_size_bytes
            }
        return response_handler.create_success_response_v1(response_data=collection_sizes, http_status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))