from fastapi import Request, APIRouter, HTTPException, Query, Body
from utility.minio import cmd
from .api_utils import PrettyJSONResponse, ApiResponseHandlerV1, StandardSuccessResponseV1, ErrorCode
from orchestration.api.mongo_schemas import Worker, ListWorker
import json
import paramiko
import csv
from bson import ObjectId
from datetime import datetime


router = APIRouter()

@router.post("/worker-stats", response_class=PrettyJSONResponse)
def get_worker_stats(ssh_key_path: str = Query(...), 
                     server_address: str = Query("123.176.98.90")):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    gpu_stats_all = []
    port_numbers = [40029, 40030, 40132]  

    try:
        for port_number in port_numbers:
            try:
                ssh.connect(server_address, port=port_number, username='root', key_filename=ssh_key_path)

                command = '''
                python -c "import json; import socket; import GPUtil; print(json.dumps([{\'temperature\': gpu.temperature, \'load\': gpu.load, \'total_memory\': gpu.memoryTotal, \'used_memory\': gpu.memoryUsed, \'worker_name\': socket.gethostname()} for gpu in GPUtil.getGPUs()]))"
                '''
                stdin, stdout, stderr = ssh.exec_command(command)

                stderr_output = stderr.read().decode('utf-8')
                if stderr_output:
                    print(f"Error on server {server_address}:{port_number}:", stderr_output)
                    continue

                gpu_stats = json.loads(stdout.read().decode('utf-8'))
                gpu_stats_all.extend(gpu_stats)

            finally:
                ssh.close()

        return gpu_stats_all

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/ping", 
            response_model=StandardSuccessResponseV1[None],
            description="dummy endpoint to test connection with the server",
            responses=ApiResponseHandlerV1.listErrors([422, 500]))
def ping(request: Request):
    response_handler = ApiResponseHandlerV1(request)
    # Simply return None for data, indicating a successful ping with no additional data
    return response_handler.create_success_response_v1(
        response_data=None,  
        http_status_code=200
    )


@router.post("/worker/register-worker",
             status_code=201,
             tags=["worker"],
             description="Register a new worker",
             response_model=StandardSuccessResponseV1[Worker],  
             responses=ApiResponseHandlerV1.listErrors([422, 500]))
async def register_worker(request: Request, worker_data: Worker):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        # Prepare worker document
        worker_document = worker_data.to_dict()
        worker_document["last_seen"] = datetime.utcnow().isoformat()  

        # Insert new worker
        inserted_id = request.app.workers_collection.insert_one(worker_document).inserted_id

        # Retrieve and serialize the new worker object
        new_worker = request.app.workers_collection.find_one({"_id": inserted_id})
        serialized_worker = {k: str(v) if isinstance(v, ObjectId) else v for k, v in new_worker.items()}

        # Return success response
        return response_handler.create_success_response_v1(
            response_data=serialized_worker,
            http_status_code=201,
        )
    except Exception as e:
        # Handle exceptions and return an error response
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=f"Internal server error: {str(e)}",
            http_status_code=500,
        )
    
@router.get("/worker/list-workers",
            status_code=200,
            tags=["worker"],
            description="List registered workers",
            response_model=StandardSuccessResponseV1[ListWorker], 
            responses=ApiResponseHandlerV1.listErrors([500]))
async def list_workers(
    request: Request):

    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        # Fetch data from MongoDB with pagination
        cursor = request.app.workers_collection.find()
        workers_data = list(cursor)

        # Prepare the data for the response
        workers = [ListWorker(**doc).dict(exclude={'_id'}) for doc in workers_data]

        # Return the fetched data with a success response
        return response_handler.create_success_response_v1(
            response_data={"worker": workers},
            http_status_code=200
        )
    except Exception as e:
        # Handle exceptions and return an error response
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=f"Internal Server Error: {str(e)}",
            http_status_code=500,
        )    