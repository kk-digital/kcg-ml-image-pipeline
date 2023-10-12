from fastapi import Request, APIRouter, Query, HTTPException, Response
from utility.minio import cmd
import json

router = APIRouter()

@router.get("/models/rank-relevancy/list-models")
def get_relevancy_models(request: Request, dataset: str = Query(...)):
    # Bucket name
    bucket_name = "datasets"
    
    # Prefix/path inside the bucket where relevancy models for the dataset are stored in MinIO
    prefix = f"{dataset}/models/relevancy/ab_ranking_efficient_net"

    # Fetch list of model objects from MinIO
    model_objects = cmd.get_list_of_objects_with_prefix(request.app.minio_client, bucket_name, prefix)

    # Parse models list from model_objects
    models_list = []
    for obj in model_objects:
        # Filter out only the .json files for processing
        if obj.endswith('.json'):
            data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, obj)
            model_content = json.loads(data.read().decode('utf-8'))
            
            # Extract model name from the JSON file name (like '2023-10-09.json') and append .pth
            model_name = obj.split('/')[-1].split('.')[0] 
            
            # Construct a new dictionary with model_name at the top
            arranged_content = {
                'model_name': model_name,
                **model_content
            }
            
            # Append the rearranged content of the JSON file to the models_list
            models_list.append(arranged_content)

    return models_list


@router.get("/models/rank-embedding/list-models")
def get_ranking_models(request: Request, dataset: str = Query(...)):
    # Bucket name
    bucket_name = "datasets"
    
    # Prefix/path inside the bucket where ranking models for the dataset are stored in MinIO
    prefix = f"{dataset}/models/ranking/ab_ranking_efficient_net"

    # Similar to the previous function, fetch, parse and rearrange the model information
    model_objects = cmd.get_list_of_objects_with_prefix(request.app.minio_client, bucket_name, prefix)

    models_list = []
    for obj in model_objects:
        if obj.endswith('.json'):
            data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, obj)
            model_content = json.loads(data.read().decode('utf-8'))
            model_name = obj.split('/')[-1].split('.')[0] 
            arranged_content = {
                'model_name': model_name,
                **model_content
            }
            models_list.append(arranged_content)

    return models_list


@router.get("/models/get-model-card")
def get_model_card(request: Request, file_path: str = Query(...)):
    bucket_name = "datasets"
    
    # Check if the file exists in the MinIO bucket
    if not cmd.is_object_exists(request.app.minio_client, bucket_name, file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, file_path)
    
    # If the file is a .json file, decode it and return the content, otherwise, return the raw content
    if file_path.endswith('.json'):
        return json.loads(data.read().decode('utf-8'))
    else:
        return data.read()

@router.get("/models/get-graph")
def get_graph(request: Request, file_path: str = Query(...)):
    bucket_name = "datasets"
    
    # Check if the file exists
    if not cmd.is_object_exists(request.app.minio_client, bucket_name, file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    image_data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, file_path)
    
    # Load data into memory
    content = image_data.read()

    # Determine content type based on file extension
    content_type = "image/png" if file_path.endswith('.png') else "application/octet-stream"
    
    return Response(content=content, media_type=content_type)


@router.get("/models/get-report")
def get_report(request: Request, file_path: str = Query(...)):
    bucket_name = "datasets"
    
    # Check if the file exists
    if not cmd.is_object_exists(request.app.minio_client, bucket_name, file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    report_data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, file_path)
    
    # Load data into memory
    content = report_data.read()

    # Determine content type based on file extension (assuming .txt for now, but you can expand this logic)
    content_type = "text/plain" if file_path.endswith('.txt') else "application/octet-stream"
    
    return Response(content=content, media_type=content_type)

