from fastapi import Request, APIRouter
from utility.minio import cmd
import json

router = APIRouter()

@router.get("/datasets/{dataset}/models")
def get_dataset_models(request: Request, dataset: str):
    # Bucket name
    bucket_name = "datasets"
    
    # Prefix/path inside the bucket where models for the dataset are stored in MinIO
    prefix = f"{dataset}/models/ab_ranking_efficient_net"

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
            model_name = obj.split('/')[-1].split('.')[0] + '.pth'
            
            # Construct a new dictionary with model_name at the top
            arranged_content = {
                'model_name': model_name,
                **model_content
            }
            
            # Append the rearranged content of the JSON file to the models_list
            models_list.append(arranged_content)

    return models_list
