from fastapi import Request, APIRouter
from utility.minio import cmd
import json


router = APIRouter()

@router.get("/datasets/{dataset}/models")
def get_dataset_models(request: Request, dataset: str):
    # Base path where models for the dataset are stored in MinIO
    base_path = f"datasets/{dataset}/models/ab_ranking_efficient_net"

    # Fetch list of model objects from MinIO
    model_objects = cmd.get_list_of_objects(request.app.minio_client, base_path)

    # Parse models list from model_objects
    models_list = []
    for obj in model_objects:
        # Filter out only the .json files for processing
        if obj['name'].endswith('.json'):
            data = cmd.get_file_from_minio(request.app.minio_client, base_path.split('/')[0], obj['name'])
            model_content = json.loads(data.read().decode('utf-8'))
            
            # Extract model name from the JSON file name (like '2023-10-09.json')
            model_name = obj['name'].split('.')[0]
            model_content['model_name'] = model_name
            
            # Append the entire content of the JSON file, along with the model name, to the models_list
            models_list.append(model_content)

    return models_list
