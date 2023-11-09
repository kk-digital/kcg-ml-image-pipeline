from fastapi import Request, APIRouter, Query, HTTPException, Response
from utility.minio import cmd
import json
from orchestration.api.mongo_schemas import RankingModel
from .api_utils import PrettyJSONResponse
from datetime import datetime

router = APIRouter()


def get_next_model_id_sequence(request: Request):
    # get models counter
    counter = request.app.counters_collection.find_one({"_id": "models"})
    print("counter=",counter)
    counter_seq = counter["seq"]
    counter_seq += 1

    try:
        ret = request.app.counters_collection.update_one(
            {"_id": "models"},
            {"$set": {"seq": counter_seq}})
    except Exception as e:
        raise Exception("Updating of model counter failed: {}".format(e))

    return counter_seq


@router.get("/models/rank-relevancy/list-models", response_class=PrettyJSONResponse)
def get_relevancy_models(request: Request, dataset: str = Query(...)):
    # Bucket name
    bucket_name = "datasets"
    
    # Base path where relevancy models for the dataset are stored in MinIO
    base_path = f"{dataset}/models/relevancy"
    
    # Fetch list of model objects from MinIO for the base path, recursively
    model_objects = []
    objects = request.app.minio_client.list_objects(bucket_name, prefix=base_path, recursive=True)
    for obj in objects:
        model_objects.append(obj.object_name)

    # Parse models list from model_objects
    models_list = []
    for obj in model_objects:
        # Filter out only the .json files for processing
        if obj.endswith('.json'):
            data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, obj)
            model_content = json.loads(data.read().decode('utf-8'))
            
            # Extract the full model name from the model_path
            model_name = model_content['model_path'].split('/')[-1].split('.')[0]
            
            # Extract model architecture from the object path (like 'ab_ranking_linear' or 'ab_ranking_efficient_net')
            model_architecture = obj.split('/')[-2]
            
            # Construct a new dictionary with model_name and model_architecture at the top
            arranged_content = {
                'model_name': model_name,
                'model_architecture': model_architecture,
                **model_content
            }
            
            # Append the rearranged content of the JSON file to the models_list
            models_list.append(arranged_content)

    # Custom sorting
    models_list.sort(key=lambda x: not x["model_name"].endswith('.pth'))
    
    models_list.sort(key=lambda x: x["model_name"].split('_')[0] if x["model_name"].endswith('.pth') else x["model_name"], reverse=True)

    return models_list

@router.get("/models/rank-embedding/list-models", response_class=PrettyJSONResponse)
def get_ranking_models(request: Request, dataset: str = Query(...)):
    # Bucket name
    bucket_name = "datasets"
    
    # Base path where ranking models for the dataset are stored in MinIO
    base_path = f"{dataset}/models/ranking"
    
    # Fetch list of model objects from MinIO for the base path, recursively
    model_objects = []
    objects = request.app.minio_client.list_objects(bucket_name, prefix=base_path, recursive=True)
    for obj in objects:
        model_objects.append(obj.object_name)

    # Parse models list from model_objects
    models_list = []
    for obj in model_objects:
        # Filter out only the .json files for processing
        if obj.endswith('.json'):
            data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, obj)
            model_content = json.loads(data.read().decode('utf-8'))
            
            # Extract the full model name from the model_path
            model_name = model_content['model_path'].split('/')[-1].split('.')[0]

            # Extract model architecture from the object path (like 'ab_ranking_linear' or 'ab_ranking_efficient_net')
            model_architecture = obj.split('/')[-2]
            
            # Construct a new dictionary with model_name and model_architecture at the top
            arranged_content = {
                'model_name': model_name,
                'model_architecture': model_architecture,
                **model_content
            }
            
            # Append the rearranged content of the JSON file to the models_list
            models_list.append(arranged_content)

    # Custom sorting
    models_list.sort(key=lambda x: not x["model_name"].endswith('.pth'))
    
    # Further refine the sorting based on the model name
    models_list.sort(key=lambda x: x["model_name"].split('_')[0] if x["model_name"].endswith('.pth') else x["model_name"], reverse=True)

    return models_list

@router.get("/models/rank-embedding/latest-model")
def get_latest_ranking_model(request: Request,
                             dataset: str = Query(...),
                             input_type: str = 'embedding',
                             output_type: str = 'score'):
    # Bucket name
    bucket_name = "datasets"

    # Base path where ranking models for the dataset are stored in MinIO
    base_path = f"{dataset}/models/ranking"

    # Fetch list of model objects from MinIO for the base path, recursively
    model_objects = []
    objects = request.app.minio_client.list_objects(bucket_name, prefix=base_path, recursive=True)
    for obj in objects:
        model_objects.append(obj.object_name)

    # Parse models list from model_objects
    models_list = []
    for obj in model_objects:
        # Filter out only the .json files for processing
        if obj.endswith('.json'):
            data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, obj)
            model_content = json.loads(data.read().decode('utf-8'))

            # Extract model name from the JSON file name (like '2023-10-09.json')
            model_name = obj.split('/')[-1].split('.')[0]

            # Extract model architecture from the object path (like 'ab_ranking_linear' or 'ab_ranking_efficient_net')
            model_architecture = obj.split('/')[-2]

            # Construct a new dictionary with model_name and model_architecture at the top
            arranged_content = {
                'model_name': model_name,
                'model_architecture': model_architecture,
                **model_content
            }

            # Append the rearranged content of the JSON file to the models_list
            models_list.append(arranged_content)


    result_model = None
    for model in models_list:

        model_input_type = model['input_type']
        model_output_type = model['output_type']

        # filter the by input_type & output_type
        if input_type != model_input_type or output_type != model_output_type:
            # quick exit
            continue

        if result_model is None:
            result_model = model
            continue

        # Here we know that the model is not 'None'
        model_date_string = model['model_creation_date']
        result_model_date_string = result_model['model_creation_date']
        # Convert strings to datetime objects
        model_date = datetime.strptime(model_date_string, "%Y-%m-%d")
        result_model_date = datetime.strptime(result_model_date_string, "%Y-%m-%d")

        if model_date > result_model_date:
            result_model = model

    return result_model


@router.get("/models/get-model-card", response_class=PrettyJSONResponse)
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


@router.post("/models/add", description="Add a model to model collection")
def add_model(request: Request, model: RankingModel):
    # check if exist
    query = {"model_file_hash": model.model_file_hash}
    item = request.app.models_collection.find_one(query)
    if item is None:
        # add one
        model.model_id = get_next_model_id_sequence(request)
        request.app.models_collection.insert_one(model.to_dict())

        return model.model_id

    return item["model_id"]


@router.get("/models/get-id", description="Get model id")
def get_model_id(request: Request, model_hash: str):
    # check if exist
    query = {"model_file_hash": model_hash}
    item = request.app.models_collection.find_one(query)
    if item is None:
        raise HTTPException(status_code=404, detail="Model not found")

    return item["model_id"]




#new Endpoints with /static/ prefix



@router.get("/static/models/get-report")
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



@router.get("/static/models/get-model-card", response_class=PrettyJSONResponse)
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
