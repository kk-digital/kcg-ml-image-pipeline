from fastapi import Request, APIRouter, Query, HTTPException, Response
from utility.minio import cmd
import json
from orchestration.api.mongo_schemas import RankingModel
from .api_utils import PrettyJSONResponse, ApiResponseHandler, ErrorCode, ApiResponseHandlerV1, StandardSuccessResponseV1, ModelResponse, ModelIdResponse, ModelTypeResponse, ModelsAndScoresResponse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from typing import List

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


@router.get("/models/rank-embedding/latest-model", tags = ['deprecated'], description="'/models/rank-embedding/get-latest-model' is the replacement")
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
        model_type = model['model_type']

        if model_type != 'image-pair-ranking-linear':
            continue

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

# TODO: deprecate
@router.get("/models/get-model-card", tags = ["deprecated"], response_class=PrettyJSONResponse)
def get_model_card(request: Request, file_path: str = Query(...)):
    bucket_name = "datasets"
    
    # Check if the file exists in the MinIO bucket
    if not cmd.is_object_exists(request.app.minio_client, bucket_name, file_path):
        return None

    data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, file_path)
    
    # If the file is a .json file, decode it and return the content, otherwise, return the raw content
    if file_path.endswith('.json'):
        return json.loads(data.read().decode('utf-8'))
    else:
        return data.read()


@router.get("/models/get-graph", tags = ["deprecated"])
def get_graph(request: Request, file_path: str = Query(...)):
    bucket_name = "datasets"
    
    # Check if the file exists
    if not cmd.is_object_exists(request.app.minio_client, bucket_name, file_path):
        return None
    
    image_data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, file_path)
    
    # Load data into memory
    content = image_data.read()

    # Determine content type based on file extension
    content_type = "image/png" if file_path.endswith('.png') else "application/octet-stream"
    
    return Response(content=content, media_type=content_type)


# TODO: deprecate
@router.get("/models/get-report", tags = ['deprecated'])
def get_report(request: Request, file_path: str = Query(...)):
    bucket_name = "datasets"
    
    # Check if the file exists
    if not cmd.is_object_exists(request.app.minio_client, bucket_name, file_path):
        return None
    
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
        return None

    return item["model_id"]


@router.get("/static/models/get-latest-graph")
async def get_latest_graph(request: Request, dataset: str = Query(...), model_type: str = Query(...)):
    bucket_name = "datasets"
    base_path = f"{dataset}/output/scores-graph"

    # List all files in the directory
    try:
        # Assuming that the list_objects method returns a list of objects with an 'object_name' attribute
        objects = request.app.minio_client.list_objects(bucket_name, prefix=base_path, recursive=True)
        files = [obj.object_name for obj in objects]  # Replace 'object_name' with the correct attribute
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Filter files by model_type and sort them to get the latest one
    filtered_files = [filename for filename in files if re.match(rf".*{model_type}.*\.png", filename)]
    filtered_files.sort(reverse=True)
    
    if not filtered_files:
        raise HTTPException(status_code=404, detail="File not found")

    latest_file_path = filtered_files[0]
    
    # Get the latest graph image data
    try:
        image_data = request.app.minio_client.get_object(bucket_name, latest_file_path)
        content = image_data.read()
        content_type = "image/png"
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return Response(content=content, media_type=content_type)


@router.get("/models/list-model-types", response_class=PrettyJSONResponse)
async def list_model_types(request: Request, dataset: str):
    response_handler = ApiResponseHandler(request)
    bucket_name = "datasets"
    base_path = f"{dataset}/output/scores-graph"

    try:
        objects = request.app.minio_client.list_objects(bucket_name, prefix=base_path, recursive=True)
        files = [obj.object_name for obj in objects]

        # Extract model types from file names
        model_types = set()
        pattern = rf".*score-(.+?)-{re.escape(dataset)}\.png"
        for file in files:
            match = re.match(pattern, file)
            if match:
                model_types.add(match.group(1))

        return response_handler.create_success_response({"model_types": list(model_types)}, http_status_code=200)

    except Exception as e:
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Error listing model types", 500)




# New Endpoints with /static/ prefix

@router.get("/static/models/get-model-card/{file_path:path}", response_class=PrettyJSONResponse)
def get_model_card(request: Request, file_path: str):
    bucket_name = "datasets"
    
    if not cmd.is_object_exists(request.app.minio_client, bucket_name, file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, file_path)
    
    if file_path.endswith('.json'):
        return json.loads(data.read().decode('utf-8'))
    else:
        return data.read()


@router.get("/static/models/get-graph/{file_path:path}")
def get_graph(request: Request, file_path: str):
    bucket_name = "datasets"
    
    if not cmd.is_object_exists(request.app.minio_client, bucket_name, file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    image_data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, file_path)
    
    content = image_data.read()
    content_type = "image/png" if file_path.endswith('.png') else "application/octet-stream"
    
    return Response(content=content, media_type=content_type)


@router.get("/static/models/get-report/{file_path:path}")
def get_report(request: Request, file_path: str):
    bucket_name = "datasets"
    
    if not cmd.is_object_exists(request.app.minio_client, bucket_name, file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    report_data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, file_path)
    
    content = report_data.read()
    content_type = "text/plain" if file_path.endswith('.txt') else "application/octet-stream"
    
    return Response(content=content, media_type=content_type)


# new apis

@router.get("/models/rank-relevancy/list-models-v1",
            response_model=StandardSuccessResponseV1[ModelResponse], 
            description="List relevancy models",
            tags=["models"],
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([400, 500]))
def get_relevancy_models(request: Request, dataset: str = Query(...)):
    response_handler = ApiResponseHandlerV1(request)
    try:
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

                # Extract the full model name and model architecture from the object path
                model_name = obj.split('/')[-1].replace('.json', '')
                model_architecture = obj.split('/')[-2]

                # Append to models_list
                models_list.append({
                    'model_name': model_name,
                    'model_architecture': model_architecture,
                    **model_content
                })

        # Custom sorting (adjust as needed)
        models_list.sort(key=lambda x: not x["model_name"].endswith('.pth'))
        models_list.sort(key=lambda x: x["model_name"].split('_')[0] if x["model_name"].endswith('.pth') else x["model_name"], reverse=True)

        # Return success response with models list
        return response_handler.create_success_response_v1(response_data={"models": models_list}, http_status_code=200)

    except Exception as e:
        # Log the exception and return an error response
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )

@router.get("/models/rank-embedding/list-models-v2",
            response_model=StandardSuccessResponseV1[ModelResponse],  
            description="List ranking models ",
            tags=["models"],
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([400, 500]))
def get_ranking_models(request: Request, dataset: str = Query(...)):
    response_handler = ApiResponseHandlerV1(request)
    try:
        bucket_name = "datasets"
        base_path = f"{dataset}/models/ranking"

        objects = request.app.minio_client.list_objects(bucket_name, prefix=base_path, recursive=True)
        model_objects = [obj.object_name for obj in objects if obj.object_name.endswith('.json')]

        def fetch_model_content(obj_name):
            data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, obj_name)
            return json.loads(data.read().decode('utf-8'))

        models_list = []
        with ThreadPoolExecutor() as executor:
            future_to_obj = {executor.submit(fetch_model_content, obj): obj for obj in model_objects}
            for future in as_completed(future_to_obj):
                obj_name = future_to_obj[future]
                try:
                    model_content = future.result()
                    model_name = model_content['model_path'].split('/')[-1].split('.')[0]
                    model_architecture = obj_name.split('/')[-2]
                    arranged_content = {
                        'model_name': model_name,
                        'model_architecture': model_architecture,
                        **model_content
                    }
                    models_list.append(arranged_content)
                except Exception as exc:
                    print(f'{obj_name} generated an exception: {exc}')

        models_list.sort(key=lambda x: not x["model_name"].endswith('.pth'))
        models_list.sort(key=lambda x: x["model_name"].split('_')[0] if x["model_name"].endswith('.pth') else x["model_name"], reverse=True)

        # Return success response with models list
        return response_handler.create_success_response_v1(response_data={"models": models_list}, http_status_code=200)

    except Exception as e:
        # Log the exception and return an error response
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )        


@router.get("/models/rank-embedding/get-latest-model",
            response_model=StandardSuccessResponseV1[ModelResponse],  
            description="Get the latest ranking model",
            tags=["models"],
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([400, 500]))
def get_latest_ranking_model_v1(request: Request,
                             dataset: str = Query(...),
                             input_type: str = 'embedding',
                             output_type: str = 'score'):
    response_handler = ApiResponseHandlerV1(request)
    try:
        bucket_name = "datasets"
        base_path = f"{dataset}/models/ranking"

        objects = request.app.minio_client.list_objects(bucket_name, prefix=base_path, recursive=True)
        models_list = [obj.object_name for obj in objects if obj.object_name.endswith('.json')]

        def fetch_model_content(obj_name):
            data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, obj_name)
            return json.loads(data.read().decode('utf-8'))

        # Initialize result_model as None
        result_model = None

        # Fetch and process model contents
        for obj_name in models_list:
            model_content = fetch_model_content(obj_name)
            model_name = obj_name.split('/')[-1].split('.')[0]
            model_architecture = obj_name.split('/')[-2]

            arranged_content = {
                'model_name': model_name,
                'model_architecture': model_architecture,
                **model_content
            }

            model_input_type = arranged_content['input_type']
            model_output_type = arranged_content['output_type']
            model_type = arranged_content.get('model_type', '')

            if model_type != 'image-pair-ranking-linear' or input_type != model_input_type or output_type != model_output_type:
                continue

            # Check for the latest model based on model_creation_date
            if result_model is None or datetime.strptime(arranged_content['model_creation_date'], "%Y-%m-%d") > datetime.strptime(result_model['model_creation_date'], "%Y-%m-%d"):
                result_model = arranged_content

        # Return the result model if found, else return None
        return response_handler.create_success_response_v1(response_data={"model": result_model}, http_status_code=200)

    except Exception as e:
        # Log the exception and return an error response
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )        
    

@router.post("/models/add-v1",
             response_model=StandardSuccessResponseV1[ModelIdResponse], 
             description="Add a model to model collection",
             tags=["deprecated2"],
             status_code=200,
             responses=ApiResponseHandlerV1.listErrors([400, 500]))
async def add_model(request: Request, model: RankingModel):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        query = {"model_file_hash": model.model_file_hash}
        item = request.app.models_collection.find_one(query)
        if item is None:
            # add one
            model.model_id = get_next_model_id_sequence(request)
            request.app.models_collection.insert_one(model.to_dict())
            model_id = model.model_id
        else:
            model_id = item["model_id"]

        return response_handler.create_success_response_v1(response_data={"model_id": model_id}, http_status_code=200)
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )    

@router.post("/models/add-model",
             response_model=StandardSuccessResponseV1[ModelIdResponse], 
             description="Add a model to model collection",
             tags=["models"],
             status_code=200,
             responses=ApiResponseHandlerV1.listErrors([400, 500]))
async def add_model(request: Request, model: RankingModel):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        query = {"model_file_hash": model.model_file_hash}
        item = request.app.models_collection.find_one(query)
        if item is None:
            # add one
            model.model_id = get_next_model_id_sequence(request)
            request.app.models_collection.insert_one(model.to_dict())
            model_id = model.model_id
        else:
            model_id = item["model_id"]

        return response_handler.create_success_response_v1(response_data={"model_id": model_id}, http_status_code=200)
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )      

@router.get("/models/get-id-v1",
            response_model=StandardSuccessResponseV1[ModelIdResponse],  
            description="Get model id",
            tags = ["models"],
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([400, 404, 500]))
def get_model_id(request: Request, model_hash: str):
    response_handler = ApiResponseHandlerV1(request)
    try:
        query = {"model_file_hash": model_hash}
        item = request.app.models_collection.find_one(query)
        if item is None:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="Model not found.",
                http_status_code=404,
            )
        return response_handler.create_success_response_v1(response_data={"model_id": item["model_id"]}, http_status_code=200)
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )    
    

@router.get("/static/models/get-latest-graph",
            description="Get the latest graph",
            tags=["models"],
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([400, 404, 500]))
async def get_latest_graph(request: Request, dataset: str = Query(...), model_type: str = Query(...)):
    response_handler = ApiResponseHandlerV1(request)
    bucket_name = "datasets"
    base_path = f"{dataset}/output/scores-graph"

    try:
        objects = request.app.minio_client.list_objects(bucket_name, prefix=base_path, recursive=True)
        files = [obj.object_name for obj in objects if re.match(rf".*{model_type}.*\.png", obj.object_name)]
        files.sort(reverse=True)

        if not files:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="File not found",
                http_status_code=404,
            )

        latest_file_path = files[0]
        image_data = request.app.minio_client.get_object(bucket_name, latest_file_path)
        content = image_data.read()
        return Response(content=content, media_type="image/png")

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )

@router.get("/models/list-model-types-v1",
            description="List model types",
            response_model=StandardSuccessResponseV1[ModelTypeResponse],
            tags=["deprecated2"],
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([400, 500]))
async def list_model_types_v1(request: Request, dataset: str):
    response_handler = ApiResponseHandlerV1(request)
    bucket_name = "datasets"
    base_path = f"{dataset}/output/scores-graph"

    try:
        objects = request.app.minio_client.list_objects(bucket_name, prefix=base_path, recursive=True)
        files = [obj.object_name for obj in objects]

        model_types = set()
        pattern = rf".*score-(.+?)-{re.escape(dataset)}\.png"
        for file in files:
            match = re.match(pattern, file)
            if match:
                model_types.add(match.group(1))

        return response_handler.create_success_response_v1(
            response_data={"model_types": list(model_types)},
            http_status_code=200,
        )

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )    
    

@router.get("/static/models/get-model-card/{file_path:path}",
            description="Get model card",
            tags=["models"],
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([404, 500]))
def get_model_card(request: Request, file_path: str):
    response_handler = ApiResponseHandlerV1(request)
    bucket_name = "datasets"

    try:
        if not cmd.is_object_exists(request.app.minio_client, bucket_name, file_path):
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="File not found",
                http_status_code=404,
            )

        data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, file_path)
        if file_path.endswith('.json'):
            content = json.loads(data.read().decode('utf-8'))
            return response_handler.create_success_response_v1(
                response_data=content,
                http_status_code=200,
            )
        else:
            return data.read() 

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )

@router.get("/static/models/get-graph/{file_path:path}",
            description="Get graph",
            tags=["models"],
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([404, 500]))
def get_graph(request: Request, file_path: str):
    response_handler = ApiResponseHandlerV1(request)
    bucket_name = "datasets"

    try:
        if not cmd.is_object_exists(request.app.minio_client, bucket_name, file_path):
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="File not found",
                http_status_code=404,
            )

        image_data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, file_path)
        content = image_data.read()
        content_type = "image/png" if file_path.endswith('.png') else "application/octet-stream"
        return Response(content=content, media_type=content_type)  

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )

@router.get("/static/models/get-report/{file_path:path}",
            description="Get report",
            tags=["models"],
            status_code=200,
            responses=ApiResponseHandlerV1.listErrors([404, 500]))
def get_report(request: Request, file_path: str):
    response_handler = ApiResponseHandlerV1(request)
    bucket_name = "datasets"

    try:
        if not cmd.is_object_exists(request.app.minio_client, bucket_name, file_path):
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="File not found",
                http_status_code=404,
            )

        report_data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, file_path)
        content = report_data.read()
        content_type = "text/plain" if file_path.endswith('.txt') else "application/octet-stream"
        return Response(content=content, media_type=content_type) 

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )
    
CACHE = {}
CACHE_EXPIRATION_DELTA = timedelta(hours=12)  

@router.get("/models/list-model-types-and-scores", 
         response_model=StandardSuccessResponseV1[ModelsAndScoresResponse],
         tags = ['models'],
         responses=ApiResponseHandlerV1.listErrors([404, 500]),
         description="List unique score types from task_attributes_dict")
def list_task_attributes_v1(request: Request, dataset: str = Query(..., description="Dataset to filter tasks")):
    api_handler = ApiResponseHandlerV1(request)
    try:
        # Check if data is in cache and not expired
        cache_key = f"task_attributes_{dataset}"
        if cache_key in CACHE and datetime.now() - CACHE[cache_key]['timestamp'] < CACHE_EXPIRATION_DELTA:
            return api_handler.create_success_response_v1(response_data=CACHE[cache_key]['data'], http_status_code=200)

        # Fetch data from the database for the specified dataset
        tasks_cursor = request.app.completed_jobs_collection.find(
            {"task_input_dict.dataset": dataset, "task_attributes_dict": {"$exists": True, "$ne": {}}},
            {'task_attributes_dict': 1}
        )

        # Use a set for score field names and a list for model names
        score_fields = set()
        model_names = []

        # Iterate through cursor and add unique score field names and model names
        for task in tasks_cursor:
            task_attr_dict = task.get('task_attributes_dict', {})
            if isinstance(task_attr_dict, dict):  # Check if task_attr_dict is a dictionary
                for model, scores in task_attr_dict.items():
                    if model not in model_names:
                        model_names.append(model)
                    score_fields.update(scores.keys())

        # Convert set to a list to make it JSON serializable
        score_fields_list = list(score_fields)

        # Store data in cache with timestamp
        CACHE[cache_key] = {
            'timestamp': datetime.now(),
            'data': {
                "Models": model_names,
                "Scores": score_fields_list
            }
        }

        # Return success response
        return api_handler.create_success_response_v1(response_data={
            "Models": model_names,
            "Scores": score_fields_list
        }, http_status_code=200)

    except Exception as exc:
        print(f"Exception occurred: {exc}")
        return api_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string="Internal Server Error",
            http_status_code=500
        )