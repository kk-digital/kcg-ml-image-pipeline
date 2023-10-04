from fastapi import Request, HTTPException, APIRouter
from fastapi.responses import StreamingResponse

from orchestration.api.schemas import Selection
import json
import os
from datetime import datetime
from utility.minio import cmd
from utility.path import separate_bucket_and_file_path, file_exists
from PIL import Image
from io import BytesIO
import base64

router = APIRouter()


@router.get("/get-random-image/{dataset}")
def get_random_image(request: Request, dataset: str = None):
    # find
    documents = request.app.completed_jobs_collection.aggregate([
        {"$match": {"task_input_dict.dataset": dataset}},
        {"$sample": {"size": 1}}
    ])

    # convert curser type to list
    documents = list(documents)
    if len(documents) == 0:
        raise HTTPException(status_code=404)

    # get only the first index
    document = documents[0]

    # remove the auto generated field
    document.pop('_id', None)

    return document


@router.get("/get-datasets")
def get_datasets(request: Request):
    objects = cmd.get_list_of_objects(request.app.minio_client, "datasets")

    return objects

@router.post("/add-selection-datapoint/{dataset}")
def add_selection_datapoint(request: Request, dataset: str, selection: Selection):
    time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    selection.datetime = time

    # prepare path
    file_name = "{}-{}.json".format(time, selection.username)
    path = "data/ranking/aggregate"
    full_path = os.path.join(dataset, path, file_name)

    # convert to bytes
    dict_data = selection.to_dict()
    json_data = json.dumps(dict_data, indent=4).encode('utf-8')
    data = BytesIO(json_data)

    # upload
    cmd.upload_data(request.app.minio_client, "datasets", full_path, data)

    return True

@router.get("/get-image-data-by-filepath")
def get_image_data_by_filepath(request: Request, file_path: str = None):

    bucket_name, file_path = separate_bucket_and_file_path(file_path)

    # if file does not exist download it first
    image_data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, file_path)

    response = StreamingResponse(image_data, media_type="image/jpeg")

    return response

@router.get("/get-images-metadata")
def get_images_metadata(request: Request, dataset: str = None, limit: int = 20, offset: int = 0):
    jobs = request.app.completed_jobs_collection.find({
        '$or': [
            {'task_type' : 'image_generation_task'},
            {'task_type' : 'inpainting_generation_task'}
        ],
        'task_input_dict.dataset': dataset
    }).skip(offset).limit(limit)

    images_metadata = []
    for job in jobs:
        image_meta_data = {
            'dataset' : job['task_input_dict']['dataset'],
            'task_type' : job['task_type'],
            'image_path': job['task_output_file_dict']['output_file_path'],
            'image_hash': job['task_output_file_dict']['output_file_hash']
        }
        images_metadata.append(image_meta_data)

    return images_metadata