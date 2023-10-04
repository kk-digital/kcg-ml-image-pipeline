from fastapi import Request, HTTPException, APIRouter
from orchestration.api.schemas import Selection
import json
import os
from datetime import datetime
from utility.minio import cmd
from utility.path import separate_bucket_and_file_path
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

    # get image from minio server
    # Get data of an object.
    output_file_path = document["task_output_file_dict"]["output_file_path"]
    bucket_name, file_path = separate_bucket_and_file_path(output_file_path)
    try:
        response = request.app.minio_client.get_object(bucket_name, file_path)
        image_data = BytesIO(response.data)
        img = Image.open(image_data)

        # convert to base64
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        document["image-data"] = img_str

    finally:
        response.close()
        response.release_conn()

    return document


@router.get("/get-datasets")
def get_datasets(request: Request):
    objects = cmd.get_list_of_objects(request.app.minio_client, "datasets")

    return objects


def print_nodes_recursive(d, parent_name=None, level=0):
    for name, child in d.items():
        full_name = f"{parent_name}/{name}" if parent_name else name

        print('\n\n')
        print(' ' * level)
        print(full_name)
        level = level + 1
        if isinstance(child, dict):
            print_nodes_recursive(child, full_name, level)
        elif isinstance(child, list):
            print(child)

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

@router.get("/get-list-images")
def get_list_images(request: Request, dataset: str = None, page_size: int = 20, page_number: int = 0):

    objects = cmd.get_list_of_objects(request.app.minio_client, "datasets", dataset)
    print(objects)
    #print_nodes_recursive(objects, 'datasets')
    return objects

@router.get("/get-images-metadata")
def get_images_metadata(request: Request, dataset: str = None, limit: int = 20, offset: int = 0):
    jobs = request.app.completed_jobs_collection.find({'task_input_dict.dataset': {'$regex': '*'}}).skip(offset).limit(limit)
    print(jobs['task_input_dict']['dataset'])
    for job in jobs:
        job.pop('_id', None)

    return jobs