from fastapi import Request, HTTPException, APIRouter, Response

from utility.minio import cmd
from utility.path import separate_bucket_and_file_path
router = APIRouter()


@router.get("/image/random/{dataset}")
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

@router.get("/image/data-by-filepath")
def get_image_data_by_filepath(request: Request, file_path: str = None):

    bucket_name, file_path = separate_bucket_and_file_path(file_path)

    image_data = cmd.get_file_from_minio(request.app.minio_client, bucket_name, file_path)

    # Load data into memory
    content = image_data.read()

    response = Response(content=content, media_type="image/jpeg")

    return response

@router.get("/image/list-metadata")
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
