from fastapi import Request, APIRouter, HTTPException, Response
import requests
from .api_utils import PrettyJSONResponse, ApiResponseHandler, ErrorCode, StandardErrorResponse, StandardSuccessResponse
from orchestration.api.mongo_schemas import  PhraseModel
from typing import Optional
from typing import List
import json
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi_cache.decorator import cache
from pydantic import BaseModel

CLIP_SERVER_ADDRESS = 'http://192.168.3.31:8002'
#CLIP_SERVER_ADDRESS = 'http://127.0.0.1:8002'
router = APIRouter()


# --------- Http requests -------------
def http_clip_server_add_phrase(phrase: str):
    url = CLIP_SERVER_ADDRESS + "/add-phrase?phrase=" + phrase

    try:
        response = requests.put(url)

        if response.status_code == 200:
            result_json = response.json()
            return result_json

    except Exception as e:
        print('request exception ', e)

    return None


def http_clip_server_clip_vector_from_phrase(phrase: str):
    url = CLIP_SERVER_ADDRESS + "/clip-vector?phrase=" + phrase

    try:
        response = requests.get(url)

        if response.status_code == 200:
            result_json = response.json()
            return result_json

    except Exception as e:
        print('request exception ', e)

    return None


def http_clip_server_get_cosine_similarity(image_path: str,
                                           phrase: str):
    url = f'{CLIP_SERVER_ADDRESS}/cosine-similarity?image_path={image_path}&phrase={phrase}'

    try:
        response = requests.get(url)

        if response.status_code == 200:
            result_json = response.json()
            return result_json

    except Exception as e:
        print('request exception ', e)

    return None

def http_clip_server_get_cosine_similarity_list(image_path_list: List[str],
                                           phrase: str):
    url = f'{CLIP_SERVER_ADDRESS}/cosine-similarity-list?phrase={phrase}'

    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data

    # Use json.dumps to convert the list to a JSON-formatted string
    json_string = json.dumps(image_path_list)

    print(json_string)

    try:
        response = requests.post(url, json=image_path_list, headers=headers)

        if response.status_code == 200:
            result_json = response.json()
            return result_json

    except Exception as e:
        print('request exception ', e)

    return None

# ----------------------------------------------------------------------------


@router.put("/clip/add-phrase",
            response_class=PrettyJSONResponse,
            description="Adds a phrase to the clip server")
def add_phrase(request: Request,
               phrase : str):

    return http_clip_server_add_phrase(phrase)

class AddClipPhraseResponse(BaseModel):
    clip_vector: str

@router.post("/clip/phrases",
             description="Adds a phrase to the clip server.",
             tags=["clip"],
             response_model=StandardSuccessResponse[AddClipPhraseResponse],
             status_code=201,
             responses=ApiResponseHandler.listErrors([400, 500, 503]))
def add_phrase(request: Request, response: Response, phrase_data: PhraseModel):
    response_handler = ApiResponseHandler(request)

    try:
        if not phrase_data.phrase:
            return response_handler.create_error_response(ErrorCode.INVALID_PARAMS, "Phrase is required", status.HTTP_400_BAD_REQUEST)

        # Use the phrase from the PhraseModel object
        response = http_clip_server_add_phrase(phrase_data.phrase)

        if not (200 <= response.status_code < 300):
            return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Clip server error", status.HTTP_503_SERVICE_UNAVAILABLE)

        # Assuming the response includes the clip vector or is None
        clip_vector = response.get("clip_vector") if response else None
        return response_handler.create_success_response({"clip_vector": clip_vector}, http_status_code=201, headers={"Cache-Control": "no-store"})

    except Exception as e:
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal server error", status.HTTP_500_INTERNAL_SERVER_ERROR)


@router.get("/clip/clip-vector",
            response_class=PrettyJSONResponse,
            description="Gets a clip vector of a specific phrase")
def add_phrase(request: Request,
               phrase : str):

    return http_clip_server_clip_vector_from_phrase(phrase)

@router.get("/clip/vectors/{phrase}", response_class=PrettyJSONResponse, tags=["clip"], summary="Get Clip Vector for a Phrase", description="Retrieves the clip vector for a given phrase.")
def get_clip_vector(request: Request, phrase: str):
    response_handler = ApiResponseHandler(request)
    try:
        vector = http_clip_server_clip_vector_from_phrase(phrase)

        if vector is None:
            return response_handler.create_error_response(ErrorCode.ELEMENT_NOT_FOUND, "Phrase not found", status.HTTP_404_NOT_FOUND)

        return response_handler.create_success_response({"vector": vector}, http_status_code=200, headers={"Cache-Control": "no-store"})

    except Exception as e:
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Internal server error", status.HTTP_500_INTERNAL_SERVER_ERROR)


@router.get("/clip/random-image-similarity-threshold",
            response_class=PrettyJSONResponse,
            description="Gets a random image from a dataset with a cosine similarity threshold")
def random_image_similarity_threshold(request: Request,
                                    dataset : str,
                                    phrase : str,
                                    similarity_threshold : float=0,
                                    max_tries : int=50):

    # maximum number of tries
    # repeat n times until we find an image
    for try_index in range(0, max_tries):
        # Use $sample to get one random document
        jobs = request.app.completed_jobs_collection.aggregate([
            {"$match": {"task_input_dict.dataset": dataset}},
            {"$sample": {"size": 1}}
        ])

        # Convert cursor type to list
        jobs = list(jobs)

        # Ensure the list isn't empty (this is just a safety check)
        if not jobs:
            raise HTTPException(status_code=404, detail="No image found for the given dataset")

        # Remove the auto generated _id field from the document
        jobs[0].pop('_id', None)
        this_job = jobs[0]

        output_file_dictionary = this_job["task_output_file_dict"]
        image_path = output_file_dictionary['output_file_path']

        # remove the datasets/ prefix
        image_path = image_path.replace("datasets/", "")

        similarity_score = http_clip_server_get_cosine_similarity(image_path, phrase)

        if similarity_score is None:
            continue

        if similarity_score >= similarity_threshold:
            result = {
                'image' : this_job,
                'similarity_score' : similarity_score
            }
            return result

    return None


@router.get("/clip/random-image-list-similarity-threshold",
            response_class=PrettyJSONResponse,
            description="Gets a random image from a dataset with a cosine similarity threshold")
def random_image_list_similarity_threshold(request: Request,
                          dataset: str,
                          phrase: str,
                          similarity_threshold: float = 0,
                          size: int = 20):
    # Use Query to get the dataset and size from query parameters

    distinct_jobs = []
    tried_ids = set()

    nb_tries = 0
    while nb_tries < size:
        # Use $sample to get 'size' random documents
        jobs = request.app.completed_jobs_collection.aggregate([
            {"$match": {"task_input_dict.dataset": dataset, "_id": {"$nin": list(tried_ids)}}},
            # Exclude already tried ids
            {"$sample": {"size": size - len(distinct_jobs)}}  # Only fetch the remaining needed size
        ])

        # Convert cursor type to list
        jobs = list(jobs)
        distinct_jobs.extend(jobs)

        # Store the tried image ids
        tried_ids.update([job["_id"] for job in jobs])

        # Ensure only distinct images are retained
        seen = set()
        distinct_jobs = [doc for doc in distinct_jobs if doc["_id"] not in seen and not seen.add(doc["_id"])]
        nb_tries = nb_tries + 1

    result_jobs = []

    for job in distinct_jobs:
        job.pop('_id', None)  # remove the auto generated field

        this_job = job

        output_file_dictionary = this_job["task_output_file_dict"]
        image_path = output_file_dictionary['output_file_path']

        # remove the datasets/ prefix
        image_path = image_path.replace("datasets/", "")

        similarity_score = http_clip_server_get_cosine_similarity(image_path, phrase)

        if similarity_score is None:
            continue

        if similarity_score >= similarity_threshold:

            result = {
                'image': this_job,
                'similarity_score': similarity_score
            }
            result_jobs.append(result)

    # Return the jobs as a list in the response

    return result_jobs

@router.get("/image/get_random_image_similarity_by_date_range", response_class=PrettyJSONResponse)
def get_random_image_similarity_date_range(
    request: Request,
    dataset: str = None,
    phrase: str = "",
    similarity_threshold: float = 0,
    start_date: str = None,
    end_date: str = None,
    size: int = None,
    prompt_generation_policy: Optional[str] = None  # Optional query parameter
):
    print('begin')
    query = {
        'task_input_dict.dataset': dataset
    }

    if start_date and end_date:
        query['task_creation_time'] = {
            '$gte': start_date,
            '$lte': end_date
        }
    elif start_date:
        query['task_creation_time'] = {
            '$gte': start_date
        }
    elif end_date:
        query['task_creation_time'] = {
            '$lte': end_date
        }

    # Include prompt_generation_policy in the query if provided
    if prompt_generation_policy:
        query['task_input_dict.prompt_generation_policy'] = prompt_generation_policy

    aggregation_pipeline = [{"$match": query}]
    if size:
        aggregation_pipeline.append({"$sample": {"size": size}})

    jobs = request.app.completed_jobs_collection.aggregate(aggregation_pipeline)
    jobs = list(jobs)

    image_path_list = []
    for job in jobs:
        job.pop('_id', None)  # Remove the auto-generated field

        this_job = job
        output_file_dictionary = this_job["task_output_file_dict"]
        image_path = output_file_dictionary['output_file_path']

        # remove the datasets/ prefix
        image_path = image_path.replace("datasets/", "")

        image_path_list.append(image_path)

    similarity_score_list = http_clip_server_get_cosine_similarity_list(image_path_list, phrase)

    print(similarity_score_list)

    if similarity_score_list is None:
        return {
            "images" : []
        }

    # make sure the similarity list is the correct format
    if 'similarity_list' not in similarity_score_list:
        return {
            "images": []
        }

    similarity_score_list = similarity_score_list['similarity_list']

    num_images = len(jobs)

    # make sure the list returned
    if num_images != len(similarity_score_list):
        return {
            "images": []
        }

    # filter the images by similarity threshold
    filtered_images = []
    for i in range(num_images):
        image_similarity_score = similarity_score_list[i]
        job = jobs[i]

        if image_similarity_score >= similarity_threshold:
            job["similarity_score"] = image_similarity_score
            filtered_images.append(job)

    return {
        "images": filtered_images
    }

@router.get("/check-clip-server-status")
def check_clip_server_status():
    try:
        # Send a simple GET request to the clip server
        response = requests.get(CLIP_SERVER_ADDRESS )

        # Check if the response status code is 200 (OK)
        if response.status_code == 200:
            return {"status": "online", "message": "Clip server is online."}
        else:
            return {"status": "offline", "message": "Clip server is offline. Received unexpected response."}

    except requests.exceptions.RequestException as e:
        # Handle any exceptions that occur during the request
        print(f"Error checking clip server status: {e}")
        return {"status": "offline", "message": "Clip server is offline or unreachable."}

@router.get("/clip/server-status", tags=["clip"], description="Checks the status of the CLIP server.")
def check_clip_server_status(request: Request):
    response_handler = ApiResponseHandler(request)
    try:
        response = requests.get(CLIP_SERVER_ADDRESS)
        reachable = response.status_code == 200
        return response_handler.create_success_response({"reachable": reachable}, http_status_code=200, headers={"Cache-Control": "no-store"})
    except requests.exceptions.RequestException as e:
        return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "CLIP server is not reachable", 503)

