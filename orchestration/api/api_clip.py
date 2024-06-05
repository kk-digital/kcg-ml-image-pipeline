from fastapi import Request, APIRouter, HTTPException, Response, File, UploadFile
import requests
from .api_utils import PrettyJSONResponse, ApiResponseHandler, ErrorCode, StandardErrorResponse, StandardErrorResponseV1, StandardSuccessResponse, StandardSuccessResponseV1, RechableResponse, GetClipPhraseResponse, ApiResponseHandlerV1, GetKandinskyClipResponse, UrlResponse
from orchestration.api.mongo_schemas import  PhraseModel, ListSimilarityScoreTask
from typing import Optional
from typing import List
import json
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi_cache.decorator import cache
from pydantic import BaseModel
import traceback
from utility.minio import cmd 
from minio import Minio
from minio.error import S3Error
from .api_utils import find_or_create_next_folder_and_index
import os
import io
from fastapi import Query
from PIL import Image

CLIP_SERVER_ADDRESS = 'http://192.168.3.31:8002'
#CLIP_SERVER_ADDRESS = 'http://127.0.0.1:8002'
router = APIRouter()

# --------- Http requests -------------
def http_clip_server_get_kandinsky_vector(image_path: str):
    url = CLIP_SERVER_ADDRESS + "/kandinsky-clip-vector?image_path=" + image_path
    response = None
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
    
    except Exception as e:
        print('request exception ', e)
    
    finally:
        if response:
            response.close()
    
    return None


def http_clip_server_add_phrase(phrase: str):
    url = CLIP_SERVER_ADDRESS + "/add-phrase?phrase=" + phrase
    response = None
    try:
        response = requests.post(url)
        if response.status_code == 200:
            return response.status_code, response.json()
        else:
            return response.status_code, None
    
    except Exception as e:
        print('request exception ', e)
        # Return a 503 status code when the server is not accessible
        return 500, None
    
    finally:
        if response:
            response.close()


def http_clip_server_clip_vector_from_phrase(phrase: str):
    url = CLIP_SERVER_ADDRESS + "/get-clip-vector?phrase=" + phrase
    response = None
    try:
        response = requests.get(url)

        if response.status_code == 200:
            result_json = response.json()
            return result_json

    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()

    return None


def http_clip_server_get_cosine_similarity(image_path: str,
                                           phrase: str):
    url = f'{CLIP_SERVER_ADDRESS}/cosine-similarity?image_path={image_path}&phrase={phrase}'
    response = None
    try:
        response = requests.get(url)

        if response.status_code == 200:
            result_json = response.json()
            return result_json

    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()

    return None

def http_clip_server_get_cosine_similarity_list(image_path_list: List[str], phrase: str):
    url = f'{CLIP_SERVER_ADDRESS}/cosine-similarity-list?phrase={phrase}'
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data
    
    # Use json.dumps to convert the list to a JSON-formatted string and print it
    json_string = json.dumps(image_path_list)
    print("Sending JSON string:", json_string)

    try:
        response = requests.post(url, json=image_path_list, headers=headers)
        print(f"Response status code: {response.status_code}")  # Print the status code of the response

        # Print the entire response text to debug what the server actually returned
        print("Response text:", response.text)

        if response.status_code == 200:
            result_json = response.json()
            print("Response JSON:", result_json)  # Print the parsed JSON response
            return result_json
        else:
            print(f"Error: Unexpected response status: {response.status_code}")
            return None  # Optionally return None explicitly here for clarity

    except Exception as e:
        print('Request exception:', e)

    finally:
        if response:
            response.close()
    return None

# ----------------------------------------------------------------------------



@router.get("/clip/random-image-similarity-threshold",
            response_class=PrettyJSONResponse,
            tags = ["deprecated3"],
            description="changed with /clip/get-random-images-with-clip-search ")
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
            tags = ["deprecated3"],
            description="changed with /clip/get-random-images-with-clip-search ")
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

@router.get("/image/get_random_image_similarity_by_date_range", 
            tags = ["deprecated3"],
            description="changed with /clip/get-random-images-with-clip-search ")
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
        query['prompt_generation_data.prompt_generation_policy'] = prompt_generation_policy

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
        print("similarity_score_list is empty")
        return {
            "images" : []
        }

    # make sure the similarity list is the correct format
    if 'similarity_list' not in similarity_score_list:
        print("'similarity_list' not in similarity_score_list")
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



#  Apis with new names and reponse format
    
@router.get("/clip/get-kandinsky-clip-vector", 
            tags=["clip"], 
            status_code=200, 
            response_model=StandardSuccessResponseV1[GetKandinskyClipResponse],
            responses=ApiResponseHandlerV1.listErrors([404,422,500]),
            description="Get kandinsky Vector for a image")
def get_clip_vector_from_phrase(request: Request, image_path: str):
    
    response_handler = ApiResponseHandlerV1(request)
    try:
        vector = http_clip_server_get_kandinsky_vector(image_path)
        
        if vector is None:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="image_path not found",
                http_status_code=404,
            )

        # Directly access the first element if vector is not empty and is a list of lists
        if vector and isinstance(vector, list) and all(isinstance(elem, list) for elem in vector):
            features_vector = vector[0]
        else:
            features_vector = vector  # Fallback if the structure is different

        return response_handler.create_success_response_v1(
            response_data= features_vector, 
            http_status_code=200,
        )

    except Exception as e:
        print(f"Exception occurred: {e}")  # Print statement 5
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string="Clip server error",
            http_status_code = 500, 

        )


@router.post("/clip/add-phrase",
             description="Adds a phrase to the clip server.",
             response_model=StandardSuccessResponseV1[None],
             tags=["clip"],
             responses=ApiResponseHandlerV1.listErrors([400, 422, 500, 503]))
def add_phrase_v1(request: Request, phrase: str):
    response_handler = ApiResponseHandlerV1(request)

    try:
        if not phrase:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS, 
                error_string="Phrase is required", 
                http_status_code=400,
    
            )

        status_code, _ = http_clip_server_add_phrase(phrase)  

        if 200 <= status_code < 300:
            return response_handler.create_success_response_v1(
                response_data=None, 
                http_status_code=201, 
    
            )
        else:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.OTHER_ERROR,
                error_string="Clip server error", 
                http_status_code=500,        
    
            )

    except Exception as e:
        traceback.print_exc()
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string="Internal server error", 
            http_status_code=500,

        )

@router.get("/clip/get-clip-vector", tags=["clip"], 
            response_model=StandardSuccessResponseV1[GetClipPhraseResponse], 
            status_code=200, 
            responses=ApiResponseHandlerV1.listErrors([400,404,422,500]), 
            description="Get Clip Vector for a Phrase")
def get_clip_vector_from_phrase(request: Request, phrase: str):
    
    response_handler = ApiResponseHandlerV1(request)
    try:
       
        vector = http_clip_server_clip_vector_from_phrase(phrase)
        
        if vector is None:

            return response_handler.create_error_response_v1(
                ErrorCode.ELEMENT_NOT_FOUND,
                "Phrase not found",
                http_status_code=404,
    
            )

        return response_handler.create_success_response_v1(
            response_data= vector, 
            http_status_code=200, 
 
        )

    except Exception as e:
        print(f"Exception occurred: {e}")  # Print statement 5
        return response_handler.create_error_response_v1(
            ErrorCode.OTHER_ERROR, 
            "Internal server error", 
            http_status_code = 500, 

        )


@router.get("/clip/get-server-status", 
            tags=["clip"], 
            response_model=StandardSuccessResponseV1[RechableResponse],  
            status_code=200,  
            responses=ApiResponseHandlerV1.listErrors([503]),  # Adapt to use ApiResponseHandlerV1
            description="Checks the status of the CLIP server")
def check_clip_server_status(request: Request):
    response_handler = ApiResponseHandlerV1(request)  
    try:
        # Update the URL to include '/docs'
        response = requests.get(CLIP_SERVER_ADDRESS + "/docs")
        reachable = response.status_code == status.HTTP_200_OK  
        # Adjust the success response creation to match the new handler
        return response_handler.create_success_response_v1(
            response_data={"reachable": reachable},  
            http_status_code=200,    
 
        )
    except requests.exceptions.RequestException as e:
        # Print statement for debugging
        print(f"Exception occurred: {e}")  
        # Adjust the error response creation to match the new handler
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string="CLIP server is not reachable", 
            http_status_code=503, 

        )



bucket_name = "datasets"
base_folder = "external-images"
    

@router.post("/upload-images",
             status_code=201,
             response_model=StandardSuccessResponseV1[UrlResponse],  # Adjusted for list of URLs response
             responses=ApiResponseHandlerV1.listErrors([400, 422, 500]),
             description="Upload multiple images on minio")
async def upload_images(request: Request, 
                          files: List[UploadFile] = File(...), 
                          check_size: bool = Query(True, description="Check if images are 512x512")):
    response_handler = ApiResponseHandlerV1(request)
    uploaded_files_paths = []

    for file in files:
        # Initialize MinIO client for each file, assuming credentials remain constant
        minio_client = cmd.get_minio_client(minio_access_key="v048BpXpWrsVIHUfdAix", minio_secret_key="4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu")
        
        # Extract the file extension and prepare the file path
        _, file_extension = os.path.splitext(file.filename)
        file_extension = file_extension.lower()
        next_folder, next_index = find_or_create_next_folder_and_index(minio_client, bucket_name, base_folder)
        file_name = f"{next_index:06}{file_extension}"
        file_path = f"{next_folder}/{file_name}"

        try:
            await file.seek(0)
            content = await file.read()

            # Optional: Perform size check if check_size is True
            if check_size:
                image = Image.open(io.BytesIO(content))
                if image.size != (512, 512):
                    continue  # Skip uploading this file, or handle differently

            content_stream = io.BytesIO(content)
            cmd.upload_data(minio_client, bucket_name, file_path, content_stream)
            uploaded_files_paths.append(f"{bucket_name}/{file_path}")
        except Exception as e:
            print(f"Exception occurred while processing {file.filename}: {e}")
            continue  # Optionally, log or handle individual file upload exceptions

    if not uploaded_files_paths:
        # Handle the case where no files were uploaded successfully
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string="No images were uploaded successfully",
            http_status_code=500,
        )

    # Return the paths of successfully uploaded files
    return response_handler.create_success_response_v1(
        response_data=uploaded_files_paths,
        http_status_code=201
    )         

@router.get("/clip/get-random-images-with-clip-search",
            tags=["clip"],
            description="Gets as many random images as set in the size param, scores each image with clip according to the value of the 'phrase' param and then returns the list sorted by the similarity score. NOTE: before using this endpoint, make sure to register the phrase using the '/clip/add-phrase' endpoint.",
            response_model=StandardSuccessResponseV1[ListSimilarityScoreTask],
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
async def get_random_image_similarity_date_range(
    request: Request,
    dataset: str = Query(..., description="Dataset to filter images"),
    phrase: str = Query(..., description="Phrase to compare similarity with"),
    similarity_threshold: float = Query(0, description="Minimum similarity threshold"),
    start_date: str = None,
    end_date: str = None,
    size: int = Query(..., description="Number of random images to return"),
    prompt_generation_policy: Optional[str] = Query(None, description="Optional prompt generation policy")
):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        query = {
            'task_input_dict.dataset': dataset
        }

        if start_date and end_date:
            query['task_creation_time'] = {'$gte': start_date, '$lte': end_date}
        elif start_date:
            query['task_creation_time'] = {'$gte': start_date}
        elif end_date:
            query['task_creation_time'] = {'$lte': end_date}

        # Include prompt_generation_policy in the query if provided
        if prompt_generation_policy:
            query['prompt_generation_data.prompt_generation_policy'] = prompt_generation_policy

        aggregation_pipeline = [{"$match": query}]
        if size:
            aggregation_pipeline.append({"$sample": {"size": size}})

        jobs = list(request.app.completed_jobs_collection.aggregate(aggregation_pipeline))

        image_path_list = []
        for job in jobs:
            job.pop('_id', None)  # Remove the auto-generated field
            output_file_dictionary = job["task_output_file_dict"]
            image_path = output_file_dictionary['output_file_path'].replace("datasets/", "")
            image_path_list.append(image_path)

        similarity_score_list = http_clip_server_get_cosine_similarity_list(image_path_list, phrase)

        if similarity_score_list is None or 'similarity_list' not in similarity_score_list:
            return response_handler.create_success_response_v1(response_data={"images": []}, http_status_code=200)

        similarity_score_list = similarity_score_list['similarity_list']

        if len(jobs) != len(similarity_score_list):
            return response_handler.create_success_response_v1(response_data={"images": []}, http_status_code=200)

        filtered_images = []
        for i in range(len(jobs)):
            image_similarity_score = similarity_score_list[i]
            job = jobs[i]

            if image_similarity_score >= similarity_threshold:
                job["similarity_score"] = image_similarity_score
                filtered_images.append(job)

        return response_handler.create_success_response_v1(response_data={"images": filtered_images}, http_status_code=200)

    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )
