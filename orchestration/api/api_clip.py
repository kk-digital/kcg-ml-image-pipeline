from fastapi import Request, APIRouter, HTTPException
import requests

CLIP_SERVER_ADRESS = 'http://127.0.0.1:8002'

router = APIRouter()


# --------- Http requests -------------
def http_clip_server_add_phrase(phrase: str):
    url = CLIP_SERVER_ADRESS + "/add-phrase?phrase=" + phrase

    try:
        response = requests.put(url)

        if response.status_code == 200:
            result_json = response.json()
            return result_json

    except Exception as e:
        print('request exception ', e)

    return None


def http_clip_server_clip_vector_from_phrase(phrase: str):
    url = CLIP_SERVER_ADRESS + "/clip-vector?phrase=" + phrase

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
    url = f'{CLIP_SERVER_ADRESS}/cosine-similarity?image_path={image_path}&phrase={phrase}'

    try:
        response = requests.get(url)

        if response.status_code == 200:
            result_json = response.json()
            return result_json

    except Exception as e:
        print('request exception ', e)

    return None

# ----------------------------------------------------------------------------


@router.put("/clip/add-phrase", description="Adds a phrase to the clip server")
def add_phrase(request: Request,
               phrase : str):

    return http_clip_server_add_phrase(phrase)


@router.get("/clip/clip-vector", description="Gets a clip vector of a specific phrase")
def add_phrase(request: Request,
               phrase : str):

    return http_clip_server_clip_vector_from_phrase(phrase)


@router.get("/clip/random-image-similarity-threshold", description="Gets a random image from a dataset with a cosine similarity threshold")
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
            return this_job

    return None


@router.get("/clip/random-image-list-similarity-threshold", description="Gets a random image from a dataset with a cosine similarity threshold")
def random_image_list_similarity_threshold(request: Request,
                          dataset: str,
                          phrase: str,
                          similarity_threshold: float = 0,
                          size: int = 20):
    # Use Query to get the dataset and size from query parameters

    print('random_image_list_similarity_threshold')

    distinct_jobs = []
    tried_ids = set()
    nb_tries = 0
    while nb_tries < size:
        print(size)

        sample_size = request.app.completed_jobs_collection.count_documents(
            {"task_input_dict.dataset": dataset})

        print(size)
        sample_size = request.app.completed_jobs_collection.count_documents(
            {"task_input_dict.dataset": dataset, "_id": {"$nin": list(tried_ids)}})

        sample_size = min(sample_size, size)

        print(sample_size)
        # Use $sample to get 'size' random documents
        jobs = request.app.completed_jobs_collection.aggregate([
            {"$match": {"task_input_dict.dataset": dataset, "_id": {"$nin": list(tried_ids)}}},
            # Exclude already tried ids
            {"$sample": {"size": sample_size}}  # Only fetch the remaining needed size
        ])

        # Convert cursor type to list
        jobs = list(jobs)
        distinct_jobs.extend(jobs)

        # Store the tried image ids
        tried_ids.update([job["_id"] for job in jobs])

        # Ensure only distinct images are retained
        seen = set()
        distinct_jobs = [doc for doc in distinct_jobs if doc["_id"] not in seen and not seen.add(doc["_id"])]
        print(nb_tries)
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
            result_jobs.append(this_job)

    # Return the jobs as a list in the response
    return result_jobs
