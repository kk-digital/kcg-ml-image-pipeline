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
                                    similarity_threshold : float,
                                    max_tries=50):

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
        print(this_job)
        print(output_file_dictionary)

        # Return the image in the response
        return []

