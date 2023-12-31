import requests
import json

SERVER_ADDRESS = 'http://192.168.3.1:8111'


def http_get_list_completed_jobs():
    url = SERVER_ADDRESS + "/queue/image-generation/list-completed"

    try:
        response = requests.get(url)

        if response.status_code == 200:
            job_json = response.json()
            return job_json

    except Exception as e:
        print('request exception ', e)

    return None

# Get request to get sequential id of a dataset
def http_get_sequential_id(dataset_name: str, limit: int):
    url = SERVER_ADDRESS + "/dataset/sequential-id/{0}?limit={1}".format(dataset_name, limit)

    try:
        response = requests.get(url)
    except Exception as e:
        print('request exception ', e)

    if response.status_code == 200:
        job_json = response.json()
        return job_json

    return None


def http_add_model(model_card):
    url = SERVER_ADDRESS + "/models/add"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data

    try:
        response = requests.post(url, data=model_card, headers=headers)

        if response.status_code != 200:
            print(f"request failed with status code: {response.status_code}")
        print("model_id=", response.content)
        return response.content
    except Exception as e:
        print('request exception ', e)

    return None


def http_get_model_id(model_hash):
    url = SERVER_ADDRESS + "/models/get-id?model_hash={}".format(model_hash)
    try:
        response = requests.get(url)

        if response.status_code != 200:
            print(f"request failed with status code: {response.status_code}")

        return int(response.content)
    except Exception as e:
        print('request exception ', e)

    return None


def http_add_score(score_data):
    url = SERVER_ADDRESS + "/score/set-image-rank-score"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data

    try:
        response = requests.post(url, json=score_data, headers=headers)

        if response.status_code != 200:
            print(f"request failed with status code: {response.status_code}: {str(response.content)}")
    except Exception as e:
        print('request exception ', e)

    return None


def http_add_sigma_score(sigma_score_data):
    url = SERVER_ADDRESS + "/sigma-score/set-image-rank-sigma-score"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data

    try:
        response = requests.post(url, json=sigma_score_data, headers=headers)

        if response.status_code != 200:
            print(f"request failed with status code: {response.status_code}: {str(response.content)}")
    except Exception as e:
        print('request exception ', e)

    return None


def http_add_residual(residual_data):
    url = SERVER_ADDRESS + "/residual/set-image-rank-residual"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data

    try:
        response = requests.post(url, json=residual_data, headers=headers)

        if response.status_code != 200:
            print(f"request failed with status code: {response.status_code}: {str(response.content)}")
    except Exception as e:
        print('request exception ', e)

    return None


def http_add_percentile(percentile_data):
    url = SERVER_ADDRESS + "/percentile/set-image-rank-percentile"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data

    try:
        response = requests.post(url, json=percentile_data, headers=headers)

        if response.status_code != 200:
            print(f"request failed with status code: {response.status_code}: {str(response.content)}")
    except Exception as e:
        print('request exception ', e)

    return None


def http_add_residual_percentile(residual_percentile_data):
    url = SERVER_ADDRESS + "/residual-percentile/set-image-rank-residual-percentile"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data

    try:
        response = requests.post(url, json=residual_percentile_data, headers=headers)

        if response.status_code != 200:
            print(f"request failed with status code: {response.status_code}: {str(response.content)}")
    except Exception as e:
        print('request exception ', e)

    return None


# Get list of all dataset names
def http_get_dataset_names():
    url = SERVER_ADDRESS + "/dataset/list"
    try:
        response = requests.get(url)

        if response.status_code == 200:
            data_json = response.json()
            return data_json

    except Exception as e:
        print('request exception ', e)

    return None


# Get completed job
def http_get_completed_job_by_image_hash(image_hash):
    url = SERVER_ADDRESS + "/job/get-completed-job-by-hash?image_hash={}".format(image_hash)
    try:
        response = requests.get(url)

        if response.status_code == 200:
            data_json = response.json()
            return data_json

    except Exception as e:
        print('request exception ', e)

    return None


def http_add_score_attributes(img_hash,
                              image_clip_score,
                              image_clip_percentile,
                              image_clip_sigma_score,
                              text_embedding_score,
                              text_embedding_percentile,
                              text_embedding_sigma_score,
                              delta_sigma_score):
    endpoint = "/job/add-attributes?image_hash={}&image_clip_score={}&image_clip_percentile={}&image_clip_sigma_score={}&text_embedding_score={}&text_embedding_percentile={}&text_embedding_sigma_score={}&delta_sigma_score={}".format(
        img_hash,
        image_clip_score,
        image_clip_percentile,
        image_clip_sigma_score,
        text_embedding_score,
        text_embedding_percentile,
        text_embedding_sigma_score,
        delta_sigma_score)
    url = SERVER_ADDRESS + endpoint

    try:
        response = requests.put(url)

        if response.status_code != 200:
            print(f"request failed with status code: {response.status_code}: {str(response.content)}")
    except Exception as e:
        print('request exception ', e)

    return None

# Get completed job
def http_get_completed_job_by_uuid(job_uuid):
    url = SERVER_ADDRESS + "/job/get-job/{}".format(job_uuid)
    try:
        response = requests.get(url)

        if response.status_code == 200:
            data_json = response.json()
            return data_json

    except Exception as e:
        print('request exception ', e)

    return None

# Get completed jobs
def http_get_completed_jobs_by_uuids(job_uuids):
    count = 0
    batch_uuids = ""
    for uuid in job_uuids:
        if count!=0:
            batch_uuids += "&uuids="

        batch_uuids += "{}".format(uuid)
        count += 1

    url = SERVER_ADDRESS + "/job/get-jobs?uuids={}".format(batch_uuids)
    try:
        response = requests.get(url)

        if response.status_code == 200:
            data_json = response.json()
            return data_json

    except Exception as e:
        print('request exception ', e)

    return None