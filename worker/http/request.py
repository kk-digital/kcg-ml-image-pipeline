# NOTE: don't add more imports here
# this is also used by training workers
import requests

SERVER_ADRESS = 'http://123.176.98.90:8764'
#SERVER_ADRESS = 'http://127.0.0.1:8000'


# Get request to get an available job
def http_get_job(worker_type: str = None):
    url = SERVER_ADRESS + "/queue/image-generation/get-job"
    if worker_type is not None:
        url = url + "?task_type={}".format(worker_type)

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
    url = SERVER_ADRESS + "/dataset/sequential-id/{0}?limit={1}".format(dataset_name, limit)

    try:
        response = requests.get(url)
    except Exception as e:
        print('request exception ', e)

    if response.status_code == 200:
        job_json = response.json()
        return job_json

    return None


def http_add_job(job):
    url = SERVER_ADRESS + "/queue/image-generation/add"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data

    try:
        response = requests.post(url, json=job, headers=headers)
    except Exception as e:
        print('request exception ', e)

    if response.status_code != 201 and response.status_code != 200:
        print(f"POST request failed with status code: {response.status_code}")

    return response.content.decode()


def http_update_job_completed(job):
    url = SERVER_ADRESS + "/queue/image-generation/update-completed"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data

    try:
        response = requests.put(url, json=job, headers=headers)
    except Exception as e:
        print('request exception ', e)

    if response.status_code != 200:
        print(f"request failed with status code: {response.status_code}")


def http_update_job_failed(job):
    url = SERVER_ADRESS + "/queue/image-generation/update-failed"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data

    try:
        response = requests.put(url, json=job, headers=headers)
    except Exception as e:
        print('request exception ', e)

    if response.status_code != 200:
        print(f"request failed with status code: {response.status_code}")


# Get list of all dataset names
def http_get_dataset_names():
    url = SERVER_ADRESS + "/dataset/list"
    try:
        response = requests.get(url)

        if response.status_code == 200:
            data_json = response.json()
            return data_json

    except Exception as e:
        print('request exception ', e)

    return None