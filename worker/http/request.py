import requests

SERVER_ADRESS = 'http://192.168.3.1:8111'


# Get request to get an available job
def http_get_job(worker_type: str = None):
    url = SERVER_ADRESS + "/get-job"
    if worker_type is not None:
        url = url + "?task_type={}".format(worker_type)

    response = requests.get(url)

    if response.status_code == 200:
        job_json = response.json()
        return job_json

    return None


# Get request to get sequential id of a dataset
def http_get_sequential_id(dataset_name: str, limit: int):
    url = SERVER_ADRESS + "/get-sequential-id/{0}?limit={1}".format(dataset_name, limit)
    response = requests.get(url)

    if response.status_code == 200:
        job_json = response.json()
        return job_json

    return None


# Used for debugging purpose
# The worker should not be adding jobs
def http_add_job(job):
    url = SERVER_ADRESS + "/add-job"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data
    response = requests.post(url, json=job, headers=headers)

    if response.status_code != 201 and response.status_code != 200:
        print(f"POST request failed with status code: {response.status_code}")


def http_update_job_completed(job):
    url = SERVER_ADRESS + "/update-job-completed"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data

    response = requests.put(url, json=job, headers=headers)

    if response.status_code != 200:
        print(f"request failed with status code: {response.status_code}")


def http_update_job_failed(job):
    url = SERVER_ADRESS + "/update-job-failed"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data

    response = requests.put(url, json=job, headers=headers)
    if response.status_code != 200:
        print(f"request failed with status code: {response.status_code}")
