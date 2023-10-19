import requests

SERVER_ADRESS = '192.168.3.1:8111'


# Get request to get an available job
def http_get_job(worker_type: str = None, minio_ip_addr=None):
    if minio_ip_addr is None:
        minio_ip_addr=SERVER_ADRESS

    url = 'http://'+ minio_ip_addr + "/queue/image-generation/get-job"
    if worker_type is not None:
        url = url + "?task_type={}".format(worker_type)

    try:
        response = requests.get(url)
    except Exception as e:
        print('request exception ', e)

    if response.status_code == 200:
        job_json = response.json()
        return job_json

    return None


# Get request to get sequential id of a dataset
def http_get_sequential_id(dataset_name: str, limit: int, minio_ip_addr=None):
    if minio_ip_addr is None:
        minio_ip_addr=SERVER_ADRESS

    url = 'http://'+ SERVER_ADRESS + "/dataset/sequential-id/{0}?limit={1}".format(dataset_name, limit)

    try:
        response = requests.get(url)
    except Exception as e:
        print('request exception ', e)

    if response.status_code == 200:
        job_json = response.json()
        return job_json

    return None


def http_add_job(job, minio_ip_addr= None):
    if minio_ip_addr is None:
        minio_ip_addr=SERVER_ADRESS

    url = 'http://' + SERVER_ADRESS + "/queue/image-generation/add"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data

    try:
        response = requests.post(url, json=job, headers=headers)
    except Exception as e:
        print('request exception ', e)

    if response.status_code != 201 and response.status_code != 200:
        print(f"POST request failed with status code: {response.status_code}")


def http_update_job_completed(job, minio_ip_addr=None):
    if minio_ip_addr is None:
        minio_ip_addr=SERVER_ADRESS

    url = 'http://' + SERVER_ADRESS + "/queue/image-generation/update-completed"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data

    try:
        response = requests.put(url, json=job, headers=headers)
    except Exception as e:
        print('request exception ', e)

    if response.status_code != 200:
        print(f"request failed with status code: {response.status_code}")


def http_update_job_failed(job, minio_ip_addr=None):
    if minio_ip_addr is None:
        minio_ip_addr=SERVER_ADRESS

    url = 'http://' + SERVER_ADRESS + "/queue/image-generation/update-failed"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data

    try:
        response = requests.put(url, json=job, headers=headers)
    except Exception as e:
        print('request exception ', e)

    if response.status_code != 200:
        print(f"request failed with status code: {response.status_code}")
