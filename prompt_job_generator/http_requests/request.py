import requests

SERVER_ADRESS = 'http://192.168.3.1:8111'
#SERVER_ADRESS = 'http://127.0.0.1:8000'


def http_get_completed_jobs_count(dataset_name: str):
    url = SERVER_ADRESS + "/queue/image-generation/count-completed?dataset=" + dataset_name

    try:
        response = requests.get(url)

        if response.status_code == 200:
            job_json = response.json()
            return job_json

    except Exception as e:
        print('request exception ', e)

    return None


def http_get_pending_jobs_count(dataset_name: str):
    url = SERVER_ADRESS + "/queue/image-generation/count-pending?dataset=" + dataset_name

    try:
        response = requests.get(url)

        if response.status_code == 200:
            job_json = response.json()
            return job_json

    except Exception as e:
        print('request exception ', e)

    return None


def http_get_in_progress_jobs_count(dataset_name: str):
    url = SERVER_ADRESS + "/queue/image-generation/count-in-progress?dataset=" + dataset_name

    try:
        response = requests.get(url)

        if response.status_code == 200:
            job_json = response.json()
            return job_json

    except Exception as e:
        print('request exception ', e)

    return None


def http_get_dataset_list():
    url = SERVER_ADRESS + "/dataset/list"

    try:
        response = requests.get(url)

        if response.status_code == 200:
            job_json = response.json()
            return job_json

    except Exception as e:
        print('request exception ', e)

    return None

def http_get_dataset_rate(dataset_name: str):
    url = SERVER_ADRESS + f"/dataset/get-rate?dataset={dataset_name}"

    try:
        response = requests.get(url)

        if response.status_code == 200:
            job_json = response.json()
            return job_json

    except Exception as e:
        print('request exception ', e)

    return None


def http_get_all_dataset_rate():
    url = SERVER_ADRESS + f"/dataset/get-all-dataset-rate"

    try:
        response = requests.get(url)

        if response.status_code == 200:
            job_json = response.json()
            return job_json

    except Exception as e:
        print('request exception ', e)

    return None


def http_get_all_dataset_config():
    url = SERVER_ADRESS + f"/dataset/get-all-dataset-config"

    try:
        response = requests.get(url)

        if response.status_code == 200:
            job_json = response.json()
            return job_json

    except Exception as e:
        print('request exception ', e)

    return None


def http_get_dataset_generation_policy(dataset : str):
    url = SERVER_ADRESS + f"/dataset/get-generation-policy?dataset={dataset}"

    try:
        response = requests.get(url)

        if response.status_code == 200:
            job_json = response.json()
            return job_json

    except Exception as e:
        print('request exception ', e)

    return None


def http_get_all_dataset_generation_policy():
    url = SERVER_ADRESS + f"/dataset/get-all-dataset-generation-policy"

    try:
        response = requests.get(url)

        if response.status_code == 200:
            job_json = response.json()
            return job_json

    except Exception as e:
        print('request exception ', e)

    return None


def http_get_dataset_top_k_value(dataset : str):
    url = SERVER_ADRESS + f"/dataset/get-top-k?dataset={dataset}"

    try:
        response = requests.get(url)

        if response.status_code == 200:
            job_json = response.json()
            return job_json

    except Exception as e:
        print('request exception ', e)

    return None


def http_get_dataset_job_per_second(dataset : str, sample_size : int):
    url = SERVER_ADRESS + f"/job/get-dataset-job-per-second?dataset={dataset}&sample_size={sample_size}"

    try:
        response = requests.get(url)

        if response.status_code == 200:
            job_json = response.json()
            return job_json

    except Exception as e:
        print('request exception ', e)

    return None