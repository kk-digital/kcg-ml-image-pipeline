import requests

#SERVER_ADRESS = 'http://192.168.3.1:8111'
SERVER_ADRESS = 'http://127.0.0.1:8000'

def http_get_completed_jobs_count(dataset_name: str):
    url = SERVER_ADRESS + "/job/count-completed?dataset=" + dataset_name

    try:
        response = requests.get(url)

        if response.status_code == 200:
            job_json = response.json()
            return job_json

    except Exception as e:
        print('request exception ', e)


    return None


def http_get_pending_jobs_count(dataset_name: str):
    url = SERVER_ADRESS + "/job/count-pending?dataset=" + dataset_name

    try:
        response = requests.get(url)

        if response.status_code == 200:
            job_json = response.json()
            return job_json

    except Exception as e:
        print('request exception ', e)


    return None


def http_get_in_progress_jobs_count(dataset_name: str):
    url = SERVER_ADRESS + "/job/count-in-progress?dataset=" + dataset_name

    try:
        response = requests.get(url)

        if response.status_code == 200:
            job_json = response.json()
            return job_json

    except Exception as e:
        print('request exception ', e)


    return None

