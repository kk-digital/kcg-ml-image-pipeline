# NOTE: don't add more imports here
# this is also used by training workers
import requests
import json

SERVER_ADDRESS = 'http://192.168.3.1:8111'
#SERVER_ADDRESS = 'http://127.0.0.1:8000'


# Get request to get an available job
def http_get_job(worker_type: str = None, model_type: str = None):
    url = SERVER_ADDRESS + "/queue/image-generation/get-job"
    response = None
    query_params = []

    if worker_type is not None:
        query_params.append("task_type={}".format(worker_type))
    if model_type is not None:
        query_params.append("model_type={}".format(model_type))

    if query_params:
        url += "?" + "&".join(query_params)

    try:
        response = requests.get(url)

        if response.status_code == 200:
            job_json = response.json()
            return job_json

    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()

    return None


def http_add_job(job):
    url = SERVER_ADDRESS + "/queue/image-generation/add"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data
    response = None
    
    try:
        response = requests.post(url, json=job, headers=headers)
        if response.status_code != 201 and response.status_code != 200:
            print(f"POST request failed with status code: {response.status_code}")

        decoded_response = json.loads(response.content.decode())
    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()

    return decoded_response

def http_add_kandinsky_job(job, positive_embedding, negative_embedding):
    url = SERVER_ADDRESS + "/queue/image-generation/add-kandinsky"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data
    response = None
    # Prepare the data in JSON format
    data = {
        "job": job,
        "positive_embedding": positive_embedding,
        "negative_embedding": negative_embedding
    }
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code != 201 and response.status_code != 200:
            print(f"POST request failed with status code: {response.status_code}")

        decoded_response = json.loads(response.content.decode())
    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()

    return decoded_response


def http_update_job_completed(job):
    url = SERVER_ADDRESS + "/queue/image-generation/update-completed"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data
    response = None

    try:
        response = requests.put(url, json=job, headers=headers)
        if response.status_code != 200:
            print(f"request failed with status code: {response.status_code}")    
    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()


def http_update_job_failed(job):
    url = SERVER_ADDRESS + "/queue/image-generation/update-failed"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data
    response = None

    try:
        response = requests.put(url, json=job, headers=headers)
        if response.status_code != 200:
            print(f"request failed with status code: {response.status_code}")
    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()
