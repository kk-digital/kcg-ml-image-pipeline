import requests

SERVER_ADDRESS = 'http://192.168.3.1:8111'


# Get request to get an available job
def http_get_job(worker_type: str = None):
    url = SERVER_ADDRESS + "/training/get-job"
    response = None

    if worker_type is not None:
        url = url + "?task_type={}".format(worker_type)

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
    url = SERVER_ADDRESS + "/training/add"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data
    response = None

    try:
        response = requests.post(url, json=job, headers=headers)
        if response.status_code != 201 and response.status_code != 200:
            print(f"POST request failed with status code: {response.status_code}")
    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()
            

def http_update_job_completed(job):
    url = SERVER_ADDRESS + "/training/update-completed"
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
    url = SERVER_ADDRESS + "/training/update-failed"
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
            
