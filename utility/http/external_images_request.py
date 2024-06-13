import requests

SERVER_ADDRESS = 'http://192.168.3.1:8111'

def http_add_external_image(image_data):
    url = SERVER_ADDRESS + "/external-images/add-external-image"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data
    response = None

    try:
        response = requests.post(url, json=image_data, headers=headers)

        if response.status_code != 200:
            print(f"request failed with status code: {response.status_code}: {str(response.content)}")
    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()

    return None

def http_get_external_image_list(dataset, size=None):
    endpoint_url= "/external-images/get-all-external-image-list?dataset={}".format(dataset)

    if size:
        endpoint_url+= f"&size={size}"

    url = SERVER_ADDRESS + endpoint_url
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            data_json = response.json()
            return data_json['response']['data']

    except Exception as e:
        print('request exception ', e)

def http_get_external_image_list_without_extracts(dataset, size=None):
    endpoint_url= "/external-images/get-external-image-list-without-extracts?dataset={}".format(dataset)

    if size:
        endpoint_url+= f"&size={size}"

    url = SERVER_ADDRESS + endpoint_url
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            data_json = response.json()
            return data_json['response']['data']

    except Exception as e:
        print('request exception ', e)

    
def http_add_extract(image_data):
    url = SERVER_ADDRESS + "/extracts/add-extracted-image"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data
    response = None

    try:
        response = requests.post(url, json=image_data, headers=headers)

        if response.status_code != 200:
            print(f"request failed with status code: {response.status_code}: {str(response.content)}")
    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()

    return None

def http_get_extract_image_list(dataset, size=None):
    endpoint_url= "/extracts/get-all-extracts-list?dataset={}".format(dataset)

    if size:
        endpoint_url+= f"&size={size}"

    url = SERVER_ADDRESS + endpoint_url
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            data_json = response.json()
            return data_json['response']['data']

    except Exception as e:
        print('request exception ', e)


def http_get_current_extract_batch_sequential_id(dataset: str):
    endpoint_url= "/extracts/get-current-data-batch-sequential-id?dataset={}".format(dataset)

    url = SERVER_ADDRESS + endpoint_url
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            data_json = response.json()
            return data_json

    except Exception as e:
        print('request exception ', e)

def http_get_next_extract_batch_sequential_id(dataset: str, is_complete: bool = True):
    endpoint_url= "/extracts/get-next-data-batch-sequential-id?dataset={}&complete={}".format(dataset, is_complete)

    url = SERVER_ADDRESS + endpoint_url
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            data_json = response.json()
            return data_json

    except Exception as e:
        print('request exception ', e)