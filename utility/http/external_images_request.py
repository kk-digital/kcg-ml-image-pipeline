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

def http_get_external_image_list(dataset= None, size=None):
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

def http_get_external_dataset_in_batches(dataset: str, batch_size: int):
    external_images=[]
    
    limit=batch_size
    offset=0

    while True:
        endpoint_url= "/external-images/list-images-v2?dataset={}&limit={}&offset={}&order=asc".format(dataset, limit, offset)

        url = SERVER_ADDRESS + endpoint_url
        try:
            response = requests.get(url)
            
            if response.status_code == 200:
                data_json = response.json()
                image_batch= data_json['response']['images']

                if len(image_batch)>0: 
                    external_images.extend(image_batch)
                else:
                    break

            else:
                break

        except Exception as e:
            print('request exception ', e)
            break

        offset += batch_size
    
    return external_images
        

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

        if response.status_code == 200:
            data_json = response.json()
            return data_json['response']['data']
        else:
            print(f"request failed with status code: {response.status_code}: {str(response.content)}")
    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()

    return None

def http_get_extract_image_list(dataset= None, size=None):
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

# Get list of all dataset names for external images
def http_get_external_dataset_list():
    url = SERVER_ADDRESS + "/external-images/list-datasets"
    response = None

    try:
        response = requests.get(url)

        if response.status_code == 200:
            data_json = response.json()
            return data_json['response']['datasets']

    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()

    return None

# Get list of all dataset names for extracts
def http_get_extract_dataset_list():
    url = SERVER_ADDRESS + "/extract-images/list-datasets"
    response = None

    try:
        response = requests.get(url)

        if response.status_code == 200:
            data_json = response.json()
            return data_json['response']['datasets']

    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()

    return None

# delete an extract by hash
def http_delete_extract(image_hash: str):
    url = SERVER_ADDRESS + "/extracts/delete-extract?image_hash={}".format(image_hash)
    response = None

    try:
        response = requests.delete(url)

        if response.status_code == 200:
            data_json = response.json()
            return data_json['response']

    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()

    return None