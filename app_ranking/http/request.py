import requests

SERVER_ADDRESS = 'http://192.168.3.1:8111'


# Get request to get random image
def http_get_random_image(dataset_name: str):
    url = SERVER_ADDRESS + "/get-random-image/{}".format(dataset_name)
    try:
        response = requests.get(url)
    except Exception as e:
        print(e)
        return None

    if response.status_code == 200:
        image_json = response.json()
        return image_json

    return None


# Get request to get datasets
def http_get_datasets():
    url = SERVER_ADDRESS + "/get-datasets"
    try:
        response = requests.get(url)
    except Exception as e:
        print(e)
        return None

    if response != None:
        datasets = response.json()
        return datasets

    return None


# Add selection datapoint
def http_add_selection(dataset_name: str, selection_json_data):
    url = SERVER_ADDRESS + "/add-selection-datapoint/{}".format(dataset_name)
    try:
        headers = {'Content-type': 'application/json'}
        response = requests.post(url, json=selection_json_data, headers=headers)
        if response.status_code != 200:
            raise Exception(response.json())
    except Exception as e:
        raise e
        return None

    return True
