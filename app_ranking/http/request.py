import requests

SERVER_ADDRESS = 'http://192.168.3.1:8111'


# Get request to get random image
def http_get_random_image(dataset_name: str):
    url = SERVER_ADDRESS + "/image/random/{}".format(dataset_name)
    try:
        response = requests.get(url)
    except Exception as e:
        print(e)
        return None

    if response.status_code == 200:
        image_json = response.json()
        return image_json

    return None


def http_get_image_by_file_path(file_path: str):
    url = SERVER_ADDRESS + "/image/data-by-filepath?file_path={}".format(file_path)
    try:
        response = requests.get(url)
    except Exception as e:
        print(e)
        return None

    if response.status_code == 200:
        return response.content

    return None


# Get request to get datasets
def http_get_datasets():
    url = SERVER_ADDRESS + "/dataset/list"
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
        requests.post(url, json=selection_json_data, headers=headers)
    except Exception as e:
        print(e)
        return None

    return True

