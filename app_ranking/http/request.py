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
