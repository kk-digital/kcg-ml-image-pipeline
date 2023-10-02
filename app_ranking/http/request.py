import requests

SERVER_ADRESS = 'http://localhost:8000'


# Get request to get random image
def http_get_random_image(dataset_name: str):
    url = SERVER_ADRESS + "/get-random-image/{}".format(dataset_name)
    try:
        response = requests.get(url)
    except Exception as e:
        print('request exception ', e)

    if response.status_code == 200:
        image_json = response.json()
        return image_json

    return None


# Get request to get datasets
def http_get_datasets():
    url = SERVER_ADRESS + "/get-datasets"
    try:
        response = requests.get(url)
    except Exception as e:
        print('request exception ', e)

    if response != None:
        datasets = response
        return datasets

    return None
