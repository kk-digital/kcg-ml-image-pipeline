import os
import sys
import argparse
import requests

base_directory = os.getcwd()
sys.path.insert(0, base_directory)


SERVER_ADRESS = 'http://192.168.3.1:8111'


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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train ab ranking linear model")

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--dataset-name', type=str,
                        help="The dataset name to use, use 'all' for all datasets",
                        default='all')

def main():
    args = parse_arguments()

    dataset_name = args.dataset_name

    # if dataset name is 'all'
    # we would kill all zombie jobs
    if dataset_name == 'all':



if __name__ == '__main__':
    main()