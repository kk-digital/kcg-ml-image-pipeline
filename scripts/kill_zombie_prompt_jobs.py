import os
import sys
import argparse
import requests
from datetime import datetime, timedelta

base_directory = os.getcwd()
sys.path.insert(0, base_directory)


SERVER_ADRESS = 'http://192.168.3.1:8111'


def http_get_in_progress_jobs_count(dataset_name: str):
    url = SERVER_ADRESS + "/queue/image-generation/in-progress-count?dataset=" + dataset_name

    try:
        response = requests.get(url)

        if response.status_code == 200:
            job_json = response.json()
            return job_json

    except Exception as e:
        print('request exception ', e)

    return None

def http_get_in_progress_jobs():
    url = SERVER_ADRESS + "/queue/image-generation/list-in-progress"

    try:
        response = requests.get(url)

        if response.status_code == 200:
            job_json = response.json()
            return job_json

    except Exception as e:
        print('request exception ', e)

    return None

def http_update_failed_job(job):
    url = SERVER_ADRESS + "/queue/image-generation/update-failed"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data

    try:
        response = requests.put(url, json=job, headers=headers)
    except Exception as e:
        print('request exception ', e)

    if response.status_code != 200:
        print(f"request failed with status code: {response.status_code}")


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
    parser.add_argument('--max_days', type=int,
                            help="The maximum number of days the job can be in progress for",
                            default=2)

    return parser.parse_args()

def is_time_difference_more_than_n_days(creation_date_str, max_days):
    # Convert the creation date string to a datetime object
    creation_date = datetime.strptime(creation_date_str, '%Y-%m-%dT%H:%M:%S.%f')

    # Get the current date and time
    current_date = datetime.now()

    # Calculate the time difference
    time_difference = current_date - creation_date

    # Check if the time difference is more than n days
    if time_difference > timedelta(days=max_days):
        return True
    else:
        return False


def kill_zombie_jobs(dataset, max_days):
    job_list = http_get_in_progress_jobs()

    job_count = len(job_list)
    job_index = 0
    number_of_removed_jobs = 0

    for job in job_list:
        job_index = job_index + 1
        print(f'processing job {job_index} out of {job_count}')
        if job is None:
            continue

        # task input dictionary is only
        # available on image generation tasks
        task_input_dict = job['task_input_dict']

        if task_input_dict is None:
            continue

        # make sure the input_dictionary has a dataset field
        # if it does not contain this field means that its not
        # an image generation task
        if 'dataset' not in task_input_dict:
            continue

        job_dataset = task_input_dict['dataset']

        # skip the job if the dataset does not match
        if job_dataset != dataset:
            continue

        # task_creation_time field is present in all job task_types
        job_creation_time = job['task_creation_time']

        if job_creation_time is None:
            continue

        result = is_time_difference_more_than_n_days(job_creation_time, max_days)

        # if the difference is more than max_days
        # clear the zombie job
        # set the in progress job as failed
        if result:
            http_update_failed_job(job)

            number_of_removed_jobs = number_of_removed_jobs + 1

    print(f'number of removed jobs {number_of_removed_jobs}')


def kill_all_zombie_jobs(max_days):
    job_list = http_get_in_progress_jobs()

    job_count = len(job_list)
    job_index = 0
    number_of_removed_jobs = 0

    for job in job_list:
        job_index = job_index + 1
        print(f'processing job {job_index} out of {job_count}')
        if job is None:
            continue

        # task_creation_time field is present in all job task_types
        job_creation_time = job['task_creation_time']

        if job_creation_time is None:
            continue

        result = is_time_difference_more_than_n_days(job_creation_time, max_days)

        # if the difference is more than max_days
        # clear the zombie job
        # set the in progress job as failed
        if result:
            http_update_failed_job(job)

            number_of_removed_jobs = number_of_removed_jobs + 1

    print(f'number of removed jobs {number_of_removed_jobs}')

# go through all the in progress jobs
# and make sure they are not too old
# if they are more than 2, 3 days it means
# that the jobs are failed
def main():
    args = parse_arguments()

    dataset_name = args.dataset_name
    max_days = args.max_days

    if dataset_name == 'all':
        # if dataset name is 'all'
        # we would kill all zombie jobs
        kill_all_zombie_jobs(max_days)
    else:
        # otherwise just kill the jobs
        # that belong to the dataset
        kill_zombie_jobs(dataset_name, max_days)

    print('all zombie jobs have been deleted successfully')


if __name__ == '__main__':
    main()