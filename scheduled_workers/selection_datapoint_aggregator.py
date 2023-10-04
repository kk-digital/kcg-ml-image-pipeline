"""
This script aggregates all individual selection datapoints json inside
/datasets/<dataset-name>/data/ranking/aggregate/ into single per day per dataset json.
"""

import os
import schedule
import sys
from pytz import timezone
from datetime import datetime
from dotenv import dotenv_values
import json
from io import BytesIO
import argparse

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.minio import cmd

config = dotenv_values("./scheduled_workers/.env")
DATASETS_BUCKET = "datasets"

def get_datasets(minio_client):
    datasets = cmd.get_list_of_objects(minio_client, DATASETS_BUCKET)
    return datasets


def get_selection_datapoints(minio_client, dataset_name):
    prefix = os.path.join(dataset_name, "data/ranking/aggregate")
    datasets = cmd.get_list_of_objects_with_prefix(minio_client, DATASETS_BUCKET, prefix=prefix)
    return datasets


def get_object(client, file_path):
    response = client.get_object(DATASETS_BUCKET, file_path)
    data = response.data

    return data


def remove_object(client, file_path):
    client.remove_object(DATASETS_BUCKET, file_path)


def remove_todays_data_from_list(selection_datapoints):
    date = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%Y-%m-%d')
    new_list = []
    for path in selection_datapoints:
        if date not in path:
            new_list.append(path)

    return new_list

def get_summary_dict(selection_datapoints):
    summary_dict = {}
    # get date
    for path in selection_datapoints:
        base = os.path.basename(path)
        date = base[:10]

        if date not in summary_dict:
            summary_dict[date] = []

        summary_dict[date].append(path)

    return summary_dict

def aggregate_dataset_selection_datapoints(minio_client, dataset_name):
    selection_datapoints = get_selection_datapoints(minio_client, dataset_name=dataset_name)

    # we don't want to prematurely aggregate today's data since more data is still being added
    selection_datapoints = remove_todays_data_from_list(selection_datapoints)

    # get summary dict containing a date key and list of paths as values
    summary_dict = get_summary_dict(selection_datapoints)
    for key, val in summary_dict.items():
        full_path = os.path.join(dataset_name, "data/ranking", "{}-{}.json".format(key, dataset_name))
        json_values = []
        for path in val:
            data = get_object(minio_client, path)
            decoded_data = data.decode().replace("'", '"')
            json_data = json.loads(decoded_data)
            json_values.append(json_data)
            json_data = json.dumps(json_values, indent=4).encode('utf-8')
            data = BytesIO(json_data)

        # upload the aggregated data
        cmd.upload_data(minio_client, DATASETS_BUCKET, full_path, data)

    print("Finished aggregating datapoints...")
    print("Now removing aggregated datapoints...")
    for path in selection_datapoints:
        remove_object(minio_client, path)
    print("Finished removing aggregated datapoints...")


def aggregate_selection_datapoints():
    start_time = datetime.now()
    print("Starting selection datapoints aggregation task...")
    # get minio client
    minio_client = cmd.get_minio_client(config["MINIO_ACCESS_KEY"], config["MINIO_SECRET_KEY"])
    dataset_list = get_datasets(minio_client)
    print(dataset_list)

    # for every dataset get aggregated json files
    for dataset_name in dataset_list:
        print("Processing dataset: {}".format(dataset_name))
        aggregate_dataset_selection_datapoints(minio_client, dataset_name)
        print("Finished processing dataset: {}".format(dataset_name))
        print("==========================================================")

    print("Finished selection datapoints aggregation task...")
    print("Time Elapsed: {}s".format(datetime.now() - start_time))



def main(time_to_run):
    print("Current datetime: {}".format(datetime.now(tz=timezone("Asia/Hong_Kong"))))
    print("The script will run everyday at {} UTC+8:00".format(time_to_run))
    schedule.every().day.at(time_to_run, timezone("Asia/Hong_Kong")).do(aggregate_selection_datapoints)
    while True:
        schedule.run_pending()

def parse_args():
    parser = argparse.ArgumentParser(description="Worker for image generation")

    # Required parameters
    parser.add_argument("--time-to-run", type=str, default="00:10")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    main(args.time_to_run)