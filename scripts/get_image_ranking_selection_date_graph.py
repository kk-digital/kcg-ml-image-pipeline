import os
import sys
import argparse
from tqdm import tqdm
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pytz import timezone
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from io import BytesIO
base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from worker.http import request
from utility.minio import cmd

SERVER_ADRESS = 'http://192.168.3.1:8111'

# Get completed job info
def http_get_completed_job(image_hash):
    url = SERVER_ADRESS + "/job/get-completed-job-by-hash?image_hash={}".format(image_hash)
    try:
        response = requests.get(url)

        if response.status_code == 200:
            data_json = response.json()
            return data_json

    except Exception as e:
        print('request exception ', e)

    return None

def get_image_hash_date(image_hash):
    data = http_get_completed_job(image_hash)

    # YYYY-MM-DD
    date = data["task_completion_time"][:10]

    return date

def get_image_hashes(minio_client, object_path):
    response = minio_client.get_object("datasets", object_path)
    data = response.data
    decoded_data = data.decode().replace("'", '"')
    item = json.loads(decoded_data)

    return [item["image_1_metadata"]["file_hash"], item["image_2_metadata"]["file_hash"]]


def get_all_image_hash(minio_client, dataset_name):
    all_image_hashes =[]

    # get all paths
    selection_datapoints_path = os.path.join(dataset_name, "data/ranking/aggregate")
    objects = cmd.get_list_of_objects_with_prefix(minio_client, "datasets", selection_datapoints_path)
    print("len objects=", len(objects))

    print("Getting image hashes from datapoints...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for object_path in objects:
            # filter only that ends with .json
            if ".json" in object_path:
                futures.append(executor.submit(get_image_hashes, minio_client=minio_client, object_path=object_path))

        for future in tqdm(as_completed(futures), total=len(futures)):
            image_hashes = future.result()
            all_image_hashes.extend(image_hashes)

    return all_image_hashes

def get_date_count(image_hashes):
    date_count_dict = {}

    print("Creating date count dict...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for image_hash in image_hashes:
            futures.append(executor.submit(get_image_hash_date, image_hash=image_hash))

        for future in tqdm(as_completed(futures), total=len(futures)):
            date = future.result()
            if date not in date_count_dict:
                date_count_dict[date] = 1
            else:
                count = date_count_dict[date]
                count += 1
                date_count_dict[date] = count
    print("Processing date count dict...")
    sorted_count_dict = dict(sorted(date_count_dict.items()))
    print("sorted=", sorted_count_dict)

    return sorted_count_dict


def get_hist_graph_buffer(sorted_count_dict):
    # Initialize all graphs/subplots
    plt.figure(figsize=(22, 20))
    figure_shape = (2, 2)
    date_distribution = plt.subplot2grid(figure_shape, (0, 0), rowspan=2, colspan=2)
    # ----------------------------------------------------------------------------------------------------------------#
    # date distribution
    date_distribution.bar(list(sorted_count_dict.keys()), sorted_count_dict.values(), color='g')

    x_keys = list(sorted_count_dict.keys())
    x_pos = range(len(x_keys))
    date_distribution.set_xticks(x_pos, x_keys, rotation='vertical')

    date_distribution.set_xlabel("Date")
    date_distribution.set_ylabel("Frequency")
    date_distribution.set_title("Selection Datapoint Image Date Distribution")
    date_distribution.legend()

    date_distribution.autoscale(enable=True, axis='both')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return buf


def get_ranking_distribution_graph(minio_client, dataset_name):
    image_hashes = get_all_image_hash(minio_client=minio_client,
                                      dataset_name=dataset_name)

    sorted_count_dict = get_date_count(image_hashes)
    buf = get_hist_graph_buffer(sorted_count_dict)

    # upload the graph report
    date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%Y-%m-%d')
    print("Current datetime: {}".format(datetime.now(tz=timezone("Asia/Hong_Kong"))))
    filename = "{}.png".format(date_now)
    graph_output = os.path.join(dataset_name, "output/selection-image-date-distribution-graph", filename)
    cmd.upload_data(minio_client, 'datasets', graph_output, buf)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Creates ranking selection datapoints distribution graph by date")

    parser.add_argument('--minio-ip-addr', type=str, help='Minio ip addr', default=None)
    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--dataset-name', type=str,
                        help="The dataset name to use, use 'all' to train models for all datasets",
                        default='environmental')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    dataset_name = args.dataset_name
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_ip_addr)
    if dataset_name != "all":
        get_ranking_distribution_graph(minio_client, dataset_name)
    else:
        # if all, train models for all existing datasets
        # get dataset name list
        dataset_names = request.http_get_dataset_names()
        print("dataset names=", dataset_names)
        for dataset in dataset_names:
            try:
                print("Generating distribution graph for {}...".format(dataset))
                get_ranking_distribution_graph(minio_client, dataset)
            except Exception as e:
                print("Error generating distribution graph for {}: {}".format(dataset, e))

