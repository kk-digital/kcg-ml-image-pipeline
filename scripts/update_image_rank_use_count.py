import os
import sys
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import requests

base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from utility.http import generation_request
from utility.minio import cmd

SERVER_ADRESS = 'http://192.168.3.1:8111'


def get_image_hashes(minio_client, object_path):
    response = minio_client.get_object("datasets", object_path)
    data = response.data
    decoded_data = data.decode().replace("'", '"')
    item = json.loads(decoded_data)

    return [item["image_1_metadata"]["file_hash"], item["image_2_metadata"]["file_hash"]]


def set_image_rank_count(image_hash, count):
    url = SERVER_ADRESS + "/rank/set-image-rank-use-count?image_hash={}&count={}".format(image_hash, count)

    try:
        response = requests.post(url)

        if response.status_code != 200:
            print(f"request failed with status code: {response.status_code}: {str(response.content)}")
    except Exception as e:
        print('request exception ', e)


def run_concurrent_check(minio_client, dataset_name):
    image_hash_count_dict = {}

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
            for image_hash in image_hashes:
                if image_hash not in image_hash_count_dict:
                    image_hash_count_dict[image_hash] = 1
                else:
                    count = image_hash_count_dict[image_hash]
                    count += 1
                    image_hash_count_dict[image_hash] = count

    print("Setting image hash counts...")
    for image_hash, count in tqdm(image_hash_count_dict.items()):
        set_image_rank_count(image_hash, count)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Checks and updates rank use count for all images.")

    parser.add_argument('--minio-ip-addr', type=str, help='Minio ip addr', default=None)
    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--dataset-name', type=str,
                        help="The dataset name to check, use 'all' to train models for all datasets",
                        default='environmental')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    dataset_name = args.dataset_name
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_ip_addr)
    if dataset_name != "all":
        run_concurrent_check(minio_client, dataset_name)
    else:
        # if all, train models for all existing datasets
        # get dataset name list
        dataset_names = generation_request.http_get_dataset_names()
        print("dataset names=", dataset_names)
        for dataset in dataset_names:
            try:
                print("Checking image rank use count for {}...".format(dataset))
                run_concurrent_check(minio_client, dataset)
            except Exception as e:
                print("Error checking image rank use count for {}: {}".format(dataset, e))

