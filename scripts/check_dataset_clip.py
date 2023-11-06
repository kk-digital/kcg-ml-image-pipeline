import os
import sys
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from worker.http import request
from utility.minio import cmd


def check_clip_and_create(minio_client, object_path):
            img_path = object_path.replace(".jpg", "_clip.msgpack")
            img_path = img_path.replace("datasets/", "")

            exists = cmd.is_object_exists(minio_client, "datasets", img_path)
            if not exists:
                input_file_path = "datasets/" + object_path
                print("creating clip calculation task for {}".format(input_file_path))
                # add clip calculation task
                clip_calculation_job = {"uuid": "",
                                        "task_type": "clip_calculation_task",
                                        "task_input_dict": {
                                            "input_file_path": input_file_path,
                                            "input_file_hash": "manual"
                                        },
                                        }

                request.http_add_job(clip_calculation_job)


def run_concurrent_check(minio_client, dataset_name):
    # get all paths
    objects = cmd.get_list_of_objects_with_prefix(minio_client, "datasets", dataset_name)
    print("len objects=", len(objects))

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for object_path in objects:
            # filter only that ends with .jpg
            if ".jpg" in object_path:
                futures.append(executor.submit(check_clip_and_create, minio_client=minio_client, object_path=object_path))

        for _ in tqdm(as_completed(futures), total=len(objects)):
            continue


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Creates clip job if doesnt exist")

    parser.add_argument('--minio-ip-addr', type=str, help='Minio ip addr', default=None)
    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--dataset-name', type=str,
                        help="The dataset name to use for training, use 'all' to train models for all datasets",
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
        dataset_names = request.http_get_dataset_names()
        print("dataset names=", dataset_names)
        for dataset in dataset_names:
            try:
                print("Checking clip data for {}...".format(dataset))
                run_concurrent_check(minio_client, dataset)
            except Exception as e:
                print("Error checking clip data for {}: {}".format(dataset, e))

