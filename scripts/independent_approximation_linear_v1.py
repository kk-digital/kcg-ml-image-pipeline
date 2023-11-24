import os
import sys
import argparse

base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from training_worker.ab_ranking.script.independent_approximation_linear_v1 import train_ranking
from worker.http import request

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train ab ranking linear model")

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--dataset-name', type=str,
                        help="The dataset name to use for training, use 'all' to train models for all datasets",
                        default='environmental')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    dataset_name = args.dataset_name

    if dataset_name != "all":
        train_ranking(minio_ip_addr=None,  # will use default if none is given
                      minio_access_key=args.minio_access_key,
                      minio_secret_key=args.minio_secret_key,
                      dataset_name=dataset_name)
    else:
        # if all, train models for all existing datasets
        # get dataset name list
        dataset_names = request.http_get_dataset_names()
        print("dataset names=", dataset_names)
        for dataset in dataset_names:
            try:
                print("Training model for {}...".format(dataset))
                train_ranking(minio_ip_addr=None,  # will use default if none is given
                              minio_access_key=args.minio_access_key,
                              minio_secret_key=args.minio_secret_key,
                              dataset_name=dataset)
            except Exception as e:
                print("Error training model for {}: {}".format(dataset, e))

