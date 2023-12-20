import os
import sys
import argparse

base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from training_worker.ab_ranking.script.ab_ranking_xgboost_ranking_pairwise import train_xgboost
from utility.http import generation_request
from utility.http import request


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train ab ranking xgboost model")

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--dataset-name', type=str,
                        help="The dataset name to use for training, use 'all' to train models for all datasets",
                        default='environmental')
    parser.add_argument('--input-type', type=str, default="embedding")
    parser.add_argument('--train-percent', type=float, default=0.9)
    parser.add_argument('--load-data-to-ram', type=bool, default=True)
    parser.add_argument('--normalize-vectors', type=bool, default=True)
    parser.add_argument('--pooling-strategy', type=int, default=0)
    parser.add_argument('--target-option', type=int, default=0)
    parser.add_argument('--duplicate-flip-option', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    dataset_name = args.dataset_name

    if dataset_name != "all":
        train_xgboost(minio_ip_addr=None,  # will use default if none is given
                      minio_access_key=args.minio_access_key,
                      minio_secret_key=args.minio_secret_key,
                      dataset_name=dataset_name,
                      input_type=args.input_type,
                      train_percent=args.train_percent,
                      load_data_to_ram=args.load_data_to_ram,
                      normalize_vectors=args.normalize_vectors,
                      pooling_strategy=args.pooling_strategy,
                      target_option=args.target_option,
                      duplicate_flip_option=args.duplicate_flip_option)
    else:
        # if all, train models for all existing datasets
        # get dataset name list
        dataset_names = request.http_get_dataset_names()
        print("dataset names=", dataset_names)
        for dataset in dataset_names:
            try:
                print("Training model for {}...".format(dataset))
                train_xgboost(minio_ip_addr=None,  # will use default if none is given
                              minio_access_key=args.minio_access_key,
                              minio_secret_key=args.minio_secret_key,
                              dataset_name=dataset,
                              input_type=args.input_type,
                              train_percent=args.train_percent,
                              load_data_to_ram=args.load_data_to_ram,
                              normalize_vectors=args.normalize_vectors,
                              pooling_strategy=args.pooling_strategy,
                              target_option=args.target_option,
                              duplicate_flip_option=args.duplicate_flip_option)
            except Exception as e:
                print("Error training model for {}: {}".format(dataset, e))

