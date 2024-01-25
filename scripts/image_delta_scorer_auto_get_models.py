import os
import sys
import argparse
import io
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
base_directory = "./"
sys.path.insert(0, base_directory)

from scripts.image_scorer import ImageScorer
from utility.http import model_training_request
from utility.http import request
from utility.minio import cmd
from scripts.image_delta_scorer import run_image_delta_scorer

def get_latest_model_filename(client, model_type, dataset_name):
    path_prefix = "{}/models/ranking".format(dataset_name)
    paths = cmd.get_list_of_objects_with_prefix(client, bucket_name="datasets", prefix=path_prefix)

    clip_model_filename = ""
    embedding_model_filename = ""

    paths = sorted(paths, key=lambda x: os.path.basename(x)[:13], reverse=True)
    # get models
    for path in paths:
        if ".safetensors" in path and model_type in path:
            if "clip" in path and clip_model_filename == "" and "positive" not in path and "negative" not in path:
                clip_model_filename = os.path.basename(path)
            if "embedding" in path and embedding_model_filename == "" and "positive" not in path and "negative" not in path:
                embedding_model_filename = os.path.basename(path)

            if clip_model_filename != "" and embedding_model_filename != "":
                break

    return clip_model_filename, embedding_model_filename

def parse_args():
    parser = argparse.ArgumentParser(description="Image Delta Scorer With auto get latest model")
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--model-type', required=True, help='linear or elm-v1')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)

    # if all, train models for all existing datasets
    # get dataset name list
    dataset_names = request.http_get_dataset_names()
    print("dataset names=", dataset_names)
    for dataset in dataset_names:
        print("Calculating delta score for dataset: {}".format(dataset))
        # get latest model
        clip_model_filename, embedding_model_filename = get_latest_model_filename(client=minio_client,
                                                                                  model_type=args.model_type,
                                                                                  dataset_name=dataset)

        print("clip model= ", clip_model_filename)
        print("embedding model = ", embedding_model_filename)
        try:
            run_image_delta_scorer(minio_client,
                                   dataset,
                                   clip_model_filename,
                                   embedding_model_filename)
        except Exception as e:
            print("Error running image scorer for {}: {}".format(dataset, e))


if __name__ == "__main__":
    main()