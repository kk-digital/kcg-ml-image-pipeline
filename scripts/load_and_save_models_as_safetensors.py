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
from utility.http import request
from utility.minio import cmd
from training_worker.ab_ranking.model.ab_ranking_linear import ABRankingModel
from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from training_worker.ab_ranking.model.independent_approximation_v1 import ABRankingIndependentApproximationV1Model


def process_linear(minio_client, paths):
    print("Processing linear models...")
    for path in tqdm(paths):
        inputs_shape = 768 * 2
        if "positive" in path or "negative" in path or "clip" in path:
            inputs_shape = 768

        linear_model = ABRankingModel(inputs_shape)

        model_file_data = cmd.get_file_from_minio(minio_client, 'datasets', path)
        if not model_file_data:
            raise Exception("No .pth file found at path: ", model_path)

        byte_buffer = io.BytesIO()
        for data in model_file_data.stream(amt=8192):
            byte_buffer.write(data)
        byte_buffer.seek(0)
        linear_model.load_pth(byte_buffer)

        # save as msgpack
        # new save fn are saving as msgpack
        linear_model.save(minio_client, 'datasets', path.replace(".pth", ".safetensors"))

def process_elm_v1(minio_client, paths):
    print("Processing elm v1 models...")

    for path in tqdm(paths):
        inputs_shape = 768 * 2
        if "positive" in path or "negative" in path or "clip" in path:
            inputs_shape = 768

        elm_model = ABRankingELMModel(inputs_shape)

        model_file_data = cmd.get_file_from_minio(minio_client, 'datasets', path)
        if not model_file_data:
            raise Exception("No .pth file found at path: ", model_path)

        byte_buffer = io.BytesIO()
        for data in model_file_data.stream(amt=8192):
            byte_buffer.write(data)
        byte_buffer.seek(0)
        elm_model.load_pth(byte_buffer)

        # save as msgpack
        # new save fn are saving as msgpack
        elm_model.save(minio_client, 'datasets', path.replace(".pth", ".safetensors"))

def process_independent_approx_v1(minio_client, paths):
    print("Processing independent approx v1 models...")
    for path in tqdm(paths):
        # place holder
        inputs_shape = 768
        input_type = "positive"
        if "negative" in path:
            input_type = "negative"
        independent_approx_model = ABRankingIndependentApproximationV1Model(inputs_shape=inputs_shape,
                                                                            input_type=input_type)

        model_file_data = cmd.get_file_from_minio(minio_client, 'datasets', path)
        if not model_file_data:
            raise Exception("No .pth file found at path: ", model_path)

        byte_buffer = io.BytesIO()
        for data in model_file_data.stream(amt=8192):
            byte_buffer.write(data)
        byte_buffer.seek(0)
        independent_approx_model.load_pth(byte_buffer)

        # save as msgpack
        # new save fn are saving as msgpack
        independent_approx_model.save(minio_client, 'datasets', path.replace(".pth", ".safetensors"))

def load_all_models_and_save_as_msgpack(minio_client,
                                        dataset_name):

    start_time = time.time()

    print("Getting all model paths for dataset: {}...".format(dataset_name))
    prefix = os.path.join(dataset_name, "models/ranking")
    all_objects = cmd.get_list_of_objects_with_prefix(minio_client, 'datasets', prefix)

    # filter out and use only the one that ends with .pth
    model_pth_paths = [path for path in all_objects if ".pth" in path]

    linear_model_pth_paths = [path for path in model_pth_paths if "linear" in path]
    elm_v1_model_pth_paths = [path for path in model_pth_paths if "elm-v1" in path]
    xgboost_ranking_pairwise_model_pth_paths = [path for path in model_pth_paths if "xgboost-rank-pairwise" in path]
    independent_approx_v1_model_pth_paths = [path for path in model_pth_paths if "independent-approximation-v1" in path]

    process_linear(minio_client, linear_model_pth_paths)
    process_elm_v1(minio_client, elm_v1_model_pth_paths)
    # process_independent_approx_v1(minio_client, independent_approx_v1_model_pth_paths)

    time_elapsed = time.time() - start_time
    print("Dataset: {}: Total Time elapsed: {}s".format(dataset_name, format(time_elapsed, ".2f")))


def parse_args():
    parser = argparse.ArgumentParser(description="Model file converter")
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--dataset-name', required=True, help='Name of the dataset for embeddings')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    dataset_name = args.dataset_name
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)
    if dataset_name != "all":
        load_all_models_and_save_as_msgpack(minio_client,
                                            args.dataset_name,)
    else:
        # if all, train models for all existing datasets
        # get dataset name list
        dataset_names = request.http_get_dataset_names()
        print("dataset names=", dataset_names)
        for dataset in dataset_names:
            try:
                load_all_models_and_save_as_msgpack(minio_client,
                                                    dataset,)
            except Exception as e:
                print("Error running image scorer for {}: {}".format(dataset, e))


if __name__ == "__main__":
    main()
