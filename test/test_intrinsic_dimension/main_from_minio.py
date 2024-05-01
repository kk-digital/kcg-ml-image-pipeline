import argparse
import os
import sys


import torch
import numpy as np
import pandas as pd
import csv

from datetime import datetime

# library for getting intrinsic dimension
from intrinsics_dimension import mle_id, twonn_numpy, twonn_pytorch
import skdim

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

# import utils
from utils import measure_running_time

# import library type
from library_type import Library

# load data loader for feature data
from dataloader import KandinskyDatasetLoader
from utils import get_minio_client

# import convert seconds into formatted time string
from utils import format_duration

# import http request
from utility.http import request

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--minio-addr', type=str, help='Minio address')
    parser.add_argument('--library', type=int, default=0, help='Lirary used for getting intrinsic dimension of data, ["intrinsic_dimension", "scipy"], now there are two available libraries')
    parser.add_argument('--data-type-list', type=str, nargs='+', default=['clip'], help='Data types to obtain intrinsic dimensions, for exmaple, clip and vae')
    parser.add_argument('--count-list', type=int, nargs='+', default=[100], help="list of count for getting intrinsic dimension")
    parser.add_argument('--dataset', type=str, default="environmental", help="Dataset name")
    return parser.parse_args()


def load_featurs_data(minio_client, data_type, max_count, dataset):
    featurs_data = []

    try:
        dataloader = KandinskyDatasetLoader(minio_client=minio_client, dataset=dataset)
    except Exception as e:
        print("Error in initializing kankinsky dataset loader:{}".format(e))
        return

    # get features data depends on data type
    if data_type == "clip":
        featurs_data = dataloader.load_clip_vector_data(limit=max_count)
    elif data_type == "vae":
        featurs_data = dataloader.load_latents(limit=max_count)
    else:
        print("No support data type {}".format(data_type))
        return featurs_data
    
    return featurs_data

def get_file_name(*args, seperator='_'):
    
    return os.path.join(os.getcwd(), 
                        "output", 
                        "{}_intrinsic_dim_results_{}.csv".format(
                            datetime.now(), seperator.join(map(str, args))))

def get_intrinsic_dimenstions(minio_client, dataset, library, count_list, data_type_list):
    """
    Calculates the intrinsic dimensions of a dataset using different feature data types.

    Args:
        minio_client (object): A MinIO client object for accessing the dataset.
        dataset (str): The name of the dataset to be analyzed.
        library (str): The library to be used for intrinsic dimension estimation.
        count_list (list): A list of testing counts for the number of clip vectors.
        data_type_list (list): A list of data types for the features.

    Returns:
        None
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # max count of clip vectors for testing intrinsic dimension
    max_count = max(count_list) * 2

    result = []

    for data_type in data_type_list:

        # load feature data from environment dataset
        feature_data = load_featurs_data(minio_client, data_type, max_count, dataset)
        if len(feature_data) == 0:
            print("Error loading feature data from {} dataset".format(dataset))
            return

        for count in count_list:

            # get specific count of data for gettting intrinsic dimension
            data = torch.tensor(feature_data[:count], device=device)

            # wrangle the latent vector [1, 4, 64, 64]
            if data_type == "vae":
                data = data.reshape((data.size(0), -1))
            
            if library == Library.INTRINSIC_DIMENSION.value:

                dimension_by_mle, mle_elapsed_time = \
                    measure_running_time(mle_id, data, k=2)

                dimension_by_twonn_numpy, twonn_numpy_elapsed_time = \
                    measure_running_time(twonn_numpy, data.cpu().numpy(), return_xy=False)

                dimension_by_twonn_torch, twonn_pytorch_elapsed_time = \
                    measure_running_time(twonn_pytorch, data, return_xy=False)
                result.append({
                        "Data type": "Clip vector" if data_type == "clip" else "VAE",
                        "Number of clip vector": data.size(0),
                        "Dimension of clip vector": data.size(1),
                        "MLE intrinsic dimension": "{:.2f}".format(dimension_by_mle),
                        "MLE elapsed time": "{}".format(format_duration(mle_elapsed_time)),
                        "Twonn_numpy intrinsic dimension": "{:.2f}".format(dimension_by_twonn_numpy),
                        "Twonn_numpy elapsed time": "{}".format(format_duration(twonn_numpy_elapsed_time)),
                        "twonn_pytorch intrinsic dimension": "{:.2f}".format(dimension_by_twonn_torch),
                        "twonn_pytorch elapsed time": "{}".format(format_duration(twonn_pytorch_elapsed_time))
                })

            elif library == Library.SCIKIT_DIMENSION.value:
                data = data.cpu().numpy()
                dimension_by_mle, mle_elapsed_time = measure_running_time(skdim.id.MLE().fit, data)
                dimension_by_twonn_numpy, twonn_elapsed_time = measure_running_time(skdim.id.TwoNN().fit, data)

                if dimension_by_mle != 'Nan':
                    dimension_by_mle = dimension_by_mle.dimension_
                if dimension_by_twonn_numpy != 'Nan':
                    dimension_by_twonn_numpy = dimension_by_twonn_numpy.dimension_

                result.append({
                        "Data type": "Clip vector" if data_type == "clip" else "VAE",
                        "Number of vae vectors": data.shape[0],
                        "Dimension of vae vector": data.shape[1],
                        "MLE intrinsic dimension": "{:.2f}".format(dimension_by_mle),
                        "MLE elapsed time": "{}".format(format_duration(mle_elapsed_time)),
                        "Twonn Intrinsic dimension": "{:.2f}".format(dimension_by_twonn_numpy),
                        "Twonn elapsed time": "{}".format(format_duration(twonn_elapsed_time))
                })
            return result

def main():
    args = parse_args()

    # get minio client
    minio_client = get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)
    
    if args.library == Library.INTRINSIC_DIMENSION.value:
        df = pd.DataFrame(columns=["Data type", "Number of vector", "Dimension of vector", "MLE intrinsic dimension", "MLE elapsed time", "Twonn_numpy intrinsic dimension", "Twonn_numpy elapsed time", "twonn_pytorch intrinsic dimension", "twonn_pytorch elapsed time"])
    
    elif args.library == Library.SCIKIT_DIMENSION.value:
        df = pd.DataFrame(columns=["Data type", "Number of vectors", "Dimension of vector", "MLE intrinsic dimension", "MLE elapsed time", "Twonn Intrinsic dimension", "Twonn elapsed time"])
    else:
        print("Not support such library")
        return

    if args.dataset == 'all':
        dataset_names = request.http_get_dataset_names()

        for dataset in dataset_names:

            results = get_intrinsic_dimenstions(minio_client=minio_client, 
                              dataset=dataset, 
                              library=args.library, 
                              count_list=args.count_list, 
                              data_type_list=args.data_type_list)
            
            for result in results:
                df.append(result, ignore_index=True)
    
    else:
        result = get_intrinsic_dimenstions(minio_client=minio_client, 
                              dataset=args.dataset, 
                              library=args.library, 
                              count_list=args.count_list, 
                              data_type_list=args.data_type_list)
        for result in results:
            df.append(result, ignore_index=True)
    df.groupby("Number of vectors")
    with open(get_file_name(), mode='w') as file:
        df.to_csv(get_file_name(), index=False)
        
if __name__ == "__main__":
    main()