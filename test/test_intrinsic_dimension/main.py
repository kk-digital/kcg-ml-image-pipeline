import argparse
import os
import sys

import torch
import csv

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

def get_file_name():
    return os.path.join(os.getcwd(), "output", "intrinsic_dim_results.csv")

def main():
    args = parse_args()

    # get minio client
    minio_client = get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # testing number of clip vectors for example 100, 1000, 10000
    count_list = args.count_list
    # features data type
    data_type_list = args.data_type_list
    # max count of clip vectors for testing intrinsic dimension
    max_count = max(count_list) * 2


    with open(get_file_name(), mode='w', newline='') as file:

        if args.library == Library.INTRINSIC_DIMENSION.value:
            writer = csv.DictWriter(file, fieldnames=["Data type", "Number of clip vector", "Dimension of clip vector", "MLE intrinsic dimension", "MLE elapsed time", "Twonn_numpy intrinsic dimension", "Twonn_numpy elapsed time", "twonn_pytorch intrinsic dimension", "twonn_pytorch elapsed time"])
        
        elif args.library == Library.SCIKIT_DIMENSION.value:
            writer = csv.DictWriter(file, fieldnames=["Data type", "Number of vae vectors", "Dimension of vae vector", "MLE intrinsic dimension", "MLE elapsed time", "Twonn Intrinsic dimension", "Twonn elapsed time"])
            
        writer.writeheader()

        for data_type in data_type_list:

            # load feature data from environment dataset
            feature_data = load_featurs_data(minio_client, data_type, max_count, args.dataset)
            if len(feature_data) == 0:
                raise Exception("Failed the loading of feature data")

            for count in count_list:

                # get specific count of data for gettting intrinsic dimension
                data = torch.tensor(feature_data[:count], device=device)

                # wrangle the latent vector [1, 4, 64, 64]
                if data_type == "vae":
                    data = data.reshape((data.size(0), -1))
                
                if args.library == Library.INTRINSIC_DIMENSION.value:

                    dimension_by_mle, mle_elapsed_time = \
                        measure_running_time(mle_id, data, k=2)

                    dimension_by_twonn_numpy, twonn_numpy_elapsed_time = \
                        measure_running_time(twonn_numpy, data.cpu().numpy(), return_xy=False)

                    dimension_by_twonn_torch, twonn_pytorch_elapsed_time = \
                        measure_running_time(twonn_pytorch, data, return_xy=False)

                    writer.writerow({
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

                elif args.library == Library.SCIKIT_DIMENSION.value:
                    data = data.cpu().numpy()

                    dimension_by_mle, mle_elapsed_time = measure_running_time(skdim.id.MLE().fit, data)
                    dimension_by_twonn_numpy, twonn_elapsed_time = measure_running_time(skdim.id.TwoNN().fit, data)

                    writer.writerow({
                        "Data type": "Clip vector" if data_type == "clip" else "VAE",
                        "Number of vae vectors": data.shape[0],
                        "Dimension of vae vector": data.shape[1],
                        "MLE intrinsic dimension": "{:.2f}".format(dimension_by_mle.dimension_),
                        "MLE elapsed time": "{}".format(format_duration(mle_elapsed_time)),
                        "Twonn Intrinsic dimension": "{:.2f}".format(dimension_by_twonn_numpy.dimension_),
                        "Twonn elapsed time": "{}".format(format_duration(twonn_elapsed_time))
                    })
                
        file.flush()

if __name__ == "__main__":
    main()