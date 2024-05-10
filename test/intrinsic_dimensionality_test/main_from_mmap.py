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
from test.load_clip_vector_over_sigma_score.clip_vector_loader import ClipVectorLoader

# import convert seconds into formatted time string
from utils import format_duration

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--minio-addr', type=str, help='Minio address')
    parser.add_argument('--library', type=int, default=0, help='Lirary used for getting intrinsic dimension of data, ["intrinsic_dimension", "scipy"], now there are two available libraries')
    parser.add_argument('--count-list', type=int, nargs='+', default=[100], help="list of count for getting intrinsic dimension")
    parser.add_argument('--min-sigma-score', type=int, default=0)
    return parser.parse_args()


def load_featurs_data():
    pass

def get_file_name():
    return os.path.join(os.getcwd(), "output", "intrinsic_dim_results.csv")

def main():
    args = parse_args()

    with open(get_file_name(), mode='w', newline='') as file:

        writer = csv.DictWriter(file, fieldnames=["Dataset", 
                "Dataset type", 
                "Number of vector", 
                "Dimension of vector", 
                "Metrics Field", 
                "Intrinsic dimension",
                "Elapsed time", "Transform", "Time_period",
                "Error"])
        
        writer.writeheader()

        dataloader = ClipVectorLoader(min_sigma_score=args.min_sigma_score)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for count in args.count_list:

            feature_data, _ = dataloader.get_clip_vector_by_random(count=count)
            # get specific count of data for gettting intrinsic dimension
            data = torch.tensor(feature_data, device=device)

            if args.library == Library.INTRINSIC_DIMENSION.value:

                dimension_by_mle, mle_elapsed_time, mle_error = \
                    measure_running_time(mle_id, data, k=2)

                dimension_by_twonn_numpy, twonn_numpy_elapsed_time, twonn_numpy_error = \
                    measure_running_time(twonn_numpy, data.cpu().numpy(), return_xy=False)

                dimension_by_twonn_torch, twonn_pytorch_elapsed_time, twonn_torch_error = \
                    measure_running_time(twonn_pytorch, data, return_xy=False)

                writer.writerow({
                    "Dataset": "Memory mapping",
                    "Dataset type": "clip-vector-1280",
                    "Number of vector": data.size(0),
                    "Dimension of vector": data.size(1),
                    "Metrics Field": "mle-intrinsic-dimension",
                    "Intrinsic dimension": "{:.2f}".format(dimension_by_mle) if dimension_by_mle is not None else "None",
                    "Elapsed time": "{}".format(format_duration(mle_elapsed_time)),
                    "Error": mle_error if mle_error else ''
                })

                writer.writerow({
                    "Dataset": "Memory mapping",
                    "Dataset type": "clip-vector-1280",
                    "Number of vector": data.size(0),
                    "Dimension of vector": data.size(1),
                    "Metrics Field": "twonn_numpy_intrinsic_dimension",
                    "Intrinsic dimension": "{:.2f}".format(dimension_by_twonn_numpy) if dimension_by_twonn_numpy is not None else "None",
                    "Elapsed time": "{}".format(format_duration(twonn_numpy_elapsed_time)),
                    "Error": twonn_numpy_error if twonn_numpy_error else ''
                })

                writer.writerow({
                    "Dataset": "Memory mapping",
                    "Dataset type": "clip-vector-1280",
                    "Number of vector": data.size(0),
                    "Dimension of vector": data.size(1),
                    "Metrics Field": "twonn_torch_intrinsic_dimension",
                    "Intrinsic dimension": "{:.2f}".format(dimension_by_twonn_torch) if dimension_by_twonn_torch is not None else "None",
                    "Elapsed time": "{}".format(format_duration(twonn_pytorch_elapsed_time)),
                    "Error": twonn_torch_error if twonn_torch_error else ''
                })

            elif args.library == Library.SCIKIT_DIMENSION.value:
                data = data.cpu().numpy()

                dimension_by_mle, mle_elapsed_time, mle_error = measure_running_time(skdim.id.MLE().fit, data)
                dimension_by_twonn_numpy, twonn_elapsed_time, twonn_error = measure_running_time(skdim.id.TwoNN().fit, data)

                writer.writerow({
                    "Dataset": "Memory mapping",
                    "Dataset type": "clip-h-1280",
                    "Number of vector": data.shape[0],
                    "Dimension of vector": data.shape[1],
                    "Metrics Field": 'scipy-mle-intrinsic',
                    "Intrinsic dimension": "{}".format(dimension_by_mle),
                    "Elapsed time": "{}".format(format_duration(mle_elapsed_time)),
                    "Error": mle_error if mle_error else ''
                })

                writer.writerow({
                    "Dataset": "Memory mapping",
                    "Dataset type": "clip-h-1280",
                    "Number of vector": data.shape[0],
                    "Dimension of vector": data.shape[1],
                    "Metrics Field": 'scipy-twonn-intrinsic',
                    "Intrinsic dimension": "{}".format(dimension_by_twonn_numpy),
                    "Elapsed time": "{}".format(format_duration(twonn_elapsed_time)),
                    "Error": twonn_error if twonn_error else ''
                })
            
        file.flush()

if __name__ == "__main__":
    main()