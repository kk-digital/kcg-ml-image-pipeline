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
from test.intrinsic_dimensionality_test.utils import http_get_dataset_names

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--minio-addr', type=str, help='Minio address')
    parser.add_argument('--library', type=int, default=0, help='Lirary used for getting intrinsic dimension of data, ["intrinsic_dimension", "scipy"], now there are two available libraries')
    parser.add_argument('--data-type-list', type=str, nargs='+', default=['clip'], help='Data types to obtain intrinsic dimensions, for exmaple, clip and vae')
    parser.add_argument('--count-list', type=int, nargs='+', default=[100], help="list of count for getting intrinsic dimension")
    parser.add_argument('--dataset', type=str, default="environmental", help="Dataset name")
    parser.add_argument('--time-period', type=int, default=None, help="The time_period is a parameter that allows you to specify the time frame for the data you want to retrieve")
    return parser.parse_args()

class IntrinsicDimensionaltiyAnalysis:
    
    def __init__(self, minio_client, library, data_type_list, count_list, dataset, time_period) -> None:
        self.minio_client = minio_client
        self.library = library
        self.data_type_list = data_type_list
        self.count_list = count_list
        self.dataset = dataset
        self.time_period = time_period

        # max count of clip vectors for testing intrinsic dimension
        self.max_count = max(count_list) * 2

    def load_featurs_data(self, data_type):
        featurs_data = []

        try:
            dataloader = KandinskyDatasetLoader(minio_client=self.minio_client, 
                                                dataset=self.dataset, 
                                                time_period=self.time_period)
        except Exception as e:
            print("Error in initializing kankinsky dataset loader:{}".format(e))
            return

        # get features data depends on data type
        if data_type == "clip":
            featurs_data = dataloader.load_clip_vector_data(limit=self.max_count)
        elif data_type == "vae":
            featurs_data = dataloader.load_latents(limit=self.max_count)
        else:
            print("No support data type {}".format(data_type))
            return featurs_data
        
        return featurs_data

    def get_file_name(self, *args, seperator='_'):
        
        return os.path.join(os.getcwd(), 
                            "output", 
                            "{}_intrinsic_dim_results_{}.csv".format(
                                datetime.now(), seperator.join(map(str, args))))

    def get_intrinsic_dimenstions(self):
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

        result = []

        for data_type in self.data_type_list:

            # load feature data from environment dataset
            feature_data = self.load_featurs_data(data_type=data_type)
            if len(feature_data) == 0:
                print("Error loading feature data from {} dataset".format(self.dataset))
                return result
            
            feature_data = torch.tensor(feature_data, dtype=torch.float32).squeeze()
            filtered_feature_data = torch.empty((0, feature_data.size(1)))
            # Add validation for nan and inf values
            for feature in feature_data:
                feature = feature.reshape(1, -1)
                if not (torch.isnan(feature).any() or torch.isinf(feature).any()):
                    filtered_feature_data = torch.concat((filtered_feature_data,
                                                         feature), dim=0)
            
            # Assign the filtered_feature_data into feature_data
            feature_data = filtered_feature_data

            for count in self.count_list:
                try:
                    # get specific count of data for gettting intrinsic dimension
                    data = feature_data[:count]

                    # wrangle the latent vector [1, 4, 64, 64]
                    if data_type == "vae":
                        data = data.reshape((data.size(0), -1))
                    
                    if self.library == Library.INTRINSIC_DIMENSION.value:
                        print("Dimension", data.size())
                        dimension_by_mle, mle_elapsed_time, mle_error = \
                            measure_running_time(mle_id, data, k=2)
                        
                        dimension_by_twonn_numpy, twonn_numpy_elapsed_time, twonn_numpy_error = \
                            measure_running_time(twonn_numpy, data.cpu().numpy(), return_xy=False)

                        dimension_by_twonn_torch, twonn_pytorch_elapsed_time, twonn_torch_error = \
                            measure_running_time(twonn_pytorch, data, return_xy=False)

                        result.append({
                            "Dataset": self.dataset,
                            "Dataset type": "clip-vector-1280" if data_type == "clip" else "VAE",
                            "Number of vector": data.size(0),
                            "Dimension of vector": data.size(1),
                            "Metrics Field": "mle-intrinsic-dimension",
                            "Intrinsic dimension": "{:.2f}".format(dimension_by_mle) if dimension_by_mle is not None else "None",
                            "Elapsed time": "{}".format(format_duration(mle_elapsed_time)),
                            "Error": mle_error if mle_error else ''
                        })

                        result.append({
                            "Dataset": self.dataset,
                            "Dataset type": "clip-vector-1280" if data_type == "clip" else "VAE",
                            "Number of vector": data.size(0),
                            "Dimension of vector": data.size(1),
                            "Metrics Field": "twonn-numpy-intrinsic-dimension",
                            "Intrinsic dimension": "{:.2f}".format(dimension_by_twonn_numpy) if dimension_by_twonn_numpy is not None else "None",
                            "Elapsed time": "{}".format(format_duration(twonn_numpy_elapsed_time)),
                            "Error": twonn_numpy_error if twonn_numpy_error else ''
                        })

                        result.append({
                            "Dataset": self.dataset,
                            "Dataset type": "clip-vector-1280" if data_type == "clip" else "VAE",
                            "Number of vector": data.size(0),
                            "Dimension of vector": data.size(1),
                            "Metrics Field": "twonn-torch-intrinsic-dimension",
                            "Intrinsic dimension": "{:.2f}".format(dimension_by_twonn_torch) if dimension_by_twonn_torch is not None else "None",
                            "Elapsed time": "{}".format(format_duration(twonn_pytorch_elapsed_time)),
                            "Error": twonn_torch_error if twonn_torch_error else ''
                        })

                    elif self.library == Library.SCIKIT_DIMENSION.value:
                        data = data.cpu().numpy()
                        dimension_by_mle, mle_elapsed_time, mle_error = measure_running_time(skdim.id.MLE().fit, data)
                        dimension_by_twonn_numpy, twonn_elapsed_time, twonn_error = measure_running_time(skdim.id.TwoNN().fit, data)

                        if dimension_by_mle is not None:
                            dimension_by_mle = round(dimension_by_mle.dimension_, 2)
                        if dimension_by_twonn_numpy is not None:
                            dimension_by_twonn_numpy = round(dimension_by_twonn_numpy.dimension_, 2)

                        result.append({
                                "Dataset": self.dataset,
                                "Dataset type": "clip-h-1280" if data_type == "clip" else "VAE",
                                "Number of vector": data.shape[0],
                                "Dimension of vector": data.shape[1],
                                "Metrics Field": 'scipy-mle-intrinsic',
                                "Intrinsic dimension": "{}".format(dimension_by_mle),
                                "Elapsed time": "{}".format(format_duration(mle_elapsed_time)),
                                "Error": mle_error if mle_error else ''
                        })

                        result.append({
                                "Dataset": self.dataset,
                                "Dataset type": "clip-h-1280" if data_type == "clip" else "VAE",
                                "Number of vector": data.shape[0],
                                "Dimension of vector": data.shape[1],
                                "Metrics Field": 'scipy-twonn-intrinsic',
                                "Intrinsic dimension": "{}".format(dimension_by_twonn_numpy),
                                "Elapsed time": "{}".format(format_duration(twonn_elapsed_time)),
                                "Error": twonn_error if twonn_error else ''
                        })

                except Exception as e:
                    print("Error in getting intrinsic dimension", e)

        return result

    def run(self):
        
        df = pd.DataFrame(columns=["Dataset", 
                                   "Dataset type", 
                                   "Number of vector", 
                                   "Dimension of vector", 
                                   "Metrics Field", 
                                   "Intrinsic dimension",
                                   "Elapsed time", "Error"])

        if self.dataset == 'all':
            dataset_names = http_get_dataset_names()

            for dataset in dataset_names:
                self.dataset = dataset
                print("Getting intrinsic dimension for dataset: {}".format(dataset))
                results = self.get_intrinsic_dimenstions()

                for result in results:
                    df.loc[len(df)] = result

                print("Getted intrinsic dimension for dataset: {}".format(dataset))
        
        else:
            results = self.get_intrinsic_dimenstions()
            for result in results:
                df.loc[len(df)] = result

        df = df.sort_values(['Dataset type', 'Number of vector'])
        df = df.assign(Transform='none')
        df = df.assign(Time_period='last {} days'.format(self.time_period) if self.time_period is not None else 'all')

        df = df.loc[:,["Dataset", 
                "Dataset type", 
                "Number of vector", 
                "Dimension of vector", 
                "Metrics Field", 
                "Intrinsic dimension",
                "Elapsed time", "Transform", "Time_period", "Error"]]

        data_type_list = df['Dataset type'].unique().tolist()
        
        for data_type in data_type_list:
            df[df['Dataset type'] == data_type].to_csv(self.get_file_name(data_type), index=False)
            

def main():
    args = parse_args()

    # get minio client
    minio_client = get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)
    
    intrinsic_dimension_analysis = IntrinsicDimensionaltiyAnalysis(minio_client=minio_client,
                                                                    library=args.library,
                                                                    data_type_list=args.data_type_list,
                                                                    count_list=args.count_list,
                                                                    dataset=args.dataset,
                                                                    time_period=args.time_period)
    
    intrinsic_dimension_analysis.run()


if __name__ == "__main__":
    main()