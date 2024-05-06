import argparse
import os
import sys

import numpy as np
import pandas as pd

from datetime import datetime

# library for PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

# load data loader for feature data
from test.test_intrinsic_dimension.dataloader import KandinskyDatasetLoader
from test.test_intrinsic_dimension.utils import get_minio_client

# import http request
from utility.http import request

# import tqdm
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--minio-addr', type=str, help='Minio address')
    
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
                        "{}_pca_result_{}.csv".format(
                            datetime.now(), seperator.join(map(str, args))))

def get_pca(minio_client, dataset, count_list, data_type_list):
    """
    PCA dataset using different feature data types.

    Args:
        minio_client (object): A MinIO client object for accessing the dataset.
        dataset (str): The name of the dataset to be analyzed.
        count_list (list): A list of testing counts for the number of clip vectors.
        data_type_list (list): A list of data types for the features.

    Returns:
        None
    """

    # max count of clip vectors for testing intrinsic dimension
    max_count = max(count_list) * 2

    result = []

    for data_type in data_type_list:

        # load feature data from environment dataset
        feature_data = load_featurs_data(minio_client, data_type, max_count, dataset)
        if len(feature_data) == 0:
            print("Error loading feature data from {} dataset".format(dataset))
            return result

        for count in count_list:
            print('Data type: {}, Sample data size'.format(data_type, count))
            try:
                # get specific count of data for gettting intrinsic dimension
                data = np.array(feature_data[:count])

                # wrangle the latent vector [1, 4, 64, 64]
                if data_type == "vae":
                    data = data.reshape((data.shape[0], -1))

                std_scaler = StandardScaler()
                scaled_df = std_scaler.fit_transform(data)

                step = round(min(count, data.shape[1]) // 20)

                for num in tqdm(range(0, count, step)):
                    pca = PCA(n_components=num)
                    pca.fit(scaled_df)
                    result.append({
                            "Dataset": dataset,
                            "Data type": "Clip vector" if data_type == "clip" else "VAE",
                            "Sample size": count,
                            "Dimension of vector": data.shape[1],
                            "n_components": num,
                            "Variance Ratio": np.sum(pca.explained_variance_ratio_)
                    })
            except Exception as e:
                print("Error in PCA", e)

    return result

def main():
    args = parse_args()

    # get minio client
    minio_client = get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)

    df = pd.DataFrame(columns=["Dataset", "Data type", "Sample size", "Dimension of vector", "n_components", "Variance Ratio"])

    if args.dataset == 'all':
        dataset_names = request.http_get_dataset_names()

        for dataset in dataset_names:
            print("Getting intrinsic dimension for dataset: {}".format(dataset))
            results = get_pca(minio_client=minio_client, 
                              dataset=dataset,
                              count_list=args.count_list, 
                              data_type_list=args.data_type_list)
            for result in results:
                df.loc[len(df)] = result

            print("Getted PCA for dataset: {}".format(dataset))
    
    else:
        results = get_pca(minio_client=minio_client, 
                              dataset=args.dataset,
                              count_list=args.count_list, 
                              data_type_list=args.data_type_list)
        for result in results:
            df.loc[len(df)] = result

    df.to_csv(get_file_name(), index=False)
        
if __name__ == "__main__":
    main()