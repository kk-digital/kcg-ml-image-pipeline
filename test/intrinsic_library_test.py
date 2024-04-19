import argparse
import os
import sys
import torch
from intrinsics_dimension import mle_id, twonn_numpy, twonn_pytorch
base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from data_loader.kandinsky_dataset_loader import KandinskyDatasetLoader
from utility.minio import cmd

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--minio-addr', type=str, help='Minio address')
    parser.add_argument('--dataset', default="environmental", type=str, help='Name of the dataset')

    return parser.parse_args()

def main():
    args = parse_args()

    # get minio client
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)
    
    dataloader = KandinskyDatasetLoader(minio_client=minio_client, dataset=args.dataset)
    feature_vectors, scores= dataloader.load_clip_vector_data(limit=500000)
    list_clip_vector_num = [1024, 4096, 500000]
    result = []
    for clip_vector_num in list_clip_vector_num:
        data = torch.tensor(feature_vectors)

        d1 = mle_id(data, k=2)
        d2 = twonn_numpy(data.numpy(), return_xy=False)
        d3 = twonn_pytorch(data, return_xy=False)

        result.append({
            "number of clip vector": clip_vector_num,
            "dimension": data.size(1),
            "mle_id": d1,
            "twonn_numpy": d2,
            "twonn_pytorch": d3
        })
    print(result)

if __name__ == "__main__":
    main()