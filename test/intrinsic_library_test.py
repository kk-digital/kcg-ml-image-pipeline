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
from utility.http import request

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--minio-addr', type=str, help='Minio address')
    parser.add_argument('--data-type', type=str, default="clip", help='Data types to obtain intrinsic dimensions, for exmaple, clip and vae')
    parser.add_argument('--num-vectors', type=int, default=1024, help="Number of vectors to get intrinsic dimensions")
    return parser.parse_args()

def main():
    args = parse_args()

    # get minio client
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)
    dataset_names = request.http_get_dataset_names()
    all_feature_vectors = []

    list_clip_vector_num = [args.num_vectors]

    for dataset_name in dataset_names:
        try:
            dataloader = KandinskyDatasetLoader(minio_client=minio_client, dataset=dataset_name)
        except Exception as e:
            print("Error in initializing kankinsky dataset loader:{}".format(e))
            continue

        if args.data_type == "clip":
            feature_vectors, _= dataloader.load_clip_vector_data(limit=max(list_clip_vector_num))
        elif args.data_type == "vae":
            feature_vectors = dataloader.load_latents(limit=max(list_clip_vector_num))
        else:
            print("No support data type {}".format(args.data_type))
            return None
        all_feature_vectors.extend(feature_vectors)
        if len(all_feature_vectors) >= max(list_clip_vector_num):
            break
    result = []

    print("length", len(all_feature_vectors))

    for clip_vector_num in list_clip_vector_num:
        data = torch.tensor(all_feature_vectors[:clip_vector_num])
        print("shape", data.size())
        d1 = mle_id(data, k=2)
        d2 = twonn_numpy(data.numpy(), return_xy=False)
        d3 = twonn_pytorch(data, return_xy=False)

        result.append({
            "number of clip vector": data.size(0),
            "dimension of clip vector": data.size(1),
            "dimension": data.size(1),
            "mle_id": d1,
            "twonn_numpy": d2,
            "twonn_pytorch": d3,
        })
    print(result)

if __name__ == "__main__":
    main()