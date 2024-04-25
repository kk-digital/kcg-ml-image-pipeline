import argparse
import os
import sys
import torch
from intrinsics_dimension import mle_id, twonn_numpy, twonn_pytorch
base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from utility.minio import cmd
from utility.http import request
from utility.path import separate_bucket_and_file_path
from data_loader.utils import get_object
import time

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--minio-addr', type=str, help='Minio address')
    parser.add_argument('--data-type', type=str, default="clip", help='Data types to obtain intrinsic dimensions, for exmaple, clip and vae')
    parser.add_argument('--num-vectors', type=int, default=1024, help="Number of vectors to get intrinsic dimensions")
    return parser.parse_args()

def load_vectors(minio_client, dataset_names, vector_type="clip",  limit=1024):
    num_loaded_vectors = 0

    feature_vectors = []

    for dataset_name in dataset_names:

        if num_loaded_vectors >= limit:
            break

        jobs = request.http_get_completed_job_by_dataset(dataset=dataset_name)
        for job in jobs:

            if num_loaded_vectors >= limit:
                break
            
            path = job.get("task_output_file_dict", {}).get("output_file_path")
            if path:
                if vector_type == "clip":
                    path = path.replace(".jpg", "_embedding.msgpack")
                elif vector_type == "vae":
                    path = path.replace(".jpg", "_vae_latent.msgpack")
                bucket, features_vector_path = separate_bucket_and_file_path(path)
                try:
                    feature_vector = get_object(minio_client, features_vector_path)
                    feature_vectors.append(feature_vector)

                    num_loaded_vectors += 1
                except Exception as e:
                    print("Error in loading feature vector: {}, {}".format(path, e))

    return feature_vectors
        
def main():
    args = parse_args()

    # get minio client
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)
    dataset_names = request.http_get_dataset_names()
    
    test_clip_vector_num_list = [args.num_vectors]
    max_test_clip_vector_num = max(test_clip_vector_num_list)

    all_feature_vectors = []
    
    all_feature_vectors = load_vectors(minio_client=minio_client, dataset_names=dataset_names, vector_type=args.data_type, limit=max_test_clip_vector_num)

    result = []

    for clip_vector_num in test_clip_vector_num_list:
        data = torch.tensor(all_feature_vectors[:clip_vector_num])

        print("shape", data.size())
        
        start_time = time.time()
        d1 = mle_id(data, k=2)
        mle_elapsed_time = time.time() - start_time
        start_time = time.time()
        d2 = twonn_numpy(data.numpy(), return_xy=False)
        twonn_numpy_elapsed_time = time.time()
        start_time = time.time()
        d3 = twonn_pytorch(data, return_xy=False)
        twonn_torch_elapsed_time = time.time() - start_time

        result.append({
            "number of clip vector": data.size(0),
            "dimension of clip vector": data.size(1),
            "dimension": data.size(1),
            "mle_id": {
                "intrinsic dimension": d1,
                "elapsed time": mle_elapsed_time
            },
            "twonn_numpy": {
                "intrinsic dimension": d2,
                "elapsed time": twonn_numpy_elapsed_time,
            },
            "twonn_pytorch": {
                "intrinsic dimension": d3,
                "elapsed time": twonn_torch_elapsed_time
            },
        })

    print(result)

if __name__ == "__main__":
    main()