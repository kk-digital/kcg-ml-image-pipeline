import sys
import argparse
import json
import numpy as np

base_dir = './'
sys.path.insert(0, base_dir)

from utility.minio import cmd
from utility.http import request
from test.load_clip_vector_over_sigma_score.kandinsky_data_loader import KandinskyDatasetLoader


API_URL="http://192.168.3.1:8111"

def parse_args():
    parser = argparse.ArgumentParser(description="Embedding Scorer")
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--batch-size', required=False, default=100, type=int, help='Name of the dataset for embeddings')
    parser.add_argument('--max-count', default=100, type=int, help='Count of clip vectors')
    parser.add_argument('--min-score', default=0, type=int, help='min value of score for filtering')

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

    shape = (args.max_count, 1281)
    dtype = np.float16

    # Create memory-mapped array
    filename = 'output/clip_{}_sigma.dat'.format(args.min_score)

    with open(filename, 'w+b') as f:
        mmapped_array = np.memmap(f, dtype=dtype, mode='w+', shape=shape)
        
        loaded_count = 0
        data_loader = KandinskyDatasetLoader(minio_client, mmapped_array)
        for dataset_name in dataset_names:
            
            data_loader.dataset = dataset_name
            try:
                data_loader.load_ranking_model()
            except Exception as e:
                print("Error occured in loading ranking model ", e)
                continue
            loaded_count = data_loader.load_clip_vector_data(args.min_score, limit=args.max_count)

            if args.max_count - loaded_count <= 0:
                break

        mmapped_array.flush()

        config_file = 'output/clip_{}_sigma.json'.format(args.min_score)
        
    with open(config_file, 'w') as file:
        json.dump({
            'loaded-count': loaded_count,
            'size': '{}GB'.format(round(mmapped_array.nbytes / (1024 ** 3), 4))
        }, file, indent=4)

    del mmapped_array

if __name__ == "__main__":
    main()