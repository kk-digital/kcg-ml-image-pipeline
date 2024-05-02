import numpy as np
import torch

# intrinsic library
import skdim

# utils
import argparse
import json

# os
import os

# feature selection library
from sklearn.linear_model import OrthogonalMatchingPursuit
from intrinsics_dimension import twonn_pytorch

import sys
base_dir = './'
sys.path.insert(0, base_dir)

# load minio utility
from utility.minio import cmd

# clip vector loader which loads data from memory mapping of numpy
from test.load_clip_vector_over_sigma_score.clip_vector_loader import ClipVectorLoader
from test.test_omp.model.scoring_fc import ScoringFCNetwork

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--count', type=int, default=100, help='count of clip vectors')
    parser.add_argument('--min_sigma_score', 
                        type=int, 
                        default=-1000, 
                        help='Min sigma score, default is -1000, it means loadding all data')
    parser.add_argument('--dataset', type=str, default='environmental', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--training-batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)

    return parser.parse_args()

def save_sparse_index(sparse_index, dataset="environmental"):

    file_path = 'output/sparse_index.json'

    # Check if the file exists
    if os.path.exists(file_path):
        # If the file exists, read the contents
        with open(file_path, mode='r+') as file:
            try:
                existed_data = json.load(file)
            except json.JSONDecodeError:
                print("Error")
                # If the file is empty or contains invalid JSON, initialize an empty dictionary
                existed_data = {}
    else:
        # If the file doesn't exist, initialize an empty dictionary
        existed_data = {}

    # Update or create the dataset in the dictionary
    existed_data[dataset] = sparse_index

    # Write the updated dictionary to the file
    with open(file_path, mode='w') as file:
        json.dump(existed_data, file)

def main():

    args = parse_args()
    
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                    minio_secret_key=args.minio_secret_key,
                                    minio_ip_addr=args.minio_addr)

    clip_vector_loader = ClipVectorLoader(min_sigma_score=-1000, dataset=args.dataset)

    # load data
    # clip_vectors, scores = clip_vector_loader.get_all_clip_vector()
    clip_vectors, scores = clip_vector_loader.get_clip_vector(start_index=0, end_index=1000)

    # remove nan, inf value which causes error
    clip_vectors = np.array(clip_vectors)
    scores = np.array(scores).reshape(-1, 1)
    clip_vectors = clip_vectors[~np.isnan(clip_vectors).any(axis=1)]
    clip_vectors = clip_vectors[~np.isinf(clip_vectors).any(axis=1)]
    
    try:
        n_features = round(twonn_pytorch(torch.tensor(clip_vectors, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu'), return_xy=False))
        print("Sparsity dimension: ", n_features)
        # Create the OMP feature selector
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_features)

        print("Start fitting OMP feature selector...")
        # Fit the OMP feature selector to the data
        omp.fit(clip_vectors, scores)
        print("OMP feature selector fitted successfully!")

        save_sparse_index(sparse_index=np.argsort(-np.abs(omp.coef_))[:n_features].tolist(), dataset=args.dataset)
        
        print("Saved sparse index successfully!")
        
        sparse_clip_vectors = clip_vectors[:, np.argsort(-np.abs(omp.coef_))[:n_features]]

        print("Training model...")
        model = ScoringFCNetwork(minio_client=minio_client, dataset="test-generations", input_size=n_features, input_type="clip-h")

        loss = model.train(inputs=np.array(sparse_clip_vectors), 
                           outputs=np.array(scores).squeeze(), 
                           num_epochs= args.epochs, 
                           batch_size=args.training_batch_size, 
                           learning_rate=args.learning_rate)
        
        model.save_model()
        print("Model trained successfully!")

    except Exception as e:
        print('Error occured! ', e)
if __name__ == '__main__':
    main()