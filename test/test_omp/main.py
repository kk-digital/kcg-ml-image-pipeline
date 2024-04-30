import numpy as np

# intrinsic library
import skdim

# utils
import argparse

# feature selection library
from sklearn.linear_model import OrthogonalMatchingPursuit

import sys
base_dir = './'
sys.path.insert(0, base_dir)

# clip vector loader which loads data from memory mapping of numpy
from test.load_clip_vector_over_sigma_score.clip_vector_loader import get_clip_0_sigma

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--count', type=int, default=10000, help='count of clip vectors')

    return parser.parse_args()

def main():

    args = parse_args()

    clip_vectors, scores = get_clip_0_sigma(0, args.count)
    clip_vectors = np.array(clip_vectors)
    scores = np.array(scores).reshape(-1, 1)
    clip_vectors = clip_vectors[~np.isnan(clip_vectors).any(axis=1)]
    clip_vectors = clip_vectors[~np.isinf(clip_vectors).any(axis=1)]
    
    try:
        n_features = round(skdim.id.TwoNN().fit(clip_vectors).dimension_)
        
        # Create the OMP feature selector
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_features)

        # Fit the OMP feature selector to the data
        omp.fit(clip_vectors, scores)

        clip_vectors[:, np.argsort(-np.abs(omp.coef_))[n_features:]] = 0

        print(clip_vectors[:2].tolist())
    except Exception as e:
        print('Error occured! ', e)
if __name__ == '__main__':
    main()