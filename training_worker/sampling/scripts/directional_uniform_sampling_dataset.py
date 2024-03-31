import argparse
from io import BytesIO
import math
import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import faiss
from tqdm import tqdm

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from data_loader.kandinsky_dataset_loader import KandinskyDatasetLoader
from utility.minio import cmd

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")
    parser.add_argument('--target-avg-points', type=int, help='Target average of datapoints per sphere', 
                        default=5)
    parser.add_argument('--n-spheres', type=int, help='Number of spheres', default=100000)
    parser.add_argument('--num-bins', type=int, help='Number of score bins', default=8)
    parser.add_argument('--bin-size', type=int, help='Range of each bin', default=1)

    return parser.parse_args()

class DirectionalUniformSphereGenerator:
    def __init__(self,
                 minio_client,
                 dataset):
        
        self.dataloader= KandinskyDatasetLoader(minio_client=minio_client,
                                                dataset=dataset)
        self.scores=[]
        self.feature_vectors=[]
        self.max_vector=None
        self.min_vector=None
        
        self.minio_client= minio_client
        self.dataset= dataset

    def load_data(self):
        # load data from mongodb
        feature_vectors, scores= self.dataloader.load_clip_vector_data()
        # set features vectors and scores
        self.feature_vectors= np.array(feature_vectors, dtype='float32')
        self.scores = scores
        # set min and max vectors
        self.max_vector = np.max(feature_vectors, axis=0)
        self.min_vector = np.min(feature_vectors, axis=0)

    def generate_spheres(self, n_spheres, target_avg_points, output_type="score_distribution", num_bins=8, bin_size=1):
        # checking if data has been loaded
        if len(self.feature_vectors)==0:
            raise Exception("You must load datapoints first before generating spheres.")
        
        print("generating the initial spheres-------------")
        # Generate random values between 0 and 1, then scale and shift them into the [min, max] range for each feature
        sphere_centers = np.random.rand(n_spheres, len(self.max_vector)) * (self.max_vector - self.min_vector) + self.min_vector
        # Convert sphere_centers to float32
        sphere_centers = sphere_centers.astype('float32')

        res = faiss.StandardGpuResources()

        d = self.feature_vectors.shape[1]
        # build a flat (CPU) index
        index_flat = faiss.IndexFlatL2(d)
        # make it into a gpu index
        index = faiss.index_cpu_to_gpu(res, 0, index_flat)

        index.add(self.feature_vectors)
        
        print("Searching for k nearest neighbors for each sphere center-------------")
        # Search for the k nearest neighbors of each sphere center in the dataset
        distances, indices = index.search(sphere_centers, target_avg_points)

        # The scaling factors of each sphere are the max distance in each feature
        scaling_factors= []
        for i, sphere_indices in enumerate(indices):
            # get neighbors for each sphere
            neighbors= self.feature_vectors[sphere_indices]
            scaling_factor= np.max(np.abs(sphere_centers[i] - neighbors), axis=0)
            # get max distance for each sphere
            scaling_factors.append(scaling_factor)

        print("Processing sphere data-------------")
        # Prepare to collect sphere data and statistics
        total_covered_points = set()
        inputs=[]
        targets=[]
        
        # prepare array for score bins
        if output_type =="score_distribution":
            bins=[]
            for i in range(num_bins-1):
                max_score= int((i+1-(num_bins/2)) * bin_size)
                bins.append(max_score)
            
            bins.append(np.inf)

        # Assuming 'scores' contains the scores for all points and 'bins' defines the score bins
        for center, scaling_factor, sphere_indices in zip(sphere_centers, scaling_factors, indices):
            # Extract indices and scores of points within the sphere
            point_indices = sphere_indices
            sphere_scores=[self.scores[idx] for idx in point_indices]

            if output_type=="score_distribution":            
                # Calculate score distribution for the sphere
                score_distribution = np.zeros(len(bins))
                for score in sphere_scores:
                    for i, bin_edge in enumerate(bins):
                        if score < bin_edge:
                            score_distribution[i] += 1
                            break

                # Normalize the score distribution by the number of points in the sphere
                target = score_distribution / len(point_indices)
            elif output_type=="mean_sigma_score":
                target= np.mean(sphere_scores)
            elif output_type=="variance":
                target= np.var(sphere_scores)
            
            # Update inputs and targets
            inputs.append(np.append(center, scaling_factor))
            targets.append(target)
            total_covered_points.update(point_indices)

        print(f"total datapoints: {len(total_covered_points)}")
        
        return inputs, targets

def main():
    args= parse_args()

    # get minio client
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key)

    generator= DirectionalUniformSphereGenerator(minio_client=minio_client,
                                    dataset=args.dataset)
    
    inputs, outputs = generator.generate_spheres(num_bins= args.num_bins,
                                                    bin_size= args.bin_size,
                                                    n_spheres=args.n_spheres,
                                                    target_avg_points= args.target_avg_points)
    
    
if __name__ == "__main__":
    main()