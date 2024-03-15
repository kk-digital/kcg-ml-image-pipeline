import argparse
import os
import sys
import numpy as np
import faiss

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from data_loader.kandinsky_dataset_loader import KandinskyDatasetLoader

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")
    parser.add_argument('--target-avg-points', type=int, help='Target average of datapoints per sphere', 
                        default=5)
    parser.add_argument('--n-spheres', type=int, help='Number of spheres', default=100000)

    return parser.parse_args()

class UniformSphereGenerator:
    def __init__(self,
                 minio_access_key,
                 minio_secret_key,
                 dataset):
        
        self.dataloader= KandinskyDatasetLoader(minio_access_key=minio_access_key,
                                       minio_secret_key=minio_secret_key,
                                       dataset=dataset)
        
        self.minio_client= self.dataloader.minio_client
        self.dataset= dataset

    def generate_spheres(self, n_spheres, target_avg_points , discard_threshold=None):
        """
        Determine sphere radii and coverage metrics using FAISS, with separate lists for datapoints and sphere centers.

        Parameters:
        - data: numpy array of shape (n_datapoints, n_features), the datapoints.
        - sphere_centers: numpy array of shape (n_spheres, n_features), the randomly selected sphere centers.
        - target_avg_points: int, number of nearest neighbors to consider for determining the sphere's radius.
        - discard_threshold: float, minimum acceptable distance to the nearest datapoint for a sphere not to be discarded.

        Returns:
        - valid_centers: List of sphere centers not discarded.
        - valid_radii: Corresponding list of radii for the valid centers.
        - avg_points_per_sphere: Average number of points per valid sphere.
        - total_covered_points: Total number of unique points covered by the valid spheres.
        """
        
        # load data from mongodb
        feature_vectors, scores= self.dataloader.load_clip_vector_data()
        feature_vectors= np.array(feature_vectors)
     
        # Calculate max and min vectors
        max_vector = np.max(feature_vectors, axis=0)
        min_vector = np.min(feature_vectors, axis=0)

        print("generating the initial spheres-------------")
        # Generate random values between 0 and 1, then scale and shift them into the [min, max] range for each feature
        sphere_centers = np.random.rand(n_spheres, len(max_vector)) * (max_vector - min_vector) + min_vector

        d = feature_vectors.shape[1]
        index = faiss.IndexFlatL2(d)
        
        # Move the index to the GPU
        # Note: 0 is the GPU ID, change it if you have multiple GPUs and want to use a different one
        index = faiss.index_cpu_to_gpus_list(index, gpus=[0])
        index.add(feature_vectors)
        
        print("Searching for k nearest neighbors-------------")
        # Search for the k nearest neighbors of each sphere center in the dataset
        distances, indices = index.search(sphere_centers, target_avg_points)

        # The radius of each sphere is the distance to the k-th nearest neighbor
        radii = distances[:, -1]
        
        # Determine which spheres to keep based on the discard threshold
        if discard_threshold is not None:
            valid_mask = radii < discard_threshold
            valid_centers = sphere_centers[valid_mask]
            valid_radii = radii[valid_mask]
            indices = indices[valid_mask]
        else:
            valid_centers = sphere_centers
            valid_radii = radii

        print("Calculating metrics for each sphere-------------")
        # Calculate coverage metrics
        unique_indices = np.unique(indices)
        total_covered_points = len(unique_indices)
        avg_points_per_sphere = total_covered_points / len(valid_centers) if len(valid_centers) > 0 else 0

        return valid_centers, valid_radii, avg_points_per_sphere, total_covered_points

    
def main():
    args= parse_args()

    generator= UniformSphereGenerator(minio_access_key=args.minio_access_key,
                                    minio_secret_key=args.minio_secret_key,
                                    dataset=args.dataset)
    
    valid_centers, valid_radii, avg_points_per_sphere, total_covered_points= generator.generate_spheres(n_spheres=args.n_spheres,
                                                       target_avg_points= args.target_avg_points)
    
    print(f"average points per sphere: {avg_points_per_sphere}")
    print(f"total points covered: {total_covered_points}")

if __name__ == "__main__":
    main()


