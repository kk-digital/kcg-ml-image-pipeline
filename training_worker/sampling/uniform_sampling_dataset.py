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
        feature_vectors= np.array(feature_vectors, dtype='float32')
     
        # Calculate max and min vectors
        max_vector = np.max(feature_vectors, axis=0)
        min_vector = np.min(feature_vectors, axis=0)

        print("generating the initial spheres-------------")
        # Generate random values between 0 and 1, then scale and shift them into the [min, max] range for each feature
        sphere_centers = np.random.rand(n_spheres, len(max_vector)) * (max_vector - min_vector) + min_vector
        # Convert sphere_centers to float32
        sphere_centers = sphere_centers.astype('float32')

        d = feature_vectors.shape[1]
        index = faiss.IndexFlatL2(d)
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
        # Prepare to collect sphere data and statistics
        sphere_data = []
        total_covered_points = set()
        
        # Perform a range search for each valid sphere to find points within its radius
        for center, radius in zip(valid_centers, valid_radii):
            # Convert center to a query matrix of shape (1, d) for FAISS
            query_matrix = center.reshape(1, d).astype('float32')
            
            # Perform the range search
            lims, D, I = index.range_search(query_matrix, radius)
            
            # Extract indices of points within the radius
            point_indices = I[lims[0]:lims[1]]
            
            # Update sphere data and covered points
            sphere_data.append({'center': center, 'radius': radius, 'points': point_indices})
            total_covered_points.update(point_indices)
        
        # Calculate statistics
        points_per_sphere = [len(sphere['points']) for sphere in sphere_data]
        print(points_per_sphere)
        avg_points_per_sphere = np.mean(points_per_sphere) if points_per_sphere else 0
        
        return sphere_data, avg_points_per_sphere, len(total_covered_points)

    
def main():
    args= parse_args()

    generator= UniformSphereGenerator(minio_access_key=args.minio_access_key,
                                    minio_secret_key=args.minio_secret_key,
                                    dataset=args.dataset)
    
    sphere_data, avg_points_per_sphere, total_covered_points= generator.generate_spheres(n_spheres=args.n_spheres,
                                                       target_avg_points= args.target_avg_points)
    
    print(f"average points per sphere: {avg_points_per_sphere}")
    print(f"total points covered: {total_covered_points}")

if __name__ == "__main__":
    main()


