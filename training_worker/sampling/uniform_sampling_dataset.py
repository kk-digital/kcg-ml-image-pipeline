import argparse
from io import BytesIO
import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist

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

    def generate_initial_spheres(self, feature_vectors, n_spheres):
        """
        Generate initial spheres with random centers from data points and random radii.
        """
        # Calculate max and min vectors
        max_vector = np.max(feature_vectors, axis=0)
        min_vector = np.min(feature_vectors, axis=0)

        # Generate random values between 0 and 1, then scale and shift them into the [min, max] range for each feature
        initial_centers = np.random.rand(n_spheres, len(max_vector)) * (max_vector - min_vector) + min_vector

        distances = cdist(feature_vectors, initial_centers, 'euclidean')

         # Find the smallest and second smallest unique distances
        unique_distances = np.unique(distances)
        if len(unique_distances) >= 2:
            min_distance, second_min_distance = unique_distances[:2]
        else:
            # In the unlikely event there's only one unique distance, use it for both
            min_distance = second_min_distance = unique_distances[0]
        
        # Select random radii within the range defined by the two smallest distances
        initial_radii = np.random.uniform(low=min_distance, high=second_min_distance, size=n_spheres)

        return initial_centers, initial_radii

    def evaluate_spheres(data, centers, radii, target_avg_points):
        """
        Evaluate spheres based on coverage and average points per sphere.
        Returns the average deviation from the target average points.
        """
        distances = cdist(data, centers, 'euclidean')
        coverage = distances <= radii[:, np.newaxis]
        points_per_sphere = np.sum(coverage, axis=1)
        avg_deviation = np.abs(np.mean(points_per_sphere) - target_avg_points)
        return avg_deviation, points_per_sphere

    def adjust_spheres(centers, radii, adjustment_factor=0.1):
        """
        Randomly adjust sphere radii within a factor of their current size.
        """
        radii_adjustment = np.random.uniform(-adjustment_factor, adjustment_factor, size=radii.shape) * radii
        adjusted_radii = np.clip(radii + radii_adjustment, a_min=0, a_max=None)  # Ensure radii remain positive
        return adjusted_radii

    def monte_carlo_optimization(self, n_spheres, target_avg_points, iterations=1000):
        # load data from mongodb
        feature_vectors, scores= self.dataloader.load_clip_vector_data()
        feature_vectors= np.array(feature_vectors)

        # generate initial spheres
        centers, radii = self.generate_initial_spheres(feature_vectors, n_spheres)
        best_deviation, _ = self.evaluate_spheres(feature_vectors, centers, radii, target_avg_points)
        best_radii = radii

        for _ in range(iterations):
            adjusted_radii = self.adjust_spheres(centers, radii)
            deviation, avg_density = self.evaluate_spheres(feature_vectors, centers, adjusted_radii, target_avg_points)
            
            if deviation < best_deviation:
                best_deviation = deviation
                best_radii = adjusted_radii

                print(f"Current best sphere density is : {avg_density}")
        
        return centers, best_radii
    
def main():
    args= parse_args()

    generator= UniformSphereGenerator(minio_access_key=args.minio_access_key,
                                    minio_secret_key=args.minio_secret_key,
                                    dataset=args.dataset)
    
    centers, radii= generator.monte_carlo_optimization(n_spheres=args.n_spheres,
                                                       target_avg_points= args.target_avg_points)

if __name__ == "__main__":
    main()


