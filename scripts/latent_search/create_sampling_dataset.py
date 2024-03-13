import argparse
import os
import sys
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import torch

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from data_loader.kandinsky_dataset_loader import KandinskyDatasetLoader

def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--minio-access-key', type=str, help='Minio access key')
        parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
        parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")
        parser.add_argument('--n-spheres', type=int, help='Number of spheres to generate', default=1000)
        parser.add_argument('--avg-points-per-sphere', type=int, help='Average number of datapoints per sphere', default=5)

        return parser.parse_args()

def adjust_sphere_radii(points, centers, target_avg=5):
    """
    Adjust the radii of spheres centered at `centers` to enclose an average of `target_avg` points per sphere.
    """
    n_spheres = len(centers)
    distances = cdist(points, centers, 'euclidean')
    sorted_distances = np.sort(distances, axis=0)
    
    # Estimate radius to enclose target_avg points, might be adjusted for better distribution
    target_radii = sorted_distances[target_avg, range(n_spheres)]
    return target_radii

def assign_points_to_spheres(points, centers, radii):
    """
    Assign points to spheres based on centers and radii.
    A point can belong to multiple spheres due to overlap.
    """
    distances = cdist(points, centers, 'euclidean')
    sphere_assignments = [[] for _ in range(len(centers))]
    
    for i, point_distances in enumerate(distances):
        for sphere_index, radius in enumerate(radii):
            if point_distances[sphere_index] <= radius:
                sphere_assignments[sphere_index].append(i)
                
    return sphere_assignments

def create_sphere_dataset(clip_vectors, n_spheres, target_avg=5):
    """
    Create spheres around clip vectors with an approximate average of `target_avg` points per sphere.
    """
    # Convert clip vectors to numpy array if they aren't already
    clip_vectors = np.stack(clip_vectors) if isinstance(clip_vectors[0], torch.Tensor) else np.array(clip_vectors)

    # Step 1: Cluster to find sphere centers
    kmeans = KMeans(n_clusters=n_spheres, random_state=42).fit(clip_vectors)
    sphere_centers = kmeans.cluster_centers_
    
    # Step 2: Adjust sphere radii
    sphere_radii = adjust_sphere_radii(clip_vectors, sphere_centers, target_avg=target_avg)
    
    # Step 3: Assign points to spheres
    sphere_assignments = assign_points_to_spheres(clip_vectors, sphere_centers, sphere_radii)
    
    return sphere_centers, sphere_radii, sphere_assignments

def main():
    args= parse_args()

    dataloader= KandinskyDatasetLoader(minio_access_key=args.minio_access_key,
                                       minio_secret_key=args.minio_secret_key,
                                       dataset=args.dataset)
    
    dataset= dataloader.load_clip_vector_data()

    clip_vectors=[data['input_clip'] for data in dataset]

    sphere_centers, sphere_radii, sphere_assignments = create_sphere_dataset(clip_vectors, args.n_spheres, args.avg_points_per_sphere)

    # Debug: Print the number of points per sphere to check distribution
    points_per_sphere = [len(assignments) for assignments in sphere_assignments]
    print("Points per sphere:", points_per_sphere)
    print("Average points per sphere:", np.mean(points_per_sphere))