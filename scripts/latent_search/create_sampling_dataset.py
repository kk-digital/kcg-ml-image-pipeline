import argparse
import os
import sys
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
        parser.add_argument('--reduce-dimensions', action="store_true", default=False)
        parser.add_argument('--n-components', type=int, help='Number of dimensions to reduce clip vector to', default=50)

        return parser.parse_args()


def create_sphere_dataset(clip_vectors, n_spheres, n_components=None):
    """
    Create spheres around clip vectors based on KMeans clusters, using PCA for dimensionality reduction.

    Parameters:
    - clip_vectors: numpy.ndarray, the CLIP vectors.
    - n_spheres: int, desired number of clusters/spheres.
    - n_components: int or None, the number of principal components to keep. If None, no PCA is applied.

    Returns:
    - sphere_centers: Coordinates of cluster centers in the reduced space (if PCA applied).
    - sphere_radii: Radius of each sphere needed to cover the points in the cluster.
    - sphere_assignments: Dictionary of points assigned to each sphere.
    """
    clip_vectors = np.array(clip_vectors)  # Ensure input is a numpy array
    
    # Optionally apply PCA for dimensionality reduction
    if n_components is not None:
        scaler = StandardScaler()
        clip_vectors_scaled = scaler.fit_transform(clip_vectors)
        pca = PCA(n_components=n_components, random_state=42)
        clip_vectors_reduced = pca.fit_transform(clip_vectors_scaled)
    else:
        clip_vectors_reduced = clip_vectors

    # Apply KMeans to find clusters and automatically assign points
    kmeans = KMeans(n_clusters=n_spheres, random_state=42).fit(clip_vectors_reduced)
    sphere_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Calculate radii for each sphere to cover all points in its cluster
    radii = np.zeros(len(sphere_centers))
    for i, center in enumerate(sphere_centers):
        cluster_points = clip_vectors_reduced[labels == i]
        if len(cluster_points) > 0:  # Ensure the cluster is not empty
            distances = np.linalg.norm(cluster_points - center, axis=1)
            radii[i] = np.max(distances)  # Radius is the maximum distance to the center

    # Organize points by their assigned sphere
    sphere_assignments = {i: [] for i in range(n_spheres)}
    for idx, label in enumerate(labels):
        sphere_assignments[label].append(idx)
    
    return sphere_centers, radii, sphere_assignments

def main():
    args= parse_args()

    dataloader= KandinskyDatasetLoader(minio_access_key=args.minio_access_key,
                                       minio_secret_key=args.minio_secret_key,
                                       dataset=args.dataset)
    
    dataset= dataloader.load_clip_vector_data()

    clip_vectors=[data['input_clip'][0].cpu().numpy().tolist() for data in dataset]

    n_components= args.n_components if args.reduce_dimensions else None
    sphere_centers, sphere_radii, sphere_assignments = create_sphere_dataset(clip_vectors, args.n_spheres, args.n_components)

    # Debug: Print the number of points per sphere to check distribution
    points_per_sphere = [len(assignments) for assignments in sphere_assignments]
    print("Points per sphere:", points_per_sphere)
    print("Average points per sphere:", np.mean(points_per_sphere))

if __name__ == "__main__":
    main()
