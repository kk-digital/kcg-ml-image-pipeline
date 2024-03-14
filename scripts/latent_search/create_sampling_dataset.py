import argparse
from io import BytesIO
import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

    # Reassign points to spheres based on the new radii
    sphere_assignments = {i: [] for i in range(n_spheres)}
    distances_to_centers = cdist(clip_vectors_reduced, sphere_centers)
    for idx, distances in enumerate(distances_to_centers):
        for sphere_index, radius in enumerate(radii):
            if distances[sphere_index] <= radius:
                sphere_assignments[sphere_index].append(idx)
    
    return clip_vectors_reduced ,sphere_centers, radii, sphere_assignments
    
def plot(minio_client, sphere_assignments, centers, scores):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    # Preparing data for plots
    n_spheres = len(centers)
    points_per_cluster = [len(sphere_assignments[i]) for i in range(n_spheres)]
    mean_scores = [np.mean([scores[j] for j in sphere_assignments[i]]) if sphere_assignments[i] else 0 for i in range(n_spheres)]
    
    # Scatter Plot of Clusters with Centers and Radii
    # for label, assignments in sphere_assignments.items():
    #     if assignments:  # Check if there are points assigned to the sphere
    #         cluster_points = clip_vectors_reduced[assignments]
    #         axs[0].scatter(cluster_points[:, 0], cluster_points[:, 1], alpha=0.5, edgecolor='k')
    
    # for i, (center, radius) in enumerate(zip(centers, radii)):
    #     circle = plt.Circle(center[:2], radius, color='r', fill=False, lw=2, ls='--')
    #     axs[0].add_artist(circle)
    #     axs[0].text(center[0], center[1], str(i), color='red', fontsize=12)
    # axs[0].set_title('Clusters with Centers and Radii')
    # axs[0].set_xlabel('Dimension 1')
    # axs[0].set_ylabel('Dimension 2')
    
    # Histogram of Points per Cluster
    axs[0].bar(range(n_spheres), points_per_cluster, color='skyblue')
    axs[0].set_xlabel('Cluster ID')
    axs[0].set_ylabel('Number of Points')
    axs[0].set_title('Reassigned Points per Cluster')
    
    # Scatter Plot of Cluster Density vs. Mean Score
    axs[1].scatter(points_per_cluster, mean_scores, c='blue', marker='o')
    axs[1].set_xlabel('Cluster Density (Number of Points)')
    axs[1].set_ylabel('Mean Score')
    axs[1].set_title('Cluster Density vs. Mean Score After Reassignment')
    axs[1].grid(True)
    
    plt.tight_layout()

    # Save the figure to a file
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # upload the graph report
    cmd.upload_data(minio_client, 'datasets', "environmental/output/sphere_dataset/graphs.png", buf)  

    # Clear the current figure
    plt.clf()

def main():
    args= parse_args()

    dataloader= KandinskyDatasetLoader(minio_access_key=args.minio_access_key,
                                       minio_secret_key=args.minio_secret_key,
                                       dataset=args.dataset)
    
    dataset= dataloader.load_clip_vector_data()

    clip_vectors=[data['input_clip'][0].cpu().numpy().tolist() for data in dataset]
    scores=[data['score'] for data in dataset]

    n_components= args.n_components if args.reduce_dimensions else None
    clip_vectors_reduced, sphere_centers, sphere_radii, sphere_assignments = create_sphere_dataset(clip_vectors, args.n_spheres, args.n_components)

    plot(dataloader.minio_client, sphere_assignments, sphere_centers, scores)

    # Debug: Print the number of points per sphere to check distribution
    points_per_sphere = [len(sphere_assignments[i]) for i in range(args.n_spheres)]
    print("Points per sphere:", points_per_sphere)
    print("Average points per sphere:", np.mean(points_per_sphere))

if __name__ == "__main__":
    main()
