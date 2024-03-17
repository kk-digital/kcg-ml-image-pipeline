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

    return parser.parse_args()

class UniformSphereGenerator:
    def __init__(self,
                 minio_client,
                 dataset):
        
        self.dataloader= KandinskyDatasetLoader(minio_client=minio_client,
                                                dataset=dataset)
        
        self.minio_client= minio_client
        self.dataset= dataset

    def generate_spheres(self, n_spheres, target_avg_points, num_bins, bin_size, discard_threshold=None):
        
        bins=[]
        for i in range(num_bins-1):
            max_score= int((i+1-(num_bins/2)) * bin_size)
            bins.append(max_score)
        
        bins[len(num_bins)-1]= np.inf

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
        
        print("Searching for k nearest neighbors for each sphere center-------------")
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

        print("Calculating points assigned to each sphere-------------")
        # Prepare to collect sphere data and statistics
        sphere_data = []
        total_covered_points = set()
        
        # Perform a range search for each valid sphere to find points within its radius
        for center, radius in tqdm(zip(valid_centers, valid_radii)):
            # Convert center to a query matrix of shape (1, d) for FAISS
            query_matrix = center.reshape(1, d).astype('float32')
            
            # Perform the range search
            lims, D, I = index.range_search(query_matrix, radius)
            
            # Extract indices of points within the radius
            point_indices = I[lims[0]:lims[1]]

            # calculate score distribution
            score_distribution=np.zeros(len(bins))
            for idx in point_indices:
                score= scores[idx]
                for i in range(len(bins)):
                    if score < bins[i]:
                        score_distribution[i]+=1
                        break
            
            score_distribution= score_distribution / len(point_indices)
            
            # Update sphere data and covered points
            sphere_data.append({'center': center, 'radius': math.sqrt(radius), 'points': point_indices, 
                                "score_distribution": score_distribution})
            total_covered_points.update(point_indices)
        
        # Calculate statistics
        points_per_sphere = [len(sphere['points']) for sphere in sphere_data]
        avg_points_per_sphere = np.mean(points_per_sphere) if points_per_sphere else 0

        print(f"total datapoints: {len(total_covered_points)}")
        print(f"average points per sphere: {avg_points_per_sphere}")
        
        return sphere_data, avg_points_per_sphere, len(total_covered_points)


    def load_sphere_dataset(self, n_spheres, target_avg_points, num_bins, bin_size):
        # generating spheres
        sphere_data, avg_points_per_sphere, total_covered_points= self.generate_spheres(n_spheres=n_spheres,
                                                       target_avg_points=target_avg_points,
                                                       num_bins=num_bins,
                                                       bin_size=bin_size)
        
        inputs=[]
        outputs=[]
        for sphere in sphere_data:
            # get input vectors
            inputs.append(np.concatenate([sphere['center'], [sphere['radius']]]))
            # get score distribution
            outputs.append(sphere['score_distribution'])

        return inputs, outputs 

    def plot(self, sphere_data, points_per_sphere, n_spheres, scores):
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        
        # Preparing data for plots
        mean_scores = [np.mean([scores[j] for j in sphere_data[i]['points']]) if sphere_data[i] else 0 for i in range(n_spheres)]
        
        # Histogram of Points per Cluster
        axs[0].bar(range(n_spheres), points_per_sphere, color='skyblue')
        axs[0].set_xlabel('Sphere ID')
        axs[0].set_ylabel('Number of Points')
        axs[0].set_title('Number of Points per Sphere')
        
        # Scatter Plot of Cluster Density vs. Mean Score
        axs[1].scatter(points_per_sphere, mean_scores, c='blue', marker='o')
        axs[1].set_xlabel('Sphere Density (Number of Points)')
        axs[1].set_ylabel('Mean Score')
        axs[1].set_title('Sphere Density vs. Mean Score')
        axs[1].grid(True)
        
        plt.tight_layout()

        # Save the figure to a file
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # upload the graph report
        cmd.upload_data(self.minio_client, 'datasets', "environmental/output/sphere_dataset/graphs.png", buf)  

        # Clear the current figure
        plt.clf()

def main():
    args= parse_args()

    # get minio client
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key)

    generator= UniformSphereGenerator(minio_client=minio_client,
                                    dataset=args.dataset)
    
    inputs, outputs = generator.load_sphere_dataset(n_spheres=args.n_spheres,
                                                       target_avg_points= args.target_avg_points)
    
    
if __name__ == "__main__":
    main()


