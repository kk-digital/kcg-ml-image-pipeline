import argparse
from io import BytesIO
import math
import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import faiss
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from data_loader.kandinsky_dataset_loader import KandinskyDatasetLoader
from utility.minio import cmd

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--minio-addr', type=str, help='Minio address')
    parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")
    parser.add_argument('--target-avg-points', type=int, help='Target average of datapoints per sphere', 
                        default=5)
    parser.add_argument('--percentile', type=int, help='Percentile', default=75)
    parser.add_argument('--std', type=float, help='Standard deviation', default=1)
    parser.add_argument('--n-spheres', type=int, help='Number of spheres', default=100000)
    parser.add_argument('--num-bins', type=int, help='Number of score bins', default=8)
    parser.add_argument('--bin-size', type=int, help='Range of each bin', default=1)

    return parser.parse_args()

class UniformSphereGenerator:
    def __init__(self,
                 minio_client,
                 dataset):
        
        self.dataloader= KandinskyDatasetLoader(minio_client=minio_client,
                                                dataset=dataset)
        
        self.minio_client= minio_client
        self.dataset= dataset

    def generate_spheres(self, n_spheres, target_avg_points, num_bins, bin_size, percentile, std, discard_threshold=None):
        
        bins=[]
        for i in range(num_bins-1):
            max_score= int((i+1-(num_bins/2)) * bin_size)
            bins.append(max_score)
        
        bins.append(np.inf)

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
            valid_distances = distances[valid_mask]
            indices = indices[valid_mask]
        else:
            valid_distances = distances
            valid_centers = sphere_centers

        print("Processing sphere data-------------")
        # Prepare to collect sphere data and statistics
        sphere_data = []
        total_covered_points = set()

        # Assuming 'scores' contains the scores for all points and 'bins' defines the score bins
        for center, distance_vector, sphere_indices in zip(valid_centers, valid_distances, indices):
            # Extract indices of points within the sphere
            point_indices = sphere_indices
            
            d = np.percentile(distance_vector, percentile)
            variance = (d / std) ** 2

            # Calculate score distribution for the sphere
            score_distribution = np.zeros(len(bins))
            sum_weights = .0

            for idx, i in enumerate(point_indices):
                score = scores[idx]
                
                weight = gaussian_pdf(distance_vector[i])
                sum_weights += weight

                for i, bin_edge in enumerate(bins):
                    if score < bin_edge:
                        score_distribution[i] += weight
                        break

            # Normalize the score distribution by the number of points in the sphere
            if len(point_indices) > 0:
                score_distribution = score_distribution / sum_weights
            
            # Update sphere data and covered points
            sphere_data.append({
                'center': center, 
                'variance': variance,  # Assuming radius needs to be sqrt to represent actual distance
                'points': point_indices, 
                "score_distribution": score_distribution
            })
            total_covered_points.update(point_indices)
        
        # Calculate statistics
        points_per_sphere = [len(sphere['points']) for sphere in sphere_data]
        avg_points_per_sphere = np.mean(points_per_sphere) if points_per_sphere else 0

        print(f"total datapoints: {len(total_covered_points)}")
        print(f"average points per sphere: {avg_points_per_sphere}")
        
        self.plot(sphere_data, points_per_sphere, n_spheres, scores, percentile, std)

        return sphere_data, avg_points_per_sphere, len(total_covered_points)


    def load_sphere_dataset(self, n_spheres, target_avg_points, num_bins, bin_size, percentile, std):
        # generating spheres
        sphere_data, avg_points_per_sphere, total_covered_points= self.generate_spheres(n_spheres=n_spheres,
                                                       target_avg_points=target_avg_points,
                                                       num_bins=num_bins,
                                                       bin_size=bin_size,
                                                       percentile=percentile,
                                                       std=std)
        
        inputs=[]
        outputs=[]
        for sphere in sphere_data:
            # get input vectors
            inputs.append(np.concatenate([sphere['center'], [sphere['radius']]]))
            # get score distribution
            outputs.append(sphere['score_distribution'])

        return inputs, outputs 

    def plot(self, sphere_data, points_per_sphere, n_spheres, scores, percentile, std):
        fig, axs = plt.subplots(1, 3, figsize=(24, 8))  # Adjust for three subplots
        
        # Calculate mean scores as before
        mean_scores = [np.mean([scores[j] for j in sphere_data[i]['points']]) if sphere_data[i] else 0 for i in range(n_spheres)]
        sphere_radii= [data['radius'] for data in sphere_data]
        
        # Histogram of Points per Sphere
        axs[0].hist(points_per_sphere, color='skyblue', bins=np.arange(min(points_per_sphere), max(points_per_sphere) + 1, 1))
        axs[0].set_xlabel('Number of Points')
        axs[0].set_ylabel('Frequency')
        axs[0].set_title('Distribution of Points per Sphere')
        
        # Histogram of Mean Scores
        axs[1].hist(mean_scores, color='lightgreen', bins=20)  # Adjust bins as needed
        axs[1].set_xlabel('Mean Score')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title('Distribution of Mean Scores')

        # Histogram of Sphere Radii
        axs[2].hist(sphere_radii, color='lightcoral', bins=20)  # Adjust bins as needed
        axs[2].set_xlabel('Sphere Radii')
        axs[2].set_ylabel('Frequency')
        axs[2].set_title('Distribution of Sphere Radii')
        
        plt.tight_layout()

        # Save the figure to a file
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Upload the graph report
        # Ensure cmd.upload_data(...) is appropriately defined to handle your MinIO upload.
        cmd.upload_data(self.minio_client, 'datasets', f"environmental/output/sphere_dataset/graphs_.png", buf)  

        # Clear the current figure to prevent overlap with future plots
        plt.clf()

def gaussian_pdf(distance, mean, variance):
    denom = (2*np.pi*variance)**.5
    num = np.exp(-float(distance)**2/(2*variance))
    return num/denom

def main():
    args= parse_args()

    # get minio client
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)

    generator= UniformSphereGenerator(minio_client=minio_client,
                                    dataset=args.dataset)
    
    inputs, outputs = generator.load_sphere_dataset(num_bins= args.num_bins,
                                                    bin_size= args.bin_size,
                                                    n_spheres=args.n_spheres,
                                                    target_avg_points= args.target_avg_points,
                                                    percentile=args.percentile,
                                                    std = args.std)
    
    
if __name__ == "__main__":
    main()


