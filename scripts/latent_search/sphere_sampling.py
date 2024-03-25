import argparse
import os
import sys
import msgpack
import numpy as np
import torch

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())
from data_loader.utils import get_object
from training_worker.sampling.models.uniform_sampling_regression_fc import SamplingFCRegressionNetwork
from kandinsky_worker.image_generation.img2img_generator import generate_img2img_generation_jobs_with_kandinsky
from utility.minio import cmd

def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--minio-access-key', type=str, help='Minio access key')
        parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
        parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")
        parser.add_argument('--num-images', type=int, help='Number of images to generate', default=1000)
        parser.add_argument('--top-k', type=float, help='Portion of spheres to select from', default=0.1)
        parser.add_argument('--total-spheres', type=float, help='Number of random spheres to rank', default=500000)
        parser.add_argument('--selected-spheres', type=float, help='Number of spheres to sample from', default=10)
        parser.add_argument('--batch-size', type=int, help='Inference batch size used by the scoring model', default=256)
        parser.add_argument('--send-job', action='store_true', default=False)
        parser.add_argument('--save-csv', action='store_true', default=False)
        parser.add_argument('--sampling-policy', type=str, default="top-k-sphere-sampling")

        return parser.parse_args()

class SphereSamplingGenerator:
    def __init__(self,
                minio_access_key,
                minio_secret_key,
                dataset,
                top_k,
                total_spheres,
                selected_spheres,
                batch_size,
                sampling_policy,
                send_job=False,
                save_csv=False
                ):
            
            self.dataset= dataset
            self.send_job= send_job
            self.save_csv= save_csv
            self.top_k= top_k
            self.total_spheres= total_spheres
            self.selected_spheres= selected_spheres
            self.selected_spheres= selected_spheres
            self.batch_size= batch_size
            self.sampling_policy= sampling_policy

            # get minio client
            self.minio_client = cmd.get_minio_client(minio_access_key=minio_access_key,
                                                    minio_secret_key=minio_secret_key)
            
            # get device
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
            self.device = torch.device(device)
            

            self.scoring_model= SamplingFCRegressionNetwork(minio_client=self.minio_client, dataset=dataset)
            self.scoring_model.load_model()
            # get min and max radius values
            self.min_radius= self.scoring_model.min_radius.item()
            self.max_radius= self.scoring_model.max_radius.item()

            self.clip_mean , self.clip_std, self.clip_max, self.clip_min= self.get_clip_distribution()
    
    def get_clip_distribution(self):
        data = get_object(self.minio_client, f"{self.dataset}/output/stats/clip_stats.msgpack")
        data_dict = msgpack.unpackb(data)

        mean_vector = np.array(data_dict["mean"])
        std_vector = np.array(data_dict["std"])
        max_vector = np.array(data_dict["max"])
        min_vector = np.array(data_dict["min"])

        return mean_vector, std_vector, max_vector, min_vector

    def generate_spheres(self):
        num_spheres= self.total_spheres

        # Generate random values between 0 and 1, then scale and shift them into the [min, max] range for each feature
        sphere_centers = np.random.rand(num_spheres, len(self.clip_max)) * (self.clip_max - self.clip_min) + self.clip_min

        # choose random radius for the spheres
        radii= np.random.rand(num_spheres) * (self.max_radius - self.min_radius) + self.min_radius

        spheres=[]
        for radius, sphere_center in zip(radii, sphere_centers):
             spheres.append({
                "sphere_center": sphere_center,
                "radius": radius
             })

        return spheres
    
    def rank_and_select_spheres(self):
        # generate initial spheres
        generated_spheres= self.generate_spheres() 

        batch=[]
        scores=[]
        for sphere in generated_spheres:
            sphere_vector= np.concatenate([sphere['sphere_center'], [sphere['radius']]])
            batch.append(sphere_vector)

            if len(batch)==self.batch_size:
                batch_scores= self.scoring_model.predict(batch, batch_size= self.batch_size).tolist()
                scores.extend(batch_scores)
                batch=[]
          
        sorted_indexes= np.flip(np.argsort(scores))[:self.selected_spheres]
        top_spheres=[generated_spheres[i] for i in sorted_indexes]

        return top_spheres
    
    def sample_clip_vectors(self, num_samples):
        # get spheres
        spheres= self.rank_and_select_spheres()
        dim = len(spheres[0]['sphere_center'])
        points_per_sphere = num_samples // self.selected_spheres  # Ensure equal distribution of points

        clip_vectors=[]
        for i, sphere in enumerate(spheres):
            center= sphere['sphere_center']
            radius= sphere['radius']
            for j in range(points_per_sphere):
                # Generate a random direction vector
                direction = np.random.randn(dim)
                direction /= np.linalg.norm(direction)  # Normalize to unit vector
                
                # Randomly choose a magnitude within the radius
                magnitude = np.random.rand()**0.5 * radius  # Square root for uniform sampling in volume
                
                # Compute the point
                point = center + direction * magnitude
                point = torch.tensor(point)

                # Store in the all_points array
                clip_vectors.append(point)

        return clip_vectors
    
    def generate_images(self, num_images):
        # generate clip vectors
        clip_vectors= self.sample_clip_vectors(num_samples=num_images)

        for clip_vector in clip_vectors:
            if self.send_job:
                try:
                    response= generate_img2img_generation_jobs_with_kandinsky(
                        image_embedding=clip_vector,
                        negative_image_embedding=None,
                        dataset_name="test-generations",
                        prompt_generation_policy=self.sampling_policy,
                        self_training=True
                    )

                    task_uuid = response['uuid']
                    task_time = response['creation_time']
                except:
                    print("An error occured.")
                    task_uuid = -1
                    task_time = -1
        
        print("Jobs were sent for generation.")

def main():
    args= parse_args()

    # initialize generator
    generator= SphereSamplingGenerator(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        dataset=args.dataset,
                                        top_k= args.top_k,
                                        total_spheres= args.total_spheres,
                                        selected_spheres= args.selected_spheres,
                                        batch_size= args.batch_size,
                                        sampling_policy= args.sampling_policy,
                                        send_job= args.send_job,
                                        save_csv= args.save_csv)

    generator.generate_images(num_images=args.num_images)

if __name__ == "__main__":
    main()