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
from utility.minio import cmd

def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--minio-access-key', type=str, help='Minio access key')
        parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
        parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")
        parser.add_argument('--num-images', type=int, help='Number of images to generate', default=1000)
        parser.add_argument('--top-k', type=float, help='Portion of spheres to select from', default=0.1)
        parser.add_argument('--total-spheres', type=float, help='Number of random spheres to rank', default=10000)
        parser.add_argument('--selected-spheres', type=float, help='Number of spheres to sample from', default=10)
        parser.add_argument('--batch-size', type=int, help='Inference batch size used by the scoring model', default=256)
        parser.add_argument('--send-job', action='store_true', default=False)
        parser.add_argument('--save-csv', action='store_true', default=False)
        parser.add_argument('--sampling-policy', type=str, default="top-k")

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
            self.min_radius= self.scoring_model.min_radius
            self.max_radius= self.scoring_model.max_radius

            self.clip_mean , self.clip_std, self.clip_max, self.clip_min= self.get_clip_distribution()
    
    def get_clip_distribution(self):
        data = get_object(self.minio_client, f"{self.dataset}/output/stats/clip_stats.msgpack")
        data_dict = msgpack.unpackb(data)

        mean_vector = torch.tensor(data_dict["mean"]).to(device=self.device, dtype=torch.float32)
        std_vector = torch.tensor(data_dict["std"]).to(device=self.device, dtype=torch.float32)
        max_vector = torch.tensor(data_dict["max"]).to(device=self.device, dtype=torch.float32)
        min_vector = torch.tensor(data_dict["min"]).to(device=self.device, dtype=torch.float32)

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
            sphere_vector= np.concatenate([sphere['sphere_center']], [sphere['radius']])
            batch.append(sphere_vector)

            if len(batch)==self.batch_size:
                batch_scores= self.scoring_model.predict(batch, batch_size= self.batch_size).tolist()
                scores.extend(batch_scores)
                batch=[]

        sorted_indexes= np.flip(np.argsort(scores))[:self.selected_spheres]
        top_spheres=[generated_spheres[i] for i in sorted_indexes]

        return top_spheres