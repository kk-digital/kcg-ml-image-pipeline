import argparse
from datetime import datetime
import io
import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import msgpack
from PIL import Image


base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from kandinsky_worker.image_generation.img2img_generator import generate_img2img_generation_jobs_with_kandinsky
from training_worker.scoring.models.scoring_fc import ScoringFCNetwork
from utility.minio import cmd
from data_loader.utils import get_object

def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--minio-access-key', type=str, help='Minio access key')
        parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
        parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")
        parser.add_argument('--send-job', action='store_true', default=False)
        parser.add_argument('--save-csv', action='store_true', default=False)

        return parser.parse_args()

class KandinskyImageGenerator:
    def __init__(self,
                 minio_access_key,
                 minio_secret_key,
                 dataset,
                 send_job=False,
                 save_csv=False
                 ):
        
        self.dataset= dataset
        self.send_job= send_job
        self.save_csv= save_csv

        # get minio client
        self.minio_client = cmd.get_minio_client(minio_access_key=minio_access_key,
                                                minio_secret_key=minio_secret_key)
        
        # get device
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = torch.device(device)
        

        self.scoring_model= ScoringFCNetwork(minio_client=self.minio_client)
        self.scoring_model.load_model()

        self.clip_mean , self.clip_std, self.clip_max, self.clip_min= self.get_clip_distribution()

    def get_clip_distribution(self):
        data = get_object(self.minio_client, f"{self.dataset}/output/stats/clip_stats.msgpack")
        data_dict = msgpack.unpackb(data)

        mean_vector = torch.tensor(data_dict["mean"]).to(device=self.device, dtype=torch.float32)
        std_vector = torch.tensor(data_dict["std"]).to(device=self.device, dtype=torch.float32)
        max_vector = torch.tensor(data_dict["max"]).to(device=self.device, dtype=torch.float32)
        min_vector = torch.tensor(data_dict["min"]).to(device=self.device, dtype=torch.float32)

        return mean_vector, std_vector, max_vector, min_vector

    def sample_embeddings(self, num_samples=1000):
        sampled_embeddings = torch.normal(mean=self.clip_mean.repeat(num_samples, 1),
                                        std=self.clip_std.repeat(num_samples, 1))
        
        # Score each sampled embedding
        scores=[]
        embeddings=[]
        for embed in sampled_embeddings:
            embeddings.append(embed.unsqueeze(0))
            score = self.scoring_model.model(embed.unsqueeze(0)).item() 
            scores.append(score)
        
        return scores
    
    def scores_histogram(self):

        scores= self.sample_embeddings(num_samples=100000)

        # Create a histogram
        plt.figure(figsize=(10, 5))
        plt.hist(scores, bins=30, alpha=0.7, color='b', label='Data')

        plt.xlabel('Sigma Score')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of scores')

        # Save the figure to a file
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # upload the graph report
        minio_path= f"{self.dataset}/data/latent-generator/score_distribution.png"
        cmd.upload_data(self.minio_client, 'datasets', minio_path, buf)

        # Clear the current figure
        plt.clf()
    
def main():
    args= parse_args()
    # initialize generator
    generator= KandinskyImageGenerator(minio_access_key=args.minio_access_key,
                                       minio_secret_key=args.minio_secret_key,
                                       dataset=args.dataset,
                                       send_job= args.send_job,
                                       save_csv= args.save_csv)
    
    generator.scores_histogram()