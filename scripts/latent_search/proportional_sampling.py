import argparse
import os
import sys
import numpy as np
import torch
import msgpack


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
        parser.add_argument('--num-bins', type=int, help='Number of bins', default=10)
        parser.add_argument('--num-images', type=int, help='Number of images to generate', default=1000)
        parser.add_argument('--top-k', type=float, help='Portion of samples to generate images with', default=0.1)
        parser.add_argument('--batch-size', type=float, help='Inference batch size used by the scoring model', default=64)
        parser.add_argument('--send-job', action='store_true', default=False)
        parser.add_argument('--save-csv', action='store_true', default=False)

        return parser.parse_args()

class KandinskyImageGenerator:
    def __init__(self,
                 minio_access_key,
                 minio_secret_key,
                 dataset,
                 num_bins,
                 top_k,
                 batch_size,
                 send_job=False,
                 save_csv=False
                ):
        
        self.dataset= dataset
        self.send_job= send_job
        self.save_csv= save_csv
        self.num_bins= num_bins
        self.top_k= top_k
        self.batch_size= batch_size

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
    
    def proportional_sampling(self, num_samples):
        num_bins = self.num_bins
        embeddings_per_bin = num_samples // self.num_bins

        # Generate a large batch of embeddings
        embeddings, scores = self.sample_embeddings(int(num_samples / self.top_k))

        # Determine min and max scores
        min_score, max_score = min(scores), max(scores)

        # Define bins based on the range of scores
        bins = np.linspace(min_score, max_score, num_bins + 1)

        # Initialize bins
        binned_vectors = {i: [] for i in range(num_bins)}
        bins_full = 0

        for embedding, score in zip(embeddings, scores):
            if bins_full == num_bins:
                break
            
            # Determine the appropriate bin for the current score
            bin_index = np.digitize(score, bins) - 1
            # Adjust bin index to fit within the range of defined bins
            bin_index = max(0, min(bin_index, num_bins - 1))
            
            # Add embedding to the bin if not full
            if len(binned_vectors[bin_index]) < embeddings_per_bin:
                binned_vectors[bin_index].append(embedding)
                
                # Check if the bin is now full
                if len(binned_vectors[bin_index]) == embeddings_per_bin:
                    bins_full += 1

        # At this point, all necessary bins are filled or the max number of generations is reached
        # Process the binned embeddings as needed for your application

        print("Binning complete. Summary:")
        for bin_index, embeddings in binned_vectors.items():
            print(f"Bin {bin_index}: {len(embeddings)} embeddings")

    def sample_embeddings(self, num_samples):
        sampled_embeddings = torch.normal(mean=self.clip_mean.repeat(num_samples, 1),
                                        std=self.clip_std.repeat(num_samples, 1))

        # Score each sampled embedding
        scores=[]
        embeddings=[]
        # Perform prediction in batches
        with torch.no_grad():
            for i in range(0, len(sampled_embeddings), self.batch_size):
                batch = sampled_embeddings[i:i + self.batch_size]  # Extract a batch
                embeddings.extend(batch)
                outputs = self.scoring_model.model(batch)  # Get predictions for this batch
                # Concatenate all scores and convert to a list
                outputs = torch.cat(outputs, dim=0).cpu().numpy().list()
                scores.extend(outputs)
        
        return embeddings, scores
    
def main():
    args= parse_args()
    # initialize generator
    generator= KandinskyImageGenerator(minio_access_key=args.minio_access_key,
                                       minio_secret_key=args.minio_secret_key,
                                       dataset=args.dataset,
                                       num_bins=args.num_bins,
                                       top_k= args.top_k,
                                       batch_size= args.batch_size,
                                       send_job= args.send_job,
                                       save_csv= args.save_csv)
    
    generator.proportional_sampling(num_samples=args.num_images)

if __name__ == "__main__":
    main()