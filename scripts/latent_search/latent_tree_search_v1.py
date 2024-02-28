import io
import os
import sys
import torch
import argparse

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from training_worker.ab_ranking.model.ab_ranking_linear import ABRankingModel
from utility.minio import cmd

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")
    parser.add_argument('--model-type', type=str, help='model type, linear or elm', default="linear")
    parser.add_argument('--steps', type=int, help='number of optimisation steps', default=100)
    parser.add_argument('--noise-scale', type=float, help='learning rate for optimization', default=0.1)
    parser.add_argument('--target-score', type=float, help='number of optimisation steps', default=5)
    parser.add_argument('--send-job', action='store_true', default=False)
    parser.add_argument('--save-csv', action='store_true', default=False)
    parser.add_argument('--generate-step', type=int, default=100)
    parser.add_argument('--print-step', type=int, default=10)

class KandinskyTreeSearchGenerator:
    def __init__(self,
                 minio_access_key,
                 minio_secret_key,
                 dataset,
                 model_type,
                 steps=100,
                 noise_scale=0.1,
                 target_score=5.0,
                 send_job=False,
                 save_csv=False,
                 generate_step=100,
                 print_step=10
                 ):
        
        self.dataset= dataset
        self.model_type= model_type
        self.steps= steps
        self.noise_scale= noise_scale
        self.target_score= target_score
        self.send_job= send_job
        self.save_csv= save_csv
        self.generate_step= generate_step
        self.print_step= print_step

        # get minio client
        self.minio_client = cmd.get_minio_client(minio_access_key=minio_access_key,
                                                minio_secret_key=minio_secret_key)
        
        # get device
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = torch.device(device)

        # load scoring model
        self.scoring_model= self.load_scoring_model()
        self.mean= float(self.scoring_model.mean)
        self.std= float(self.scoring_model.standard_deviation)

        self.clip_mean , self.clip_std, self.clip_max, self.clip_min= self.get_clip_distribution()
        print(self.clip_mean, self.clip_std)


    # load elm or linear scoring models
    def load_scoring_model(self):
        input_path=f"{self.dataset}/models/ranking/"

        if(self.model_type=="elm"):
            scoring_model = ABRankingELMModel(1280)
            file_name=f"score-elm-v1-kandinsky-clip.safetensors"
        else:
            scoring_model= ABRankingModel(1280)
            file_name=f"score-linear-kandinsky-clip.safetensors"

        model_files=cmd.get_list_of_objects_with_prefix(self.minio_client, 'datasets', input_path)
        most_recent_model = None

        for model_file in model_files:
            if model_file.endswith(file_name):
                most_recent_model = model_file

        if most_recent_model:
            model_file_data =cmd.get_file_from_minio(self.minio_client, 'datasets', most_recent_model)
        else:
            print("No .safetensors files found in the list.")
            return

        print(most_recent_model)

        # Create a BytesIO object and write the downloaded content into it
        byte_buffer = io.BytesIO()
        for data in model_file_data.stream(amt=8192):
            byte_buffer.write(data)
        # Reset the buffer's position to the beginning
        byte_buffer.seek(0)

        scoring_model.load_safetensors(byte_buffer)
        scoring_model.model=scoring_model.model.to(torch.device(self.device))

        return scoring_model

    def sample_embedding(self, num_samples=1):
        # Sample from a normal distribution using the mean and standard deviation vectors
        sampled_embeddings = torch.normal(self.clip_mean, self.clip_std)
        
        # Clip the sampled embeddings based on the min and max vectors to ensure they stay within observed bounds
        clipped_embeddings = torch.max(torch.min(sampled_embeddings, self.clip_max), self.clip_min)
        
        return clipped_embeddings.to(
            device=self.device, dtype=torch.float32
        )    

    def generate_latent(self):
        """
        Optimize the embedding using a tree search-like algorithm with Gaussian noise variations.
        
        Returns:
        - torch.Tensor: The optimized embedding.
        """
        sampled_embedding= self.sample_embedding()
        current_embedding = sampled_embedding.clone().detach()
        current_embedding.requires_grad_(False)  # Ensure no gradients are tracked

        for iteration in range(self.steps):
            # Generate variations by adding Gaussian noise
            noise = torch.randn((1000, *current_embedding.shape), device=current_embedding.device) * self.noise_scale
            variations = current_embedding + noise

            # Evaluate all variations and select the best
            scores = torch.tensor([self.scoring_model.predict_clip(variation).item() for variation in variations])
            best_variation_idx = scores.argmax()
            best_variation = variations[best_variation_idx]

            # Update current embedding to the best variation
            current_embedding = best_variation.clone().detach()

            print(f"Iteration {iteration}, Best Score: {scores[best_variation_idx].item()}")

        return current_embedding
    

def main():
    args= parse_args()
    # initialize generator
    generator= KandinskyTreeSearchGenerator(minio_access_key=args.minio_access_key,
                                       minio_secret_key=args.minio_secret_key,
                                       dataset=args.dataset,
                                       model_type=args.model_type,
                                       steps=args.steps,
                                       noise_scale= args.noise_scale,
                                       target_score=args.target_score,
                                       send_job= args.send_job,
                                       save_csv= args.save_csv,
                                       generate_step=args.generate_step,
                                       print_step=args.print_step)
    
    generator.generate_latent()

if __name__=="__main__":
    main()
