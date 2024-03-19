import argparse
from datetime import datetime
import io
import json
import os
import random
import sys
import numpy as np
import pandas as pd
import requests
import torch
import msgpack
import torch.optim as optim

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from kandinsky_worker.image_generation.img2img_generator import generate_img2img_generation_jobs_with_kandinsky
from training_worker.scoring.models.scoring_fc import ScoringFCNetwork
from utility.minio import cmd
from utility.http import request
from data_loader.utils import get_object
from utility.path import separate_bucket_and_file_path

def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--minio-access-key', type=str, help='Minio access key')
        parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
        parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")
        parser.add_argument('--num-bins', type=int, help='Number of bins', default=10)
        parser.add_argument('--num-images', type=int, help='Number of images to generate', default=1000)
        parser.add_argument('--top-k', type=float, help='Portion of samples to generate images with', default=0.1)
        parser.add_argument('--batch-size', type=int, help='Inference batch size used by the scoring model', default=256)
        parser.add_argument('--learning-rate', type=float, help='Learning rate of optimization', default=0.001)
        parser.add_argument('--steps', type=int, help='Number of steps for optimization', default=200)
        parser.add_argument('--send-job', action='store_true', default=False)
        parser.add_argument('--save-csv', action='store_true', default=False)
        parser.add_argument('--generation-policy', type=str, help='generation policy', default="proportional_sampling")
        parser.add_argument('--sampling-policy', type=str, help='sampling policy random, or existing', default="existing")

        return parser.parse_args()

API_URL="http://192.168.3.1:8111"

class KandinskyImageGenerator:
    def __init__(self,
                 minio_access_key,
                 minio_secret_key,
                 dataset,
                 num_bins,
                 top_k,
                 batch_size,
                 steps,
                 learning_rate,
                 generation_policy,
                 sampling_policy,
                 send_job=False,
                 save_csv=False
                ):
        
        self.dataset= dataset
        self.send_job= send_job
        self.save_csv= save_csv
        self.num_bins= num_bins
        self.top_k= top_k
        self.batch_size= batch_size
        self.steps= steps
        self.learning_rate= learning_rate
        self.generation_policy= generation_policy
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
        

        self.scoring_model= ScoringFCNetwork(minio_client=self.minio_client, dataset=dataset)
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
        num_generated_samples= max(int(num_samples / self.top_k), 1000)
        embeddings, scores = self.sample_embeddings(num_generated_samples)

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
                binned_vectors[bin_index].append(embedding.unsqueeze(0))
                
                # Check if the bin is now full
                if len(binned_vectors[bin_index]) == embeddings_per_bin:
                    bins_full += 1

        # At this point, all necessary bins are filled or the max number of generations is reached
        # Process the binned embeddings as needed for your application

        final_list=[]
        print("Binning complete. Summary:")
        for bin_index, embeddings in binned_vectors.items():
            print(f"Bin {bin_index}: {len(embeddings)} embeddings")
            final_list.extend(embeddings)
        
        random.shuffle(final_list)

        return final_list
    
    def get_file_paths(self, num_samples):
        print('Loading image file paths')
        response = requests.get(f'{API_URL}/queue/image-generation/list-by-dataset?dataset={self.dataset}&size={num_samples}&min_clip_sigma_score=1')

        jobs = json.loads(response.content)

        file_paths=[job['file_path'] for job in jobs]

        return file_paths

    # From multiples image paths
    def get_clip_vectors(self, file_paths):
        clip_vectors = []

        for path in file_paths:
            try:
                print("path : " , path)
                clip_path = path.replace(".jpg", "_clip_kandinsky.msgpack")
                bucket, features_vector_path = separate_bucket_and_file_path(clip_path)

                features_data = get_object(self.minio_client, features_vector_path)
                features = msgpack.unpackb(features_data)["clip-feature-vector"]
                features = torch.tensor(features).to(self.device)
                clip_vectors.append(features)
            except Exception as e:
                # Handle the specific exception (e.g., FileNotFoundError, ConnectionError) or a general exception.
                print(f"Error processing clip at path {path}: {e}")
                # You might want to log the error for further analysis or take alternative actions.

        return clip_vectors

    def sampling_from_dataset(self, num_samples):
        # get random file paths for samples 
        file_paths= self.get_file_paths(num_samples)
        # get their clip vectors
        clip_vectors= self.get_clip_vectors(file_paths)

        return clip_vectors
    
    def top_k_sampling(self, num_samples):
        # Generate a large batch of embeddings
        num_generated_samples= max(int(num_samples / self.top_k), 1000)
        embeddings, scores = self.sample_embeddings(num_generated_samples)

        indexes= np.argsort(scores)[:num_samples]
        samples= [embeddings[index].unsqueeze(0) for index in indexes]

        return samples
    
    def gradient_descent_optimization(self, num_samples):
        if(self.sampling_policy=="random"):
            clip_vectors = self.top_k_sampling(num_samples=num_samples)
        elif(self.sampling_policy=="existing"):
            clip_vectors = self.sampling_from_dataset(num_samples=num_samples)

        # Convert list of embeddings to a tensor
        all_embeddings = torch.stack(clip_vectors).detach()

        # Calculate the total number of batches
        num_batches = len(all_embeddings) // self.batch_size + (0 if len(all_embeddings) % self.batch_size == 0 else 1)
        
        optimized_embeddings_list = []

        for batch_idx in range(num_batches):
            # Select a batch of embeddings
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(all_embeddings))
            batch_embeddings = all_embeddings[start_idx:end_idx].clone().detach().requires_grad_(True)
            
            # Setup the optimizer for the current batch
            optimizer = optim.Adam([batch_embeddings], lr=self.learning_rate)
            
            for step in range(self.steps):
                optimizer.zero_grad()

                # Compute scores for the current batch of embeddings
                scores = self.scoring_model.model(batch_embeddings)

                # Calculate the loss for each embedding in the batch
                score_losses = -scores.squeeze()

                # Calculate the total loss for the batch
                total_loss = score_losses.mean()

                # Backpropagate
                total_loss.backward()

                optimizer.step()

                print(f"Batch: {batch_idx + 1}/{num_batches}, Step: {step}, Mean Score: {scores.mean().item()}, Loss: {total_loss.item()}")

            # After optimization, detach and add the optimized batch embeddings to the list
            optimized_batch_embeddings = batch_embeddings.detach()
            optimized_embeddings_list.extend([emb for emb in optimized_batch_embeddings])

        return optimized_embeddings_list

    def sample_embeddings(self, num_samples):
        if(self.sampling_policy == "random"):
            sampled_embeddings = torch.normal(mean=self.clip_mean.repeat(num_samples, 1),
                                            std=self.clip_std.repeat(num_samples, 1))
        elif(self.sampling_policy == "existing"):
            sampled_embeddings = self.sampling_from_dataset(num_samples=num_samples)
            sampled_embeddings = torch.stack(sampled_embeddings).squeeze(1)

        # Score each sampled embedding
        scores=[]
        embeddings=[]
        # Perform prediction in batches
        with torch.no_grad():
            for i in range(0, len(sampled_embeddings), self.batch_size):
                batch = sampled_embeddings[i:i + self.batch_size]  # Extract a batch
                embeddings.extend(batch)
                outputs = self.scoring_model.model(batch).squeeze(1)  # Get predictions for this batch
                # Concatenate all scores and convert to a list
                scores.extend(outputs.tolist())
        
        return embeddings, scores
    
    def generate_images(self, num_images):

        if(self.generation_policy=="proportional_sampling"):
            clip_vectors= self.proportional_sampling(num_samples=num_images)
        elif(self.generation_policy=="top_k"):
            clip_vectors= self.top_k_sampling(num_samples=num_images)
        elif(self.generation_policy=="gradient_descent_optimization"):
            clip_vectors= self.gradient_descent_optimization(num_samples=num_images)

        df_data=[]
        for clip_vector in clip_vectors:
            if self.send_job:
                try:
                    response= generate_img2img_generation_jobs_with_kandinsky(
                        image_embedding=clip_vector,
                        negative_image_embedding=None,
                        dataset_name=self.dataset,
                        prompt_generation_policy=self.generation_policy,
                        self_training=True
                    )

                    task_uuid = response['uuid']
                    task_time = response['creation_time']
                except:
                    print("An error occured.")
                    task_uuid = -1
                    task_time = -1
            
            if self.save_csv:
                df_data.append({
                    'task_uuid': task_uuid,
                    'generation_policy_string': self.generation_policy,
                    'time': task_time
                })

        if self.save_csv:
            self.store_uuids_in_csv_file(df_data)
    
    # store list of initial prompts in a csv to use for prompt mutation
    def store_uuids_in_csv_file(self, data):
        minio_path=f"{self.dataset}/output/generated-images-csv"
        local_path="output/generated_images.csv"
        pd.DataFrame(data).to_csv(local_path, index=False)
        # Read the contents of the CSV file
        with open(local_path, 'rb') as file:
            csv_content = file.read()

        #Upload the CSV file to Minio
        buffer = io.BytesIO(csv_content)
        buffer.seek(0)

        current_date=datetime.now().strftime("%Y-%m-%d-%H:%M")
        minio_path= minio_path + f"/{current_date}-{self.generation_policy}-{self.dataset}.csv"
        cmd.upload_data(self.minio_client, 'datasets', minio_path, buffer)
        # Remove the temporary file
        os.remove(local_path)

def main():
    args= parse_args()

    if args.dataset != "all":
        # initialize generator
        generator= KandinskyImageGenerator(minio_access_key=args.minio_access_key,
                                            minio_secret_key=args.minio_secret_key,
                                            dataset=args.dataset,
                                            num_bins=args.num_bins,
                                            top_k= args.top_k,
                                            batch_size= args.batch_size,
                                            learning_rate= args.learning_rate,
                                            steps= args.steps,
                                            generation_policy= args.generation_policy,
                                            sampling_policy= args.sampling_policy,
                                            send_job= args.send_job,
                                            save_csv= args.save_csv)

        generator.generate_images(num_images=args.num_images)
    else:
        # if all, generate num_images for each dataset
        # get dataset name list
        dataset_names = request.http_get_dataset_names()
        print("dataset names=", dataset_names)
        for dataset in dataset_names:
            try:
                generator= KandinskyImageGenerator(minio_access_key=args.minio_access_key,
                                            minio_secret_key=args.minio_secret_key,
                                            dataset=dataset,
                                            num_bins=args.num_bins,
                                            top_k= args.top_k,
                                            batch_size= args.batch_size,
                                            learning_rate= args.learning_rate,
                                            steps= args.steps,
                                            generation_policy= args.generation_policy,
                                            sampling_policy= args.sampling_policy,
                                            send_job= args.send_job,
                                            save_csv= args.save_csv)

                generator.generate_images(num_images=args.num_images)

            except Exception as e:
                print("Error generating images for {}: {}".format(dataset, e))


if __name__ == "__main__":
    main()