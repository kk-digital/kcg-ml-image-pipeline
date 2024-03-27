import argparse
import math
import os
import sys
import msgpack
import numpy as np
import torch
import torch.optim as optim

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())
from data_loader.utils import get_object
from training_worker.sampling.models.uniform_sampling_regression_fc import SamplingFCRegressionNetwork
from training_worker.scoring.models.scoring_fc import ScoringFCNetwork
from kandinsky_worker.image_generation.img2img_generator import generate_img2img_generation_jobs_with_kandinsky
from utility.minio import cmd

def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--minio-access-key', type=str, help='Minio access key')
        parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
        parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")
        parser.add_argument('--num-images', type=int, help='Number of images to generate', default=1000)
        parser.add_argument('--top-k', type=float, help='Portion of spheres to select from', default=0.1)
        parser.add_argument('--total-spheres', type=int, help='Number of random spheres to rank', default=500000)
        parser.add_argument('--selected-spheres', type=int, help='Number of spheres to sample from', default=10)
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
            
            # get the signle point scoring model
            self.scoring_model= ScoringFCNetwork(minio_client=self.minio_client, dataset=dataset)
            self.scoring_model.load_model()

            # get the sphere average score model
            self.sphere_scoring_model= SamplingFCRegressionNetwork(minio_client=self.minio_client, dataset=dataset)
            self.sphere_scoring_model.load_model()

            # get min and max radius values
            self.min_radius= self.sphere_scoring_model.min_radius.item()
            self.max_radius= self.sphere_scoring_model.max_radius.item()

            # get distribution of clip vectors for the dataset
            self.clip_mean , self.clip_std, self.clip_max, self.clip_min= self.get_clip_distribution()
    
    def get_clip_distribution(self):
        data = get_object(self.minio_client, f"{self.dataset}/output/stats/clip_stats.msgpack")
        data_dict = msgpack.unpackb(data)

        mean_vector = np.array(data_dict["mean"][0])
        std_vector = np.array(data_dict["std"][0])
        max_vector = np.array(data_dict["max"][0])
        min_vector = np.array(data_dict["min"][0])

        return mean_vector, std_vector, max_vector, min_vector

    def generate_spheres(self):
        num_spheres= self.total_spheres

        # Generate sphere centers
        sphere_centers = np.random.normal(loc=self.clip_mean, scale=self.clip_std, size=(num_spheres, len(self.clip_mean)))
        
        # Optionally, you may want to clip the generated centers to ensure they fall within expected min/max bounds
        sphere_centers = np.clip(sphere_centers, self.clip_min, self.clip_max)

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
                batch_scores= self.sphere_scoring_model.predict(batch, batch_size= self.batch_size).tolist()
                scores.extend(batch_scores)
                batch=[]
          
        print("scores: ",scores)  
        sorted_indexes= np.flip(np.argsort(scores))[:self.selected_spheres]
        top_spheres=[generated_spheres[i] for i in sorted_indexes]

        return top_spheres
    
    def sample_clip_vectors(self, num_samples):
        # get spheres
        spheres= self.rank_and_select_spheres()
        dim = len(spheres[0]['sphere_center'])
        points_per_sphere = max(int(num_samples / self.top_k), 1000)

        clip_vectors=[]
        scores= []
        for i, sphere in enumerate(spheres):
            center= sphere['sphere_center']
            radius= sphere['radius']

            # Calculate z-scores for each feature
            z_scores = (center - self.clip_mean) / self.clip_std

            # Calculate proportional direction adjustments based on z-scores
            adjustment_factor = np.clip(np.abs(z_scores), 0, 1)  # This caps the maximum adjustment
            direction_adjustment = -np.sign(z_scores) * adjustment_factor

            for j in range(points_per_sphere):
                random_direction= np.random.randn(dim)
                direction = direction_adjustment + random_direction
                direction /= np.linalg.norm(direction)
                
                # Randomly choose a magnitude within the radius
                magnitude = (np.random.rand()) * radius # Square root for uniform sampling in volume

                # Compute the point
                point = center + (direction * magnitude)

                # Clamp the point between the min and max vectors
                point = np.clip(point, self.clip_min, self.clip_max)
                point = torch.tensor(point).unsqueeze(0).to(device= self.device, dtype=torch.float32)
                # get score
                with torch.no_grad():
                    score= self.scoring_model.model(point).item()

                scores.append(score)
                clip_vectors.append(point)
        
        sorted_indexes= np.flip(np.argsort(scores))[:num_samples]
        clip_vectors= [clip_vectors[index] for index in sorted_indexes]
        mean_scores= np.mean([scores[index] for index in sorted_indexes])

        print(f"initial average score: {mean_scores}")

        return clip_vectors
    
    def gradient_descent_optimization(self, clip_vectors):

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
            optimizer = optim.Adam([batch_embeddings], lr=0.001)
            
            for step in range(200):
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
    
    def generate_images(self, num_images):
        # generate clip vectors
        clip_vectors= self.sample_clip_vectors(num_samples=num_images)
        optimized_vectors= self.gradient_descent_optimization(clip_vectors)

        for clip_vector in optimized_vectors:
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