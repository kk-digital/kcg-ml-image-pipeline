import argparse
from datetime import datetime
import io
import os
import sys
import pandas as pd
import torch
import msgpack
from tqdm import tqdm
import torch.optim as optim

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())
from data_loader.utils import get_object
from training_worker.sampling.models.directional_uniform_sampling_regression_fc import DirectionalSamplingFCRegressionNetwork
from kandinsky_worker.image_generation.img2img_generator import generate_img2img_generation_jobs_with_kandinsky
from training_worker.classifiers.models.elm_regression import ELMRegression
from training_worker.scoring.models.classifier_fc import ClassifierFCNetwork
from utility.minio import cmd

def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--minio-access-key', type=str, help='Minio access key')
        parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
        parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")
        parser.add_argument('--tag-name', type=str, help='Name of the tag to generate for', default="topic-forest")
        parser.add_argument('--num-images', type=int, help='Number of images to generate', default=100)
        parser.add_argument('--nodes-per-iteration', type=int, help='Number of nodes to evaluate each iteration', default=1000)
        parser.add_argument('--top-k', type=int, help='Number of nodes to expand on each iteration', default=10)
        parser.add_argument('--max-nodes', type=int, help='Number of maximum nodes', default=1e+6)
        parser.add_argument('--jump-distance', type=float, help='Jump distance for each node', default=0.01)
        parser.add_argument('--batch-size', type=int, help='Inference batch size used by the scoring model', default=256)
        parser.add_argument('--steps', type=int, help='Optimization steps', default=200)
        parser.add_argument('--learning-rate', type=float, help='Optimization learning rate', default=0.001)
        parser.add_argument('--send-job', action='store_true', default=False)
        parser.add_argument('--save-csv', action='store_true', default=False)
        parser.add_argument('--sampling-policy', type=str, default="rapidly_exploring_tree_search")
        parser.add_argument('--optimize-samples', action='store_true', default=False)

        return parser.parse_args()

class RapidlyExploringTreeSearch:
    def __init__(self,
                 minio_access_key,
                 minio_secret_key,
                 dataset,
                 tag_name,
                 batch_size,
                 steps,
                 learning_rate,
                 sampling_policy,
                 send_job,
                 save_csv,
                 optimize_samples):
        
        # parameters
        self.dataset= dataset  
        self.tag_name= tag_name
        self.batch_size= batch_size
        self.steps= steps
        self.learning_rate= learning_rate  
        self.sampling_policy= sampling_policy  
        self.send_job= send_job
        self.save_csv= save_csv
        self.optimize_samples= optimize_samples
        # get minio client
        self.minio_client = cmd.get_minio_client(minio_access_key=minio_access_key,
                                                minio_secret_key=minio_secret_key)
        
        # get device
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = torch.device(device)

        self.sphere_scoring_model= DirectionalSamplingFCRegressionNetwork(minio_client=self.minio_client, dataset=dataset)
        self.sphere_scoring_model.load_model()

        # get classifier model for selected tag
        self.classifier_model= ClassifierFCNetwork(minio_client=self.minio_client, tag_name=tag_name)

        # get distribution of clip vectors for the dataset
        self.clip_mean , self.clip_std, self.clip_max, self.clip_min= self.get_clip_distribution()
        self.min_radius= torch.tensor(self.sphere_scoring_model.max_scaling_factors).to(device=self.device)
        self.max_radius= torch.tensor(self.sphere_scoring_model.min_scaling_factors).to(device=self.device)
    
    def get_clip_distribution(self):
        data = get_object(self.minio_client, f"{self.dataset}/output/stats/clip_stats.msgpack")
        data_dict = msgpack.unpackb(data)

        # Convert to PyTorch tensors
        mean_vector = torch.tensor(data_dict["mean"], device=self.device, dtype=torch.float32)
        std_vector = torch.tensor(data_dict["std"], device=self.device, dtype=torch.float32)
        max_vector = torch.tensor(data_dict["max"], device=self.device, dtype=torch.float32)
        min_vector = torch.tensor(data_dict["min"], device=self.device, dtype=torch.float32)

        return mean_vector, std_vector, max_vector, min_vector
    
    def get_classifier_model(self, tag_name):
        input_path = f"{self.dataset}/models/classifiers/{tag_name}/"
        file_suffix = "elm-regression-clip-h.safetensors"

        # Use the MinIO client's list_objects method directly with recursive=True
        model_files = [obj.object_name for obj in self.minio_client.list_objects('datasets', prefix=input_path, recursive=True) if obj.object_name.endswith(file_suffix)]
        
        if not model_files:
            print(f"No .safetensors models found for tag: {tag_name}")
            return None

        # Assuming there's only one model per tag or choosing the first one
        model_files.sort(reverse=True)
        model_file = model_files[0]
        print(f"Loading model: {model_file}")

        return self.load_model_with_filename(self.minio_client, model_file, tag_name)

    def load_model_with_filename(self, minio_client, model_file, model_info=None):
        model_data = minio_client.get_object('datasets', model_file)
        
        clip_model = ELMRegression(device=self.device)
        
        # Create a BytesIO object from the model data
        byte_buffer = io.BytesIO(model_data.data)
        clip_model.load_safetensors(byte_buffer)

        print(f"Model loaded for tag: {model_info}")
        
        return clip_model
    
    def find_nearest_points(self, sphere, num_samples, covariance_matrix):
        dim= sphere.size(1)//2
        point = sphere[:,:dim].squeeze(0)
        
        # Sampling from a multivariate Gaussian distribution
        distribution = torch.distributions.MultivariateNormal(point, covariance_matrix)
        clip_vectors = distribution.sample((num_samples,))
        clip_vectors = torch.clamp(clip_vectors, self.clip_min, self.clip_max)

        # sample random scaling factors
        radii= torch.rand(num_samples, len(self.max_radius), device=self.device) * (self.max_radius - self.min_radius) + self.min_radius
        sphere_centers= torch.cat([clip_vectors, radii], dim=1)

        return sphere_centers

    def score_points(self, points):
        scores= self.sphere_scoring_model.predict(points, batch_size=1000)
        return scores
    
    def classifiy_points(self, points):
        dim= points.size(1)//2
        points = points[:,:dim]
        scores= self.classifier_model.predict(points, batch_size=points.size(0)).to(device=self.device)
        return scores

    def expand_tree(self, nodes_per_iteration, max_nodes, top_k, jump_distance, num_images):
        radius= torch.rand(1, len(self.max_radius), device=self.device) * (self.max_radius - self.min_radius) + self.min_radius
        sphere= torch.cat([self.clip_mean, radius], dim=1)
        current_generation = [sphere.squeeze()]
        all_nodes = []
        all_scores = torch.tensor([], dtype=torch.float32, device=self.device)

        # generate covariance matrix
        covariance_matrix = torch.diag((self.clip_std * jump_distance).squeeze(0))
        
        # Initialize tqdm
        pbar = tqdm(total=max_nodes)
        nodes=0
        while(nodes < max_nodes):
            next_generation = []
            
            for point in current_generation:
                point= point.unsqueeze(0)
                # Find nearest k points to the current point
                nearest_points = self.find_nearest_points(point, nodes_per_iteration, covariance_matrix)
                
                # Score these points
                nearest_scores = self.classifiy_points(nearest_points)
                
                # Select top n points based on scores
                _, sorted_indices = torch.sort(nearest_scores.squeeze(), descending=True)
                top_points = nearest_points[sorted_indices[:top_k]]
                top_scores = nearest_scores[sorted_indices[:top_k]]

                # Keep track of all nodes and their scores for selection later
                all_scores = torch.cat((all_scores, top_scores), dim=0)
                all_nodes.extend(top_points)

                next_generation.extend(top_points)
                nodes+= nodes_per_iteration
                pbar.update(nodes_per_iteration)
                if nodes > max_nodes:
                    break
            
            # Prepare for the next iteration
            current_generation = next_generation
        
        # Close the progress bar when done
        pbar.close()
        
        # After the final iteration, choose the top n highest scoring points overall
        values, sorted_indices = torch.sort(all_scores.squeeze(1), descending=True)
        final_top_points = torch.stack(all_nodes, dim=0)[sorted_indices]
        # ranking_scores= self.score_points(final_top_points)

        # values, sorted_indices = torch.sort(ranking_scores.squeeze(1), descending=True)
        final_top_points=final_top_points[:num_images]

        # select n random spheres from the top k spheres
        # indices = torch.randperm(final_top_points.size(0))[:num_images]
        # selected_points = final_top_points[indices]

        return final_top_points
    
    def optimize_datapoints(self, clip_vectors):
        # Calculate the total number of batches
        num_batches = len(clip_vectors) // self.batch_size + (0 if len(clip_vectors) % self.batch_size == 0 else 1)

        for batch_idx in range(num_batches):
            # Select a batch of embeddings
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(clip_vectors))
            batch_embeddings = clip_vectors[start_idx:end_idx].clone().detach().requires_grad_(True)
            
            # Setup the optimizer for the current batch
            optimizer = optim.Adam([batch_embeddings], lr=self.learning_rate)
            
            for step in range(self.steps):
                optimizer.zero_grad()

                # Compute ranking scores for the current batch of embeddings
                ranking_scores = self.sphere_scoring_model.model(batch_embeddings).squeeze()
                
                # Compute classifier scores for the current batch of embeddings
                classifier_scores = self.classifier_model.model(batch_embeddings[:,:1280]).squeeze()
                
                # Calculate the total loss for the batch
                total_loss = - (0.1 * ranking_scores.mean()) - (10 * classifier_scores.mean())

                # Backpropagate
                total_loss.backward()

                optimizer.step()

                print(f"Batch: {batch_idx + 1}/{num_batches}, Step: {step}, Mean ranking Score: {ranking_scores.mean().item()}, Mean classifier Score: {classifier_scores.mean().item()}, Loss: {total_loss.item()}")

        return batch_embeddings

    def generate_images(self, nodes_per_iteration, max_nodes, top_k, jump_distance, num_images):
        spheres= self.expand_tree(nodes_per_iteration, max_nodes, top_k, jump_distance, num_images)

        # Optimization step
        if(self.optimize_samples):
            spheres = self.optimize_datapoints(spheres)
        
        clip_vectors = spheres[:,:1280]
        df_data=[]

        for clip_vector in clip_vectors:
            if self.send_job:
                try:
                    response= generate_img2img_generation_jobs_with_kandinsky(
                        image_embedding=clip_vector.unsqueeze(0),
                        negative_image_embedding=None,
                        dataset_name="test-generations",
                        prompt_generation_policy=self.sampling_policy
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
                    'generation_policy_string': self.sampling_policy,
                    'time': task_time
                })

        if self.save_csv:
            self.store_uuids_in_csv_file(df_data)
        
        print("Jobs were sent for generation.")

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
        minio_path= minio_path + f"/{current_date}-{self.sampling_policy}-{self.dataset}.csv"
        cmd.upload_data(self.minio_client, 'datasets', minio_path, buffer)
        # Remove the temporary file
        os.remove(local_path)

def main():
    args= parse_args()

    # initialize generator
    generator= RapidlyExploringTreeSearch(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        dataset=args.dataset,
                                        tag_name= args.tag_name,
                                        batch_size= args.batch_size,
                                        steps= args.steps,
                                        learning_rate= args.learning_rate,
                                        sampling_policy= args.sampling_policy,
                                        send_job= args.send_job,
                                        save_csv= args.save_csv,
                                        optimize_samples= args.optimize_samples)

    generator.generate_images(nodes_per_iteration=args.nodes_per_iteration,
                          max_nodes= args.max_nodes,
                          top_k= args.top_k,
                          jump_distance= args.jump_distance,
                          num_images= args.num_images)

if __name__ == "__main__":
    main()
