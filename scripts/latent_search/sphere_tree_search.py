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
import faiss
import umap
import numpy as np
import matplotlib.pyplot as plt

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())
from data_loader.utils import get_object
from training_worker.sampling.models.directional_uniform_sampling_regression_fc import DirectionalSamplingFCRegressionNetwork
from training_worker.scoring.models.scoring_fc import ScoringFCNetwork
from kandinsky_worker.image_generation.img2img_generator import generate_img2img_generation_jobs_with_kandinsky
from training_worker.classifiers.models.elm_regression import ELMRegression
from training_worker.scoring.models.classifier_fc import ClassifierFCNetwork
from utility.minio import cmd
from utility.http import request

def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--minio-access-key', type=str, help='Minio access key')
        parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
        parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")
        parser.add_argument('--tag-name', type=str, help='Name of the tag to generate for', default="topic-forest")
        parser.add_argument('--defect-tag', type=str, help='Use this tag for filtering defects', default=None)
        parser.add_argument('--num-images', type=int, help='Number of images to generate', default=100)
        parser.add_argument('--nodes-per-iteration', type=int, help='Number of nodes to evaluate each iteration', default=1000)
        parser.add_argument('--branches-per-iteration', type=int, help='Number of branches to expand each iteration', default=10)
        parser.add_argument('--top-k', type=float, default=1)
        parser.add_argument('--max-nodes', type=int, help='Number of maximum nodes', default=1e+7)
        parser.add_argument('--jump-distance', type=float, help='Jump distance for each node', default=0.01)
        parser.add_argument('--batch-size', type=int, help='Inference batch size used by the scoring model', default=256)
        parser.add_argument('--steps', type=int, help='Optimization steps', default=200)
        parser.add_argument('--learning-rate', type=float, help='Optimization learning rate', default=0.001)
        parser.add_argument('--send-job', action='store_true', default=False)
        parser.add_argument('--save-csv', action='store_true', default=False)
        parser.add_argument('--sampling-policy', type=str, default="rapidly_exploring_tree_search")
        parser.add_argument('--optimize-samples', action='store_true', default=False)
        parser.add_argument('--classifier-weight', type=float, default=0.5)
        parser.add_argument('--ranking-weight', type=float, default=0.5)
        parser.add_argument('--graph-tree', action='store_true', default=False)

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
                 optimize_samples,
                 ranking_weight,
                 classifier_weight,
                 graph_tree,
                 defect_tag=None):
        
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
        self.ranking_weight= ranking_weight
        self.classifier_weight= classifier_weight
        self.graph_tree= graph_tree
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

        # get classifier model for selected tag
        self.classifier_model= ClassifierFCNetwork(minio_client=self.minio_client, tag_name=tag_name)
        self.classifier_model.load_model()

        # get classifier model for the defect tag
        self.defect_model= None
        if defect_tag:
            self.defect_model= ClassifierFCNetwork(minio_client=self.minio_client, tag_name=defect_tag)
            self.defect_model.load_model()

        # load all topic classifiers
        tag_names= request.http_get_tag_list()
        tag_names= [tag for tag in tag_names if "topic" in tag]

        self.classifier_models=[]
        for tag in tag_names:
            classifier_model= ClassifierFCNetwork(minio_client=self.minio_client, tag_name=tag)
            model_loaded= classifier_model.load_model()
            if model_loaded:
                self.classifier_models.append({
                    "model": classifier_model,
                    "tag_name": tag
                })

        # get distribution of clip vectors for the dataset
        self.clip_mean , self.clip_std, self.clip_max, self.clip_min, self.covariance_matrix= self.get_clip_distribution()

    
    def get_clip_distribution(self):
        data = get_object(self.minio_client, f"{self.dataset}/output/stats/clip_stats.msgpack")
        data_dict = msgpack.unpackb(data)

        # Convert to PyTorch tensors
        mean_vector = torch.tensor(data_dict["mean"], dtype=torch.float32).unsqueeze(0)
        std_vector = torch.tensor(data_dict["std"], dtype=torch.float32).unsqueeze(0)
        max_vector = torch.tensor(data_dict["max"], dtype=torch.float32).unsqueeze(0)
        min_vector = torch.tensor(data_dict["min"], dtype=torch.float32).unsqueeze(0)
        covariance_matrix = torch.tensor(data_dict["cov_matrix"], dtype=torch.float32)

        return mean_vector, std_vector, max_vector, min_vector, covariance_matrix
    
    def get_classifier_model(self, tag_name):
        input_path = f"{self.dataset}/models/classifiers/{tag_name}/"
        file_suffix = "elm-regression-clip-h-with-length.safetensors"

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

    def min_max_normalize_scores(self, scores):
        min_val = torch.min(scores)
        max_val = torch.max(scores)
        normalized_tensor = (scores - min_val) / (max_val - min_val)
        return normalized_tensor

    def setup_faiss(self, all_nodes):
        # Assuming all_nodes is a list of torch tensors (nodes)
        dimension = all_nodes[0].size(0)
        faiss_index = faiss.IndexFlatL2(dimension)
        
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)

        # Convert all_nodes to a contiguous array of float32, required by FAISS
        node_matrix = torch.stack(all_nodes).cpu().numpy().astype('float32')
        faiss_index.add(node_matrix)

        return faiss_index

    def compute_distances(self, faiss_index, nodes):
        # Convert new_points to numpy float32 array
        nodes = nodes.cpu().numpy().astype('float32')
        # Compute distances to all existing nodes
        distances, indices = faiss_index.search(nodes, 1)  # Find the nearest node
        return distances

    def find_nearest_points(self, point, num_samples, covariance_matrix):
        # Sampling from a multivariate Gaussian distribution
        distribution = torch.distributions.MultivariateNormal(point, covariance_matrix)
        clip_vectors = distribution.sample((num_samples,))
        clip_vectors = torch.clamp(clip_vectors, self.clip_min, self.clip_max)

        return clip_vectors

    def filter_defects(self, points, defect_threshold=0.6):
        points= points.to(device=self.device)
        # get defect scores
        scores= self.defect_model.predict(points, batch_size=points.size(0)).squeeze(1)
        # filter for indices of elements that aren't defective
        filtered_indices= torch.where(scores<defect_threshold)[0]
        # filter datapoints
        filtered_points= points[filtered_indices]

        return filtered_points.detach().cpu()

    def score_points(self, points):
        points= points.to(device=self.device)
        scores= self.scoring_model.predict(points, batch_size=1000)
        return scores.detach().cpu()
    
    def classifiy_points(self, points):
        points= points.to(device=self.device)
        scores= self.classifier_model.predict(points, batch_size=points.size(0))
        return scores.detach().cpu()
    
    def rank_points_by_distance(self, nodes, faiss_index):
        # calculate ranking and classifier scores
        classifier_scores = self.classifiy_points(nodes).squeeze(1)
        ranking_scores = self.score_points(nodes).squeeze(1)

        # increase classifier scores with a threshold
        classifier_scores= torch.where(classifier_scores>0.6, torch.tensor(1), classifier_scores)

        # rank by distance
        distances = self.compute_distances(faiss_index, nodes)
        distance_scores = torch.tensor(distances.squeeze())

        return distance_scores, classifier_scores, ranking_scores

    def rank_points_by_quality(self, nodes):
        # calculate ranking and classifier scores
        if self.classifier_weight!=0:
            classifier_scores = self.classifiy_points(nodes).squeeze(1)
            # increase classifier scores with a threshold
            classifier_scores= torch.where(classifier_scores>0.6, torch.tensor(1), classifier_scores)
        if self.ranking_weight!=0:
            ranking_scores = self.score_points(nodes).squeeze(1)

        # combine scores
        if self.ranking_weight!=0 and self.classifier_weight!=0:
            classifier_ranks= self.classifier_weight * self.min_max_normalize_scores(classifier_scores) 
            quality_ranks= self.ranking_weight * self.min_max_normalize_scores(ranking_scores)
            ranks= classifier_ranks + quality_ranks
        elif self.ranking_weight!=0:
            quality_ranks= self.min_max_normalize_scores(ranking_scores)
            ranks= quality_ranks
        elif self.classifier_weight!=0:
            classifier_ranks= self.min_max_normalize_scores(classifier_scores)
            ranks= quality_ranks

        return ranks, classifier_scores, ranking_scores 

    def expand_tree(self, nodes_per_iteration, branches_per_iteration, max_nodes, top_k, jump_distance, num_images):
        current_generation = [self.clip_mean.squeeze()]
        all_nodes = [self.clip_mean.squeeze()]
        if self.sampling_policy=="rapidly_exploring_tree_search":
            faiss_index = self.setup_faiss(all_nodes)

        all_classifier_scores = torch.tensor([], dtype=torch.float32)
        all_ranking_scores = torch.tensor([], dtype=torch.float32)

        # generate covariance matrix
        covariance_matrix = torch.diag((self.clip_std.pow(2) * jump_distance).squeeze(0))
        
        # Initialize tqdm
        pbar = tqdm(total=max_nodes)
        nodes=0
        while(nodes < max_nodes):
            next_generation = []
            
            for point in current_generation:
                # Find nearest k points to the current point
                nearest_points = self.find_nearest_points(point, nodes_per_iteration, covariance_matrix)
                # Filter defective points
                if self.defect_model:
                    nearest_points= self.filter_defects(nearest_points)
                    # skip if all nearest points are defective
                    if len(nearest_points)==0:
                        continue

                # Score these points
                if self.sampling_policy == "rapidly_exploring_tree_search":
                    ranks, classifier_scores, ranking_scores = self.rank_points_by_distance(nearest_points, faiss_index)
                elif self.sampling_policy == "jump_point_tree_search":
                    ranks, classifier_scores, ranking_scores = self.rank_points_by_quality(nearest_points)

                # Select top n points based on scores
                _, sorted_indices = torch.sort(ranks, descending=True)
                top_points = nearest_points[sorted_indices[:branches_per_iteration]]
                if self.classifier_weight!=0:
                    top_classifier_scores = classifier_scores[sorted_indices[:branches_per_iteration]]
                    top_classifier_scores= torch.where(top_classifier_scores>0.6, torch.tensor(1), top_classifier_scores)
                    all_classifier_scores = torch.cat((all_classifier_scores, top_classifier_scores), dim=0)
                if self.ranking_weight!=0:   
                    top_ranking_scores = ranking_scores[sorted_indices[:branches_per_iteration]]
                    all_ranking_scores = torch.cat((all_ranking_scores, top_ranking_scores), dim=0)

                # Keep track of all nodes and their scores for selection later
                all_nodes.extend(top_points)

                next_generation.extend(top_points)
                nodes+= nodes_per_iteration
                pbar.update(nodes_per_iteration)
                if nodes > max_nodes:
                    break

                # add selected points to the index
                current_nodes=top_points.cpu().numpy().astype('float32')
                if self.sampling_policy=="rapidly_exploring_tree_search":
                    faiss_index.add(current_nodes) 
            
            # Prepare for the next iteration
            current_generation = next_generation
        
        # Close the progress bar when done
        pbar.close()

        # graph tree
        if self.graph_tree:
            labels= self.label_nodes(all_nodes)
            self.graph_datapoints(all_nodes, labels)
        
        # After the final iteration, choose the top n highest scoring points overall
        if self.classifier_weight!=0:
            classifier_ranks= self.classifier_weight * self.min_max_normalize_scores(all_classifier_scores)
        if self.ranking_weight!=0: 
            quality_ranks= self.ranking_weight * self.min_max_normalize_scores(all_ranking_scores)

        if self.classifier_weight!=0 and self.ranking_weight!=0:
            all_ranks= classifier_ranks + quality_ranks
        elif self.classifier_weight!=0:
            all_ranks= classifier_ranks
        elif self.ranking_weight!=0:
            all_ranks= quality_ranks
    
        values, sorted_indices = torch.sort(all_ranks, descending=True)
        final_top_points = torch.stack(all_nodes, dim=0)[sorted_indices[:int(num_images/top_k)]]

        # select n random spheres from the top k spheres
        indices = torch.randperm(final_top_points.size(0))[:num_images]
        selected_points = final_top_points[indices]

        return selected_points
    
    def label_nodes(self, tree):
        labels = []
        num_nodes= tree.size(0)
        threshold= 0.6

        for start_index in range(0, num_nodes, self.batch_size):
            end_index = min(start_index + self.batch_size, num_nodes)
            nodes = tree[start_index:end_index]

            max_scores = torch.full((nodes.size(0),), fill_value=-float('inf'), dtype=torch.float32)
            max_labels = ["undefined"] * nodes.size(0)  # Initialize with 'undefined'

            # Process each batch with all classifiers
            for model in self.classifier_models:
                nodes = nodes.to(device=self.device)
                scores = model["model"].predict(nodes, batch_size=nodes.size(0)).squeeze(1)

                # Update labels based on scores
                for i in range(scores.size(0)):
                    if scores[i] > max_scores[i] and scores[i] > threshold:
                        max_scores[i] = scores[i]
                        max_labels[i] = model["tag_name"]  # Update label with the tag_name of the model

            # Add the batch's labels to the overall list
            labels.extend(max_labels)

        return labels
        
    def graph_datapoints(self, tree, labels):
        minio_path=f"{self.dataset}/output/tree_search/graph.png"
        reducer = umap.UMAP(random_state=42)
        umap_embeddings = reducer.fit_transform(tree)

        # Convert labels to categorical type for color mapping
        unique_labels = np.unique(labels)
        label_to_color = {label: idx for idx, label in enumerate(unique_labels)}
        color_labels = [label_to_color[label] for label in labels]

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=color_labels, cmap='viridis', alpha=0.6)
        # Create a color bar with label names
        cbar = plt.colorbar(scatter, ticks=np.arange(len(unique_labels)))
        cbar.ax.set_yticklabels(unique_labels)  # Set text labels for the color bar
        cbar.set_label('Classifier Labels')
        
        plt.title('UMAP Projection of CLIP Vectors, Clustered by topic')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')

        buffer = io.BytesIO()
        plt.savefig("graph.png", buffer)
        buffer.seek(0)

        cmd.upload_data(self.minio_client, 'datasets', minio_path, buffer)
        # Remove the temporary file
        os.remove("graph.png")
    
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
                ranking_scores = self.scoring_model.model(batch_embeddings).squeeze()
                
                # Compute classifier scores for the current batch of embeddings
                classifier_scores = self.classifier_model.model(batch_embeddings).squeeze()
                
                # Calculate the total loss for the batch
                total_loss = - (self.ranking_weight * ranking_scores.mean()) - (self.classifier_weight * classifier_scores.mean())

                # Backpropagate
                total_loss.backward()

                optimizer.step()

                print(f"Batch: {batch_idx + 1}/{num_batches}, Step: {step}, Mean ranking Score: {ranking_scores.mean().item()}, Mean classifier Score: {classifier_scores.mean().item()}, Loss: {total_loss.item()}")

        return batch_embeddings

    def generate_images(self, nodes_per_iteration, branches_per_iteration, max_nodes, top_k, jump_distance, num_images):
        clip_vectors= self.expand_tree(nodes_per_iteration, branches_per_iteration, max_nodes, top_k, jump_distance, num_images)

        # Optimization step
        if(self.optimize_samples):
            clip_vectors = self.optimize_datapoints(clip_vectors)
        
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
        minio_path= minio_path + f"/{current_date}-{self.sampling_policy}-{self.dataset}-{self.tag_name}.csv"
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
                                        optimize_samples= args.optimize_samples,
                                        ranking_weight= args.ranking_weight,
                                        classifier_weight= args.classifier_weight,
                                        graph_tree= args.graph_tree,
                                        defect_tag= args.defect_tag)

    generator.generate_images(nodes_per_iteration=args.nodes_per_iteration,
                          branches_per_iteration=args.branches_per_iteration,
                          max_nodes= args.max_nodes,
                          top_k= args.top_k,
                          jump_distance= args.jump_distance,
                          num_images= args.num_images)

if __name__ == "__main__":
    main()
