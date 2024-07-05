import argparse
import os
import sys
import torch
import msgpack
from tqdm import tqdm
import torch.optim as optim
import faiss
import torch.nn.functional as F
from PIL import Image

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())
from data_loader.utils import get_object
from training_worker.scoring.models.clip_to_clip_fc import CliptoClipFCNetwork
from utility.minio import cmd
from utility.http import request
from utility.path import separate_bucket_and_file_path
from kandinsky.models.kandisky import KandinskyPipeline

def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--minio-access-key', type=str, help='Minio access key')
        parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
        parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")
        parser.add_argument('--batch-size', type=int, help='Inference batch size used by the clip model', default=256)
        parser.add_argument('--steps', type=int, help='Optimization steps', default=200)
        parser.add_argument('--learning-rate', type=float, help='Optimization learning rate', default=0.001)
        parser.add_argument('--sampling-policy', type=str, default="rapidly_exploring_tree_search")
        parser.add_argument('--optimize-samples', action='store_true', default=False)

        return parser.parse_args()

class InversionPipeline:
    def __init__(self,
                 minio_access_key,
                 minio_secret_key,
                 dataset,
                 batch_size,
                 steps,
                 learning_rate,
                 sampling_policy,
                 optimize_samples):
        
        # parameters
        self.dataset = dataset
        self.batch_size= batch_size
        self.steps= steps
        self.learning_rate= learning_rate  
        self.sampling_policy= sampling_policy  
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

        # load clip to output clip model
        self.output_clip_model= CliptoClipFCNetwork(minio_client=self.minio_client, dataset=dataset)
        self.output_clip_model.load_model()

        # load kandinsky pipeline
        self.kandisnky_generator = KandinskyPipeline(
            device=self.device,
            width= 512,
            height= 512,
            batch_size=1,
            decoder_steps= 40,
            strength= 0.75,
            decoder_guidance_scale= 8
        )

        self.kandisnky_generator.load_models(task_type="img2img")

        # get distribution of clip vectors for the dataset
        self.clip_mean , self.clip_std, self.clip_max, self.clip_min, self.covariance_matrix= self.get_clip_distribution()

    def get_clip_vectors(self, tagged_images):
        clip_vectors=[]

        for image in tqdm(tagged_images):
            try:
                file_path= image['file_path']
                bucket_name, input_file_path = separate_bucket_and_file_path(file_path)
                file_path = os.path.splitext(input_file_path)[0]
                
                output_clip_path = file_path + "_clip-h.msgpack"
                features_data = cmd.get_file_from_minio(self.minio_client, bucket_name, output_clip_path)
                features_vector = msgpack.unpackb(features_data.data)["clip-feature-vector"]
                output_clip_vector= torch.tensor(features_vector)
                clip_vectors.append(output_clip_vector)
            except Exception as e:
                print(f"An excpection occured while loading clip vectors: {e}")

        clip_vectors = torch.stack(clip_vectors).squeeze().to(device=self.device)
        return clip_vectors

    def get_tagged_images(self):
        image_categories = {}
        # get all data based on tag
        tag_list = request.http_get_tag_list()

        for tag in tag_list:
            if "perspective" in tag['tag_string'] or "topic" in tag['tag_string']:
                tagged_images = request.http_get_tagged_extracts(tag["tag_id"])

                if len(tagged_images)<20:
                    continue
               
                print(f"loading clip vectors from the tag {tag['tag_string']}.........")

                # get image hashes
                image_hashes= [tag['image_hash'] for tag in tagged_images]
                
                # get clip vectors
                clip_vectors= self.get_clip_vectors(tagged_images)

                image_categories[tag['tag_string']]={
                    "image_hashes": image_hashes,
                    "clip_vectors": clip_vectors
                }
        
        return image_categories

    
    def get_clip_distribution(self):
        data = get_object(self.minio_client, f"{self.dataset}/output/stats/clip_stats.msgpack")
        data_dict = msgpack.unpackb(data)

        # Convert to PyTorch tensors
        mean_vector = torch.tensor(data_dict["mean"], dtype=torch.float32, device=self.device).unsqueeze(0)
        std_vector = torch.tensor(data_dict["std"], dtype=torch.float32, device=self.device).unsqueeze(0)
        max_vector = torch.tensor(data_dict["max"], dtype=torch.float32, device=self.device).unsqueeze(0)
        min_vector = torch.tensor(data_dict["min"], dtype=torch.float32, device=self.device).unsqueeze(0)
        covariance_matrix = torch.tensor(data_dict["cov_matrix"], dtype=torch.float32, device=self.device)

        return mean_vector, std_vector, max_vector, min_vector, covariance_matrix

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
        clip_vectors = torch.clamp(clip_vectors, self.clip_min, self.clip_max).to(self.device)

        return clip_vectors
    
    def get_similarities(self, points, target):
        # get output clip vectors
        points= points.to(device=self.device)
        outputs= self.output_clip_model.predict(points, batch_size=64)

        # Normalize vectors to have unit norm
        outputs_norm = F.normalize(outputs, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)
        # Calculate cosine similarity and convert to loss
        cosine_sims = torch.mm(outputs_norm, target_norm.t()).squeeze()

        return cosine_sims
    
    def expand_tree_node(self, nodes, faiss_index, target):

        cosine_similiarities= self.get_similarities(nodes, target)

        # rank by distance and quality
        similarity_ranks= self.min_max_normalize_scores(cosine_similiarities)
        distances = self.compute_distances(faiss_index, nodes)
        distances = torch.tensor(distances.squeeze()).to(device= self.device)
        distance_ranks= self.min_max_normalize_scores(distances)

        ranks= similarity_ranks + distance_ranks

        return ranks, cosine_similiarities
    
    def jump_point(self, nodes, target):

        cosine_similiarities= self.get_similarities(nodes, target)

        # rank by distance and quality
        similarity_ranks= self.min_max_normalize_scores(cosine_similiarities)

        return similarity_ranks, cosine_similiarities

    def expand_tree(self, target, nodes_per_iteration, branches_per_iteration, max_nodes, top_k, jump_distance, num_images):
        current_generation = [self.clip_mean.squeeze()]
        all_nodes = [self.clip_mean.squeeze()]
        if self.sampling_policy=="rapidly_exploring_tree_search":
            faiss_index = self.setup_faiss(all_nodes)

        all_cosine_similarities = torch.tensor([], dtype=torch.float32, device=self.device)

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

                # Score these points
                if self.sampling_policy == "rapidly_exploring_tree_search":
                    ranks, cosine_similarities = self.expand_tree_node(nearest_points, faiss_index, target)
                elif self.sampling_policy == "jump_point_tree_search":
                    ranks, cosine_similarities = self.jump_point(nearest_points, target)

                # Select top n points based on scores
                _, sorted_indices = torch.sort(ranks, descending=True)
                top_points = nearest_points[sorted_indices[:branches_per_iteration]]

                top_cosine_similarities = cosine_similarities[sorted_indices[:branches_per_iteration]]
                all_cosine_similarities = torch.cat((all_cosine_similarities, top_cosine_similarities), dim=0)

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
        
        # After the final iteration, choose the top n highest scoring points overall
        values, sorted_indices = torch.sort(all_cosine_similarities, descending=True)
        final_top_points = torch.stack(all_nodes, dim=0)[sorted_indices]

        # select the the inverted clip vector
        inverted_clip = final_top_points[0]

        return inverted_clip
    
    def optimize_datapoints(self, image_hashes, target_vectors):
        # get number of target vectors
        num_images= len(target_vectors)

        # sample random clip vectors
        clip_vectors = torch.normal(mean=self.clip_mean.repeat(num_images, 1),
                                        std=self.clip_std.repeat(num_images, 1))

        # Calculate the total number of batches
        num_batches = num_images // self.batch_size + (0 if num_images % self.batch_size == 0 else 1)

        optimized_embeddings_list=[]
        cosine_similarities=[]

        for batch_idx in range(num_batches):
            # Select a batch of embeddings
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(clip_vectors))
            batch_embeddings = clip_vectors[start_idx:end_idx].clone().detach().requires_grad_(True)
            target_batch = target_vectors[start_idx:end_idx].clone().detach().requires_grad_(True)
            
            # Setup the optimizer for the current batch
            optimizer = optim.Adam([batch_embeddings], lr=self.learning_rate)
            
            for step in range(self.steps):
                optimizer.zero_grad()

                # Compute ranking scores for the current batch of embeddings
                outputs = self.output_clip_model.model(batch_embeddings).squeeze()
                
                # Compute the cosine similarities to the target clip vectors
                # Normalize vectors to have unit norm
                outputs_norm = F.normalize(outputs, p=2, dim=1)
                target_norm = F.normalize(target_batch, p=2, dim=1)
                # Calculate cosine similarity and convert to loss
                cosine_sims = torch.mm(outputs_norm, target_norm.t()).squeeze()

                # Convert cosine similarities to loss
                cosine_loss = 1 - cosine_sims
                        
                # Calculate the total loss for the batch
                total_loss = cosine_loss.mean()

                # Backpropagate
                total_loss.backward()

                optimizer.step()

                print(f"Batch: {batch_idx + 1}/{num_batches}, Step: {step}, Mean cosine similarity: {cosine_sims.mean().item()}, Loss: {total_loss.item()}")

            # After optimization, detach and add the optimized batch embeddings and their cosine similarities to the target to the list
            optimized_batch_embeddings = batch_embeddings.detach()
            cosine_sims = cosine_sims.detach()
            optimized_embeddings_list.extend([emb for emb in optimized_batch_embeddings])
            cosine_similarities.extend([cosine_sim for cosine_sim in cosine_sims])
      
        cosine_similarities = torch.stack(cosine_similarities).squeeze()
        cosine_sims, sorted_indices = torch.sort(cosine_similarities, descending=True)
        sorted_clip_vectors =  torch.stack(optimized_embeddings_list).squeeze()[sorted_indices]
        sorted_indices_list = sorted_indices.tolist()[0]
        image_hashes_sorted = [image_hashes[i] for i in sorted_indices_list]

        return sorted_clip_vectors, cosine_sims, image_hashes_sorted
    
    def image_inversion(self):
        target_images= self.get_tagged_images()

        for key, data in target_images.items():
            print(f"optimizing images in the {key} category")

            images_hashes= data['image_hashes']
            target_vectors= data['clip_vectors']

            sorted_clip_vectors, sorted_hashes= self.optimize_datapoints(images_hashes, target_vectors)

            self.generate_images(tag_name= key, clip_vectors= sorted_clip_vectors)

    def generate_images(self, tag_name, clip_vectors):
        print(f"generating images for the tag {tag_name}")

        init_image= Image.open("./test/test_inpainting/white_512x512.jpg") 
        # generate each image
        for input_clip in tqdm(clip_vectors):
            print(f"input clip shape: {input_clip.shape}")
            # generate image
            image, _ = self.kandisnky_generator.generate_img2img(init_img=init_image,
                                                                 image_embeds= input_clip)
            
            _, img_byte_arr = self.kandisnky_generator.convert_image_to_png(image)

            cmd.upload_data(self.minio_client, "extracts", f"{self.dataset}/output/{tag_name}", img_byte_arr)

def main():
    args= parse_args()

    # initialize generator
    generator= InversionPipeline(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        dataset=args.dataset,
                                        batch_size= args.batch_size,
                                        steps= args.steps,
                                        learning_rate= args.learning_rate,
                                        sampling_policy= args.sampling_policy,
                                        optimize_samples= args.optimize_samples)

    generator.image_inversion()

if __name__ == "__main__":
    main()