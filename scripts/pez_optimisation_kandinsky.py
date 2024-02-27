import argparse
import io
import os
import sys
import torch
import torch.optim as optim
import msgpack

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from training_worker.ab_ranking.model.ab_ranking_linear import ABRankingModel
from kandinsky_worker.image_generation.img2img_generator import generate_img2img_generation_jobs_with_kandinsky
from kandinsky.models.clip_image_encoder.clip_image_encoder import KandinskyCLIPImageEncoder
from utility.minio import cmd
from data_loader.utils import get_object
from kandinsky.model_paths import PRIOR_MODEL_PATH
from kandinsky.pipelines.kandinsky_prior import KandinskyV22PriorPipeline
from diffusers import PriorTransformer

def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--minio-access-key', type=str, help='Minio access key')
        parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
        parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")
        parser.add_argument('--model-type', type=str, help='model type, linear or elm', default="linear")
        parser.add_argument('--steps', type=int, help='number of optimisation steps', default=100)
        parser.add_argument('--learning-rate', type=float, help='learning rate for optimization', default=0.01)
        parser.add_argument('--target-score', type=float, help='number of optimisation steps', default=5)
        parser.add_argument('--penalty-weight', type=float, help='weight of deviation panalty', default=1)
        parser.add_argument('--deviation-threshold', type=float, help='deviation penalty threshold', default=2)
        parser.add_argument('--send-job', action='store_true', default=False)
        parser.add_argument('--generate-step', type=int, default=100)
        parser.add_argument('--print-step', type=int, default=10)

        return parser.parse_args()

class KandinskyImageGenerator:
    def __init__(self,
                 minio_access_key,
                 minio_secret_key,
                 dataset,
                 model_type,
                 steps=100,
                 target_score=5.0,
                 penalty_weight=1,
                 deviation_threshold= 2,
                 send_job=False,
                 generate_step=100,
                 print_step=10
                 ):
        
        self.dataset= dataset
        self.model_type= model_type
        self.steps= steps
        self.penalty_weight= penalty_weight
        self.deviation_threshold= deviation_threshold
        self.target_score= target_score
        self.send_job= send_job
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
        
        # load kandinsky clip
        clip= KandinskyCLIPImageEncoder(device= self.device)
        clip.load_submodels()
        self.image_encoder= clip.vision_model

        # load scoring model
        self.scoring_model= self.load_scoring_model()
        self.mean= float(self.scoring_model.mean)
        self.std= float(self.scoring_model.standard_deviation)

        # get clip mean and std values
        # prior_model = PriorTransformer.from_pretrained(PRIOR_MODEL_PATH, subfolder="prior").to(self.device)

        # self.clip_mean= prior_model.clip_mean.clone().to(self.device)
        # self.clip_std= prior_model.clip_std.clone().to(self.device)

        self.clip_mean , self.clip_std, self.clip_max, self.clip_min= self.get_clip_distribution()
        print(self.clip_mean, self.clip_std)
        

    def get_clip_distribution(self):
        data = get_object(self.minio_client, "environmental/output/stats/clip_stats.msgpack")
        data_dict = msgpack.unpackb(data)

        mean_vector = torch.tensor(data_dict["mean"]).to(device=self.device, dtype=torch.float32)
        std_vector = torch.tensor(data_dict["std"]).to(device=self.device, dtype=torch.float32)
        max_vector = torch.tensor(data_dict["max"]).to(device=self.device, dtype=torch.float32)
        min_vector = torch.tensor(data_dict["min"]).to(device=self.device, dtype=torch.float32)

        return mean_vector, std_vector, max_vector, min_vector


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
    
    def get_initial_latent(self, batch_size=1):
        random_img= torch.zeros(1, 3, self.image_encoder.config.image_size, self.image_encoder.config.image_size).to(
            device=self.device, dtype=self.image_encoder.dtype
        )
        zero_image_emb = self.image_encoder(random_img)["image_embeds"]
        zero_image_emb = zero_image_emb.repeat(batch_size, 1)
        return zero_image_emb.to(
            device=self.device, dtype=torch.float32
        )
    
    def sample_embedding(self, num_samples=1):
        # Sample from a normal distribution using the mean and standard deviation vectors
        sampled_embeddings = torch.normal(self.clip_mean, self.clip_std)
        
        # Clip the sampled embeddings based on the min and max vectors to ensure they stay within observed bounds
        clipped_embeddings = torch.max(torch.min(sampled_embeddings, self.clip_max), self.clip_min)
        
        return clipped_embeddings.to(
            device=self.device, dtype=torch.float32
        )
    
    def penalty_function(self, embedding):
        """
        Calculates a penalty for embeddings that deviate from the mean beyond the allowed threshold (in standard deviations).
        
        Args:
        - embeddings (torch.Tensor): The current embeddings.
        - mean (torch.Tensor): The mean values for each dimension.
        - std (torch.Tensor): The standard deviation values for each dimension.
        - threshold (float): The number of standard deviations considered acceptable.
        
        Returns:
        - torch.Tensor: A scalar tensor representing the penalty.
        """
        # Standardize embeddings
        z_scores = (embedding - self.clip_mean) / self.clip_std

        # Calculate the squared distances beyond the threshold
        squared_distances = torch.where(torch.abs(z_scores) > self.deviation_threshold,
                                        (torch.abs(z_scores) - self.deviation_threshold)**2, 
                                        torch.tensor(0.0, device=embedding.device))
        # Sum the penalties
        penalty = squared_distances.sum()
        return penalty

    def generate_latent(self):
        # Ensure image embeddings require gradients
        #image_embedding= self.get_initial_latent()

        # features_data = get_object(self.minio_client, "environmental/0435/434997_clip_kandinsky.msgpack")
        # features_vector = msgpack.unpackb(features_data)["clip-feature-vector"]
        # image_embedding= torch.tensor(features_vector).to(device=self.device, dtype=torch.float32)
        
        sampled_embedding= self.sample_embedding()
        optimized_embedding = sampled_embedding.clone().detach().requires_grad_(True)

        # Setup the optimizer
        optimizer = optim.Adam([optimized_embedding], lr=0.001)

        for step in range(self.steps):
            optimizer.zero_grad()

            # Calculate the custom score
            inputs = optimized_embedding.reshape(len(optimized_embedding), -1)
            score = self.scoring_model.model.forward(inputs).squeeze()
            score= (score - self.mean) / self.std
            # Custom loss function
            # Original loss based on the scoring function
            score_loss = torch.abs(self.target_score - score)
            
            # Calculate the penalty for the embeddings
            penalty = self.penalty_weight * self.penalty_function(optimized_embedding)
            
            # Total loss
            total_loss = score_loss + penalty

            if self.send_job and (step % self.generate_step == 0):
                try:
                    response= generate_img2img_generation_jobs_with_kandinsky(
                        image_embedding=optimized_embedding,
                        negative_image_embedding=None,
                        dataset_name="test-generations",
                        prompt_generation_policy="pez_optimization",
                    )
                except:
                    print("An error occured.")

            if total_loss<0.01:
                break

            # Backpropagate
            total_loss.backward()

            if step % self.print_step == 0:
                # Debugging gradients: check the .grad attribute
                if optimized_embedding.grad is not None:
                    print(f"Step {step}: Gradient norm is {optimized_embedding.grad.norm().item()}")
                else:
                    print(f"Step {step}: No gradient computed.")

            optimizer.step()

            if step % self.print_step == 0:
                print(f"Step: {step}, Score: {score.item()}, Penalty: {penalty}, Loss: {total_loss.item()}")
        
        if self.send_job:
            try:
                response= generate_img2img_generation_jobs_with_kandinsky(
                    image_embedding=optimized_embedding,
                    negative_image_embedding=None,
                    dataset_name="test-generations",
                    prompt_generation_policy="pez_optimization",
                )
            except:
                print("An error occured.")

        return optimized_embedding

    def test_image_score(self):

        features_data1 = get_object(self.minio_client, "environmental/0435/434997_clip_kandinsky.msgpack")
        features_vector1 = msgpack.unpackb(features_data1)["clip-feature-vector"]
        features_vector1= torch.tensor(features_vector1).to(device=self.device, dtype=torch.float32)

        inputs1 = features_vector1.reshape(len(features_vector1), -1)
        score1 = self.scoring_model.model.forward(inputs1).squeeze()

        features_data2 = get_object(self.minio_client, "test-generations/0024/023629_clip_kandinsky.msgpack")
        features_vector2 = msgpack.unpackb(features_data2)["clip-feature-vector"]
        features_vector2= torch.tensor(features_vector2).to(device=self.device, dtype=torch.float32)

        inputs2 = features_vector2.reshape(len(features_vector2), -1)
        score2 = self.scoring_model.model.forward(inputs2).squeeze()

        print(f"score before {score1} and after {score2}")

def main():
    args= parse_args()
    # initialize generator
    generator= KandinskyImageGenerator(minio_access_key=args.minio_access_key,
                                       minio_secret_key=args.minio_secret_key,
                                       dataset=args.dataset,
                                       model_type=args.model_type,
                                       steps=args.steps,
                                       target_score=args.target_score,
                                       deviation_threshold=args.deviation_threshold,
                                       penalty_weight=args.penalty_weight,
                                       send_job= args.send_job,
                                       generate_step=args.generate_step,
                                       print_step=args.print_step)
    
    generator.generate_latent()

if __name__=="__main__":
    main()