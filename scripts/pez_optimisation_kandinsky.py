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
                 deviation_threshold= 2
                 ):
        
        self.dataset= dataset
        self.model_type= model_type
        self.steps= steps
        self.penalty_weight= penalty_weight
        self.deviation_threshold= deviation_threshold

        # get minio client
        self.minio_client = cmd.get_minio_client(minio_access_key=minio_access_key,
                                                minio_secret_key=minio_secret_key)
        
        # get device
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = torch.device(device)

        self.target_score= torch.tensor(target_score, dtype=torch.float, device=self.device, requires_grad=True)
        
        # load kandinsky clip
        clip= KandinskyCLIPImageEncoder(device= self.device)
        clip.load_submodels()
        self.image_encoder= clip.vision_model

        # load scoring model
        self.scoring_model= self.load_scoring_model()
        self.mean= float(self.scoring_model.mean)
        self.std= float(self.scoring_model.standard_deviation)

        # get clip mean and std values
        prior_model = PriorTransformer.from_pretrained(PRIOR_MODEL_PATH, subfolder="prior").to(self.device)

        self.clip_mean= prior_model.clip_mean.clone().to(self.device)
        self.clip_std= prior_model.clip_std.clone().to(self.device)

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
    
    def get_zero_embed(self, batch_size=1):
        zero_img = torch.zeros(1, 3, self.image_encoder.config.image_size, self.image_encoder.config.image_size).to(
            device=self.device, dtype=self.image_encoder.dtype
        )
        zero_image_emb = self.image_encoder(zero_img)["image_embeds"]
        zero_image_emb = zero_image_emb.repeat(batch_size, 1)
        return zero_image_emb.to(
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
    
    def harmonic_loss(self, score_loss, penalty):
        # Ensure non-zero to avoid division by zero
        epsilon = 1e-6
        score_loss_abs = torch.abs(score_loss) + epsilon
        penalty_abs = torch.abs(penalty) + epsilon
        
        # Harmonic mean
        harmonic_mean = 2 / ((1 / score_loss_abs) + (1 / penalty_abs))
        
        # Inverting the harmonic mean so that lower is better
        loss = 1 / harmonic_mean
        
        return loss

    def generate_latent(self):
        # Ensure image embeddings require gradients
        image_embedding= self.get_zero_embed()
        optimized_embedding = image_embedding.clone().detach().requires_grad_(True)

        # Setup the optimizer
        optimizer = optim.Adam([optimized_embedding], lr=0.01)

        for step in range(self.steps):
            optimizer.zero_grad()

            # normalized_embeddings = (optimized_embedding - self.clip_mean) / self.clip_std
            # clamped_embeddings = torch.clamp(normalized_embeddings, -10, 10)
            # optimized_embedding = (clamped_embeddings * self.clip_std) + self.clip_mean

            # Calculate the custom score
            inputs = optimized_embedding.reshape(len(optimized_embedding), -1)
            score = self.scoring_model.model.forward(inputs).squeeze()
            score= (score - self.mean) / self.std
            # Custom loss function
            # Original loss based on the scoring function
            score_loss = self.target_score - score
            
            # Calculate the penalty for the embeddings
            penalty = self.penalty_weight * self.penalty_function(optimized_embedding)
            
            # Total loss
            total_loss = self.harmonic_loss(score_loss, penalty)

            if total_loss==0:
                break

            # Backpropagate
            total_loss.backward()
            optimizer.step()

            print(f"Step: {step}, Score: {score.item()}, Penalty: {penalty}, Loss: {total_loss.item()}")

        return optimized_embedding
    
    def test_image_score(self):

        features_data = get_object(self.minio_client, "propaganda-poster/0242/241057_clip_kandinsky.msgpack")
        features_vector = msgpack.unpackb(features_data)["clip-feature-vector"]
        features_vector= torch.tensor([features_vector]).to(device=self.device, dtype=torch.float32)

        inputs = features_vector.reshape(len(features_vector), -1)
        score = self.scoring_model.model.forward(inputs).squeeze()

        print(f"score is {score}")


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
                                       penalty_weight=args.penalty_weight)
    
    generator.test_image_score()
    #result_latent=generator.generate_latent()

    # if(args.send_job):
    #     try:
    #         response= generate_img2img_generation_jobs_with_kandinsky(
    #             image_embedding=result_latent,
    #             negative_image_embedding=None,
    #             dataset_name="test-generations",
    #             prompt_generation_policy="pez_optimization",
    #         )
    #     except:
    #         print("An error occured.")

if __name__=="__main__":
    main()