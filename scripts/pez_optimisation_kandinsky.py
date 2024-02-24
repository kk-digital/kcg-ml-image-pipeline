import argparse
import io
import os
import sys
import torch
import torch.optim as optim

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from training_worker.ab_ranking.model.ab_ranking_linear import ABRankingModel
from kandinsky_worker.image_generation.img2img_generator import generate_img2img_generation_jobs_with_kandinsky
from kandinsky.models.clip_image_encoder.clip_image_encoder import KandinskyCLIPImageEncoder
from utility.minio import cmd

def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--minio-access-key', type=str, help='Minio access key')
        parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
        parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")
        parser.add_argument('--model-type', type=str, help='model type, linear or elm', default="linear")
        parser.add_argument('--steps', type=int, help='number of optimisation steps', default=100)

        return parser.parse_args()

class KandinskyImageGenerator:
    def __init__(self,
                 minio_access_key,
                 minio_secret_key,
                 dataset,
                 model_type,
                 steps=100,
                 target_score=5.0
                 ):
        
        self.dataset= dataset
        self.model_type= model_type
        self.steps= steps

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

    def generate_latent(self):
        # Ensure image embeddings require gradients
        image_embedding= self.get_zero_embed()
        optimized_embedding = image_embedding.clone().detach().requires_grad_(True)

        # Setup the optimizer
        optimizer = optim.Adam([optimized_embedding], lr=0.01)

        for step in range(self.steps):
            optimizer.zero_grad()

            # Calculate the custom score
            inputs = optimized_embedding.reshape(len(optimized_embedding), -1)
            score = self.scoring_model.model.forward(inputs).squeeze()

            # Custom loss function
            loss = self.target_score - score

            # Backpropagate
            loss.backward()
            optimizer.step()

            print(f"Step: {step}, Score: {score.item()}, Loss: {loss.item()}")

        return optimized_embedding

def main():
    args= parse_args()
    # initialize generator
    generator= KandinskyImageGenerator(minio_access_key=args.minio_access_key,
                                       minio_secret_key=args.minio_secret_key,
                                       dataset=args.dataset,
                                       model_type=args.model_type,
                                       steps=args.steps)
    
    result_latent=generator.generate_latent()

    print(result_latent)

    # try:
    #     response= generate_img2img_generation_jobs_with_kandinsky(
    #         image_embedding=result_latent,
    #         negative_image_embedding=None,
    #         dataset_name="test-generations",
    #         prompt_generation_policy="pez_optimization",
    #     )
    # except:
    #     print("An error occured.")

if __name__=="__main__":
    main()