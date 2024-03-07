import argparse
from datetime import datetime
import io
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import msgpack
from PIL import Image


base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from kandinsky.model_paths import PRIOR_MODEL_PATH
from transformers import CLIPImageProcessor
from kandinsky_worker.image_generation.img2img_generator import generate_img2img_generation_jobs_with_kandinsky
from training_worker.scoring.models.scoring_fc import ScoringFCNetwork
from utility.minio import cmd
from data_loader.utils import get_object

GENERATION_POLICY= "gradient_descent_optimization"

def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--minio-access-key', type=str, help='Minio access key')
        parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
        parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")
        parser.add_argument('--steps', type=int, help='number of optimisation steps', default=100)
        parser.add_argument('--learning-rate', type=float, help='learning rate for optimization', default=0.001)
        parser.add_argument('--target-score', type=float, help='number of optimisation steps', default=5)
        parser.add_argument('--send-job', action='store_true', default=False)
        parser.add_argument('--save-csv', action='store_true', default=False)
        parser.add_argument('--generate-step', type=int, default=100)
        parser.add_argument('--print-step', type=int, default=10)

        return parser.parse_args()

class KandinskyImageGenerator:
    def __init__(self,
                 minio_access_key,
                 minio_secret_key,
                 dataset,
                 steps=100,
                 learning_rate=0.005,
                 target_score=5.0,
                 send_job=False,
                 save_csv=False,
                 generate_step=100,
                 print_step=10
                 ):
        
        self.dataset= dataset
        self.steps= steps
        self.learning_rate= learning_rate
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
        
        # load kandinsky clip
        self.image_processor= CLIPImageProcessor.from_pretrained(PRIOR_MODEL_PATH, subfolder="image_processor", local_files_only=True)

        self.scoring_model= ScoringFCNetwork(minio_client=self.minio_client)
        self.scoring_model.load_model()

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
    
    def sample_embedding(self, num_samples=1000):
        sampled_embeddings = torch.normal(mean=self.clip_mean.repeat(num_samples, 1),
                                      std=self.clip_std.repeat(num_samples, 1))
        
        # Score each sampled embedding
        scores=[]
        embeddings=[]
        for embed in sampled_embeddings:
            embeddings.append(embed.unsqueeze(0))
            score = self.scoring_model.model(embed.unsqueeze(0)).item() 
            scores.append(score)
        
        # Find the index of the highest scoring embedding
        highest_score_index = np.argmax(scores)
        
        # Select the highest scoring embedding
        highest_scoring_embedding = embeddings[highest_score_index]
        
        return highest_scoring_embedding.to(device=self.device)
    
    
    def get_image_features(self, image):
        # Preprocess image
        if isinstance(image, Image.Image):
            image = self.image_processor(image, return_tensors="pt")['pixel_values']
        
         # Compute CLIP features
        if isinstance(image, torch.Tensor):
            features = self.image_encoder(pixel_values= image.half().to(self.device)).image_embeds
        else:
            raise ValueError(
                f"`image` can only contains elements to be of type `PIL.Image.Image` or `torch.Tensor`  but is {type(image)}"
            )
        
        return features

    def generate_latent(self):
        df_data=[]
        sampled_embedding= self.sample_embedding()
        optimized_embedding = sampled_embedding.clone().detach().requires_grad_(True)

        # Setup the optimizer
        optimizer = optim.Adam([optimized_embedding], lr=self.learning_rate)

        for step in range(self.steps):
            optimizer.zero_grad()

            score= self.scoring_model.model(optimized_embedding)

            score_loss =  self.target_score - score
            
            # Total loss
            total_loss = score_loss

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
                print(f"Step: {step}, Score: {score.item()}, Loss: {total_loss.item()}")
        
        if self.send_job:
            try:
                response= generate_img2img_generation_jobs_with_kandinsky(
                    image_embedding=optimized_embedding,
                    negative_image_embedding=None,
                    dataset_name="environmental",
                    prompt_generation_policy=GENERATION_POLICY,
                    self_training=True
                )

                task_uuid = response['uuid']
                task_time = response['creation_time']
            except:
                print("An error occured.")
                task_uuid = -1
                task_time = -1

        if self.save_csv:
            self.store_uuids_in_csv_file(df_data)

        return optimized_embedding
    
    # store list of initial prompts in a csv to use for prompt mutation
    def store_uuids_in_csv_file(self, data):
        minio_path=f"environmental/output/generated-images-csv"
        local_path="output/generated_images.csv"
        pd.DataFrame(data).to_csv(local_path, index=False)
        # Read the contents of the CSV file
        with open(local_path, 'rb') as file:
            csv_content = file.read()

        #Upload the CSV file to Minio
        buffer = io.BytesIO(csv_content)
        buffer.seek(0)

        current_date=datetime.now().strftime("%Y-%m-%d-%H:%M")
        minio_path= minio_path + f"/{current_date}-gradient_descent_optimization-environmental.csv"
        cmd.upload_data(self.minio_client, 'datasets', minio_path, buffer)
        # Remove the temporary file
        os.remove(local_path)

def main():
    args= parse_args()
    # initialize generator
    generator= KandinskyImageGenerator(minio_access_key=args.minio_access_key,
                                       minio_secret_key=args.minio_secret_key,
                                       dataset=args.dataset,
                                       steps=args.steps,
                                       learning_rate= args.learning_rate,
                                       target_score=args.target_score,
                                       send_job= args.send_job,
                                       save_csv= args.save_csv,
                                       generate_step=args.generate_step,
                                       print_step=args.print_step)
    
    generator.generate_latent()

if __name__=="__main__":
    main()