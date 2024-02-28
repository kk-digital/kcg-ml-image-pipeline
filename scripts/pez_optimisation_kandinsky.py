import argparse
from datetime import datetime
import io
import os
import sys
import pandas as pd
import torch
import torch.optim as optim
import msgpack
from PIL import Image
import time
import random


base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from training_worker.ab_ranking.model.ab_ranking_linear import ABRankingModel
from kandinsky_worker.image_generation.img2img_generator import generate_img2img_generation_jobs_with_kandinsky
from kandinsky.models.clip_image_encoder.clip_image_encoder import KandinskyCLIPImageEncoder
from kandinsky.models.kandisky import KandinskyPipeline
from utility.minio import cmd
from kandinsky.model_paths import DECODER_MODEL_PATH
from data_loader.utils import get_object

def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--minio-access-key', type=str, help='Minio access key')
        parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
        parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")
        parser.add_argument('--model-type', type=str, help='model type, linear or elm', default="linear")
        parser.add_argument('--steps', type=int, help='number of optimisation steps', default=100)
        parser.add_argument('--learning-rate', type=float, help='learning rate for optimization', default=0.001)
        parser.add_argument('--target-score', type=float, help='number of optimisation steps', default=5)
        parser.add_argument('--penalty-weight', type=float, help='weight of deviation panalty', default=1)
        parser.add_argument('--deviation-threshold', type=float, help='deviation penalty threshold', default=2)
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
                 model_type,
                 steps=100,
                 learning_rate=0.005,
                 target_score=5.0,
                 penalty_weight=1,
                 deviation_threshold= 2,
                 send_job=False,
                 save_csv=False,
                 generate_step=100,
                 print_step=10
                 ):
        
        self.dataset= dataset
        self.model_type= model_type
        self.steps= steps
        self.learning_rate= learning_rate
        self.penalty_weight= penalty_weight
        self.deviation_threshold= deviation_threshold
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
        self.clip= KandinskyCLIPImageEncoder(device= self.device)
        self.clip.load_submodels()

        # load kandinsky's autoencoder
        self.image_generator= KandinskyPipeline(device= self.device, strength=0.75, decoder_guidance_scale=12,
                                                decoder_steps=10)
        self.image_generator.load_models(task_type="img2img")

        # load scoring model
        self.scoring_model= self.load_scoring_model()
        self.mean= float(self.scoring_model.mean)
        self.std= float(self.scoring_model.standard_deviation)

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
    
    def get_image_features(self, image):
        # Preprocess image
        if isinstance(image, Image.Image):
            image = self.clip.image_processor(image, return_tensors="pt")['pixel_values']
        
         # Compute CLIP features
        if isinstance(image, torch.Tensor):
            features = self.clip.vision_model(pixel_values= image.to(self.device).half()).image_embeds
        else:
            raise ValueError(
                f"`image` can only contains elements to be of type `PIL.Image.Image` or `torch.Tensor`  but is {type(image)}"
            )
        
        return features.to(torch.float16)

    def generate_latent(self):
        # features_data = get_object(self.minio_client, "environmental/0435/434997_clip_kandinsky.msgpack")
        # features_vector = msgpack.unpackb(features_data)["clip-feature-vector"]
        # image_embedding= torch.tensor(features_vector).to(device=self.device, dtype=torch.float32)
        
        random.seed(time.time())
        seed = random.randint(0, 2 ** 24 - 1)

        init_image = Image.open("./test/test_inpainting/white_512x512.jpg")
        df_data=[]
        sampled_embedding= self.sample_embedding()
        optimized_embedding = sampled_embedding.clone().detach().requires_grad_(True)

        # Setup the optimizer
        optimizer = optim.Adam([optimized_embedding], lr=self.learning_rate)

        for step in range(self.steps):
            optimizer.zero_grad()

            init_image, latent= self.image_generator.generate_img2img(init_img=init_image,
                                                  image_embeds= optimized_embedding,
                                                  seed=seed
                                                  )
            
            clip_vector= self.get_image_features(init_image).to(dtype=optimized_embedding.dtype)

            # Calculate the custom score
            inputs = clip_vector.reshape(len(clip_vector), -1)
            score = self.scoring_model.model.forward(inputs).squeeze()
            sigma_score= (score - self.mean) / self.std
            # Custom loss function
            # Original loss based on the scoring function
            reg_loss = torch.mean((clip_vector - optimized_embedding) ** 2)
            score_loss =  self.target_score - score
            
            # Total loss
            total_loss = score_loss + self.penalty_weight * reg_loss

            if self.send_job and (step % self.generate_step == 0):
                try:
                    response= generate_img2img_generation_jobs_with_kandinsky(
                        image_embedding=optimized_embedding,
                        negative_image_embedding=None,
                        dataset_name="test-generations",
                        prompt_generation_policy="pez_optimization",
                    )
                    task_uuid = response['uuid']
                    task_time = response['creation_time']
                except:
                    print("An error occured.")
                    task_uuid = -1
                    task_time = -1
                  
                # storing job data to put in csv file later
                df_data.append({
                    'task_uuid': task_uuid,
                    'score': score.item(),
                    'sigma_score': sigma_score.item(),
                    'step': step,
                    'generation_policy_string': "pez_optimization",
                    'time': task_time
                })

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
                print(f"Step: {step}, Score: {score.item()}, Penalty: {reg_loss.item()}, Loss: {total_loss.item()}")
        
        if self.send_job:
            try:
                response= generate_img2img_generation_jobs_with_kandinsky(
                    image_embedding=optimized_embedding,
                    negative_image_embedding=None,
                    dataset_name="test-generations",
                    prompt_generation_policy="pez_optimization",
                )

                task_uuid = response['uuid']
                task_time = response['creation_time']
            except:
                print("An error occured.")
                task_uuid = -1
                task_time = -1
        
            df_data.append({
                'task_uuid': task_uuid,
                'score': score,
                'sigma_score': sigma_score,
                'step': step,
                'generation_policy_string': "pez_optimization",
                'time': task_time
            })

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
        minio_path= minio_path + f"/{current_date}-pez_optimization-environmental.csv"
        cmd.upload_data(self.minio_client, 'datasets', minio_path, buffer)
        # Remove the temporary file
        os.remove(local_path)

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
                                       learning_rate= args.learning_rate,
                                       target_score=args.target_score,
                                       deviation_threshold=args.deviation_threshold,
                                       penalty_weight=args.penalty_weight,
                                       send_job= args.send_job,
                                       save_csv= args.save_csv,
                                       generate_step=args.generate_step,
                                       print_step=args.print_step)
    
    generator.generate_latent()

if __name__=="__main__":
    main()