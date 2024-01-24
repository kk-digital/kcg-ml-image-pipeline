import argparse
import io
import os
import random
import sys
from PIL import Image, ImageDraw
import numpy as np
import torch

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from training_worker.ab_ranking.model.ab_ranking_linear import ABRankingModel
from worker.image_generation.scripts.inpainting_pipeline import StableDiffusionInpaintingPipeline
from scripts.prompt_mutator.greedy_substitution_search_v1 import PromptSubstitutionGenerator
from utility.minio import cmd
from utility.clip import clip

OUTPUT_PATH="environmental/output/iterative_painting"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--csv-phrase', help='CSV containing phrases, must have "phrase str" column', default='input/civitai_phrases_database_v7_no_nsfw.csv')
    parser.add_argument('--send-job', action='store_true', default=False)
    parser.add_argument('--update-prompts', action='store_true', default=False)
    parser.add_argument('--dataset-name', default='test-generations')
    parser.add_argument('--model-dataset', default='environmental')
    parser.add_argument('--substitution-model', help="substitution model type: xgboost or linear", default='linear')
    parser.add_argument('--scoring-model', help="elm or linear", default="linear")
    parser.add_argument('--sigma-threshold', type=float, help="threshold of rejection policy for increase of sigma score", default=-0.1)
    parser.add_argument('--variance-weight', type=float, help="weight of variance when optimizing score", default=0)
    parser.add_argument('--boltzman-temperature', type=int, default=11)
    parser.add_argument('--boltzman-k', type=float, default=1.0)
    parser.add_argument('--max-iterations', type=int, help="number of mutation iterations", default=80)
    parser.add_argument('--self-training', action='store_true', default=False)
    parser.add_argument('--store-embeddings', action='store_true', default=False)
    parser.add_argument('--store-token-lengths', action='store_true', default=False)
    parser.add_argument('--save-csv', action='store_true', default=False)
    parser.add_argument('--initial-generation-policy', help="the generation policy used for generating the initial seed prompts", default="fixed_probabilities")
    parser.add_argument('--top-k', type=float, help="top percentage of prompts taken from generation to be mutated", default=0.1)
    parser.add_argument('--num_choices', type=int, help="Number of substituion choices tested every iteration", default=128)
    parser.add_argument('--clip-batch-size', type=int, help="Batch size for clip embeddings", default=1000)
    parser.add_argument('--substitution-batch-size', type=int, help="Batch size for the substitution model", default=100000)

    return parser.parse_args()

class IterativePainter:
    def __init__(self, prompt_generator=None):
        self.max_iterations=100
        self.image_size=1024 
        self.context_size=512 
        self.paint_size=128
        self.start= int((self.context_size - self.paint_size)/2)
        self.end=self.image_size - self.start
        self.painted_areas= int((self.end - self.start) / self.paint_size)
        self.current_score=0
        self.paint_matrix= np.zeros((self.end - self.start, self.end - self.start))
        self.max_repaints=3
        self.painted_centers=[]
        self.image= Image.new("RGB", (1024, 1024), "white")
        self.num_prompts=10

        left = (self.context_size - self.paint_size) // 2
        top = (self.context_size - self.paint_size) // 2
        right = left + self.paint_size
        bottom = top + self.paint_size
        
        self.center_area=(left, top, right, bottom)

        # self.prompt_generator= prompt_generator
        # self.minio_client = self.prompt_generator.minio_client
        # self.text_embedder=self.prompt_generator.embedder

        self.minio_client = cmd.connect_to_minio_client()

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.image_embedder= clip.ClipModel(device=torch.device(self.device))
        self.image_embedder.load_clip()

        # self.scoring_model= self.load_scoring_model()

        # self.inpainting_processor= StableDiffusionProcessingImg2Img(
        #     sampler_name="ddim", 
        #     batch_size=1, 
        #     n_iter=1, 
        #     steps=20, 
        #     cfg_scale=7.0, 
        #     width=self.context_size, 
        #     height=self.context_size, 
        #     mask_blur=4.0, 
        #     inpainting_fill=1, 
        #     styles=None, 
        #     resize_mode=0, 
        #     denoising_strength=0.75, 
        #     image_cfg_scale=None, 
        #     inpaint_full_res_padding=0, 
        #     inpainting_mask_invert=0,
        #     clip_text_embedder=self.text_embedder, 
        #     device=self.device)
        
        # # load stable diffusion model for inpainting
        # self.inpainting_processor.load_model()

        self.pipeline = StableDiffusionInpaintingPipeline()
        self.pipeline.load_models()

    # load elm or linear scoring models
    def load_scoring_model(self):
        input_path=f"{self.prompt_generator.model_dataset}/models/ranking/"

        if(self.prompt_generator.scoring_model=="elm"):
            scoring_model = ABRankingELMModel(768)
            file_name=f"score-elm-v1-clip.safetensors"
        else:
            scoring_model= ABRankingModel(768)
            file_name=f"score-linear-clip.safetensors"

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

    def get_seed_prompts(self):    
        # generate a set number of prompts
        prompt_list = self.prompt_generator.generate_initial_prompts_with_fixed_probs(self.num_prompts)
        prompts_data, _= self.prompt_generator.mutate_prompts(prompt_list)

        sorted_prompts= sorted(prompts_data, key=lambda data: data.positive_score, reverse=True)
        print(sorted_prompts[0].positive_prompt)

        return sorted_prompts[0].positive_prompt

    def paint_image(self):
        prompt= self.get_seed_prompts()

        index=0
        while(True):
            # choose random area to paint in
            x = random.randint(self.start, self.end - self.paint_size)
            y = random.randint(self.start, self.end - self.paint_size)
            
            paint_area = (x, y, x + self.paint_size, y + self.paint_size)
            context_box= self.get_context_area(paint_area)

            context_image= self.image.crop(context_box)
            generated_image= self.generate_image(context_image, prompt)

            # paste generated image in the main image
            self.image.paste(generated_image, paint_area)

            # increment counter for each pixel that was counted
            for i in range(x, x+self.paint_size):
                for j in range(y, y+self.paint_size):
                    row= i - self.start 
                    col= j - self.start
                    self.paint_matrix[row][col]+=1
            
            if index % 100==0:
                # save image state in current step
                img_byte_arr = io.BytesIO()
                self.image.save(img_byte_arr, format="png")
                img_byte_arr.seek(0)  # Move to the start of the byte array

                cmd.upload_data(self.minio_client, 'datasets', OUTPUT_PATH + f"/step_{index}.png" , img_byte_arr)
            
            index+=1
            
            if np.all(self.paint_matrix > self.max_repaints):
                break
 
    def get_context_area(self, paint_area):
        # get surrounding context
        context_x= paint_area[0] - self.context_size // 2 + self.paint_size // 2
        context_y= paint_area[1] - self.context_size // 2 + self.paint_size // 2
        
        return (context_x, context_y, context_x + self.context_size, context_y + self.context_size)

    def generate_prompt(self):
        # generate a prompt
        prompt_list = self.prompt_generator.generate_initial_prompts_with_fixed_probs(1)
        prompt, _= self.prompt_generator.mutate_prompts(prompt_list)

        prompt_str= prompt[0].positive_prompt
        print(prompt_str)

        return prompt_str
    
    # def generate_image(self, context_image, prompt):
    #     # Use the context image as an initial image
    #     draw = ImageDraw.Draw(context_image)
    #     draw.rectangle(self.center_area, fill="white")  # Unmasked (white) center area
    #     init_images = [context_image]

    #     # Create mask
    #     mask = Image.new("L", (self.context_size, self.context_size), 0)  # Fully masked (black)
    #     draw = ImageDraw.Draw(mask)
    #     draw.rectangle(self.center_area, fill=255)  # Unmasked (white) center area

    #     # Generate the image
    #     image, seed = self.inpainting_processor.img2img(prompt=prompt, negative_prompt="", init_images=init_images, image_mask=mask)

    #     cropped_image = image.crop(self.center_area)

    #     return cropped_image

    def generate_image(self, context_image, prompt):
        # Use the context image as an initial image
        draw = ImageDraw.Draw(context_image)
        draw.rectangle(self.center_area, fill="white")  # Unmasked (white) center area

        # Create mask
        mask = Image.new("L", (self.context_size, self.context_size), 0)  # Fully masked (black)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(self.center_area, fill=255)  # Unmasked (white) center area

        result_image= self.pipeline.inpaint(prompt=prompt, image=context_image, image_mask= mask)
        
        cropped_image = result_image.crop(self.center_area)

        return cropped_image

    def test(self):
        prompt="2D side scrolling, forest, electric atmosphere, mechanical ascension cyberpunk, tropical jungle theme, undewear, ruined walls, beastly, ruined cityscape, mechanical, ruins in a jungle, deep jungle, showchest, bad chest, adventurer, a ruin"
        white_background= Image.new("RGB", (512, 512), "white")
        mask= Image.new("L", (512, 512), 255)

        # Generate the image
        init_image = self.pipeline.inpaint(prompt=prompt, image=white_background, image_mask= mask)

        draw = ImageDraw.Draw(init_image)
        draw.rectangle(self.center_area, fill="white")  # Unmasked (white) center area
        
        img_byte_arr = io.BytesIO()
        init_image.save(img_byte_arr, format="png")
        img_byte_arr.seek(0)  # Move to the start of the byte array
        cmd.upload_data(self.minio_client, 'datasets', OUTPUT_PATH + f"/initial_image.png" , img_byte_arr) 

        generated_image= self.generate_image(init_image, prompt)
        
        img_byte_arr = io.BytesIO()
        generated_image.save(img_byte_arr, format="png")
        img_byte_arr.seek(0)  # Move to the start of the byte array

        cmd.upload_data(self.minio_client, 'datasets', OUTPUT_PATH + f"/test.png" , img_byte_arr)


def main():
   args = parse_args()

   # set the base prompts csv path
   if(args.model_dataset=="icons"):
        csv_base_prompts='input/dataset-config/icon/base-prompts-dsp.csv'
   elif(args.model_dataset=="propaganda-poster"):
        csv_base_prompts='input/dataset-config/propaganda-poster/base-prompts-propaganda-poster.csv'
   elif(args.model_dataset=="mech"):
        csv_base_prompts='input/dataset-config/mech/base-prompts-dsp.csv'
   elif(args.model_dataset=="character" or args.model_dataset=="waifu"):
        csv_base_prompts='input/dataset-config/character/base-prompts-waifu.csv'
   elif(args.model_dataset=="environmental"):  
        csv_base_prompts='input/dataset-config/environmental/base-prompts-environmental.csv'

#    prompt_generator= PromptSubstitutionGenerator(minio_access_key=args.minio_access_key,
#                                   minio_secret_key=args.minio_secret_key,
#                                   minio_ip_addr=args.minio_addr,
#                                   csv_phrase=args.csv_phrase,
#                                   csv_base_prompts=csv_base_prompts,
#                                   model_dataset=args.model_dataset,
#                                   substitution_model=args.substitution_model,
#                                   scoring_model=args.scoring_model,
#                                   max_iterations=args.max_iterations,
#                                   sigma_threshold=args.sigma_threshold,
#                                   variance_weight=args.variance_weight,
#                                   boltzman_temperature=args.boltzman_temperature,
#                                   boltzman_k=args.boltzman_k,
#                                   dataset_name=args.dataset_name,
#                                   store_embeddings=args.store_embeddings,
#                                   store_token_lengths=args.store_token_lengths,
#                                   self_training=args.self_training,
#                                   send_job=args.send_job,
#                                   save_csv=args.save_csv,
#                                   initial_generation_policy=args.initial_generation_policy,
#                                   top_k=args.top_k,
#                                   num_choices_per_iteration=args.num_choices,
#                                   clip_batch_size=args.clip_batch_size,
#                                   substitution_batch_size=args.substitution_batch_size)
   
   Painter= IterativePainter()
   Painter.test()

if __name__ == "__main__":
    main()
