import argparse
import os
import random
import sys
from PIL import Image, ImageDraw
import torch

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from stable_diffusion.model.clip_text_embedder.clip_text_embedder import CLIPTextEmbedder
from worker.image_generation.scripts.inpaint_A1111 import get_model, img2img
from scripts.prompt_mutator.greedy_substitution_search_v1 import PromptSubstitutionGenerator
from utility.minio import cmd

OUTPUT_PATH="environmental/output/iterative_painting/result.png"

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
    parser.add_argument('--substitution-model', help="substitution model type: xgboost or linear", default='xgboost')
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
    parser.add_argument('--initial-generation-policy', help="the generation policy used for generating the initial seed prompts", default="independant_approximation")
    parser.add_argument('--top-k', type=float, help="top percentage of prompts taken from generation to be mutated", default=0.1)
    parser.add_argument('--num_choices', type=int, help="Number of substituion choices tested every iteration", default=128)
    parser.add_argument('--clip-batch-size', type=int, help="Batch size for clip embeddings", default=1000)
    parser.add_argument('--substitution-batch-size', type=int, help="Batch size for the substitution model", default=100000)

    return parser.parse_args()

class IterativePainter:
    def __init__(self, prompt_generator):

        self.painted_centers=[]
        self.image= Image.new("RGBA", (1024, 1024))
        self.steps=20

        self.prompt_generator= prompt_generator
        self.minio_client = self.prompt_generator.minio_client
        self.embedder=self.prompt_generator.embedder

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.sd, config, self.model = get_model(self.device, self.steps)

    def check_center_overlap(self, new_center):
        new_cx1, new_cy1, new_cx2, new_cy2 = new_center
        for center in self.painted_centers:
            cx1, cy1, cx2, cy2 = center
            if not (new_cx2 < cx1 or new_cy2 < cy1 or new_cx1 > cx2 or new_cy1 > cy2):
                return True
        return False

    def create_inpainting_mask(self, square_size=512, center_size=128):
        while True:
            square_start_x = random.randint(0, 1024 - square_size)
            square_start_y = random.randint(0, 1024 - square_size)
            center_x = square_start_x + square_size // 2 - center_size // 2
            center_y = square_start_y + square_size // 2 - center_size // 2
            new_center = (center_x, center_y, center_x + center_size, center_y + center_size)
            if not self.check_center_overlap(new_center):
                self.painted_centers.append(new_center)
                break

        mask = Image.new('L', (1024, 1024), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([square_start_x, square_start_y, square_start_x + square_size, square_start_y + square_size], fill=255)
        draw.rectangle(new_center, fill=0)
        return mask
    
    def paint_image(self):
        while(len(self.painted_centers) < 100):
            mask = self.create_inpainting_mask()

            generated_prompt= self.generate_prompt()
            img_byte_arr, generated_image = self.generate_image(generated_prompt, mask)

            # Apply mask to the generated image and then composite it over the existing image
            mask = mask.convert("L")
            generated_image.putalpha(mask)
            self.image.paste(generated_image, (0, 0), generated_image)

        cmd.upload_data(self.minio_client, 'datasets', OUTPUT_PATH , img_byte_arr)

    def generate_prompt(self):
        # generate a prompt
        prompt_list = self.prompt_generator.generate_initial_prompts_with_fixed_probs(1)
        prompt, _= self.mutate_prompts(prompt_list)

        return prompt.positive_prompt
    
    def generate_image(self, generated_prompt, mask):
        init_images = [Image.new("RGBA", (1024, 1024), "white")]
        # Generate the image
        output_file_path, output_file_hash, img_byte_arr, seed, subseed = img2img(
            generated_prompt, '', sampler_name="ddim", batch_size=1, n_iter=1, steps=self.steps, 
            cfg_scale=7.0, width=512, height=512, mask_blur=0, inpainting_fill=0, outpath='output', 
            styles=None, init_images=init_images, mask=mask, resize_mode=0, denoising_strength=0.75, 
            image_cfg_scale=None, inpaint_full_res_padding=0, inpainting_mask_invert=0,
            sd=self.sd, clip_text_embedder=self.embedder, model=self.model, device=self.device)
        
        img_byte_arr.seek(0)  # Reset the buffer
        return img_byte_arr, Image.open(img_byte_arr)
        
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

   prompt_generator= PromptSubstitutionGenerator(minio_access_key=args.minio_access_key,
                                  minio_secret_key=args.minio_secret_key,
                                  minio_ip_addr=args.minio_addr,
                                  csv_phrase=args.csv_phrase,
                                  csv_base_prompts=csv_base_prompts,
                                  model_dataset=args.model_dataset,
                                  substitution_model=args.substitution_model,
                                  scoring_model=args.scoring_model,
                                  max_iterations=args.max_iterations,
                                  sigma_threshold=args.sigma_threshold,
                                  variance_weight=args.variance_weight,
                                  boltzman_temperature=args.boltzman_temperature,
                                  boltzman_k=args.boltzman_k,
                                  dataset_name=args.dataset_name,
                                  store_embeddings=args.store_embeddings,
                                  store_token_lengths=args.store_token_lengths,
                                  self_training=args.self_training,
                                  send_job=args.send_job,
                                  save_csv=args.save_csv,
                                  initial_generation_policy=args.initial_generation_policy,
                                  top_k=args.top_k,
                                  num_choices_per_iteration=args.num_choices,
                                  clip_batch_size=args.clip_batch_size,
                                  substitution_batch_size=args.substitution_batch_size)
   
   Painter= IterativePainter(prompt_generator=prompt_generator)
   Painter.paint_image()

if __name__ == "__main__":
    main()


#     args = parse_args()

#     mask = Image.new('L', (1024, 1024), 0)
#     square_start_x= 128
#     square_start_y= 128
#     square = (square_start_x, square_start_y, square_start_x + 512, square_start_y + 512) 
#     draw = ImageDraw.Draw(mask)
#     draw.rectangle(square, fill=255)
#     center_x = square_start_x + 224  # 224 = 512/2 - 64/2
#     center_y = square_start_y + 224
#     center_size=64
#     center_area = (center_x, center_y, center_x + center_size, center_y + center_size)
#     draw.rectangle(center_area, fill=0)

#     prompt = "A beautiful landscape"  # Example prompt
#     negative_prompt = ""  # Negative prompt (can be an empty string if not used)
#     sampler_name = "ddim"
#     batch_size = 1
#     n_iter = 1
#     steps = 20
#     cfg_scale = 7.0
#     width = 512
#     height = 512
#     mask_blur = 0
#     inpainting_fill = 0
#     outpath = "output"  # Specify the output path
#     styles = None
#     init_images = [Image.new("RGBA", (1024, 1024), "white")]
#     mask = mask
#     resize_mode = 0
#     denoising_strength = 0.75
#     image_cfg_scale = None
#     inpaint_full_res_padding = 0
#     inpainting_mask_invert = 0

#     minio_client = cmd.get_minio_client(
#             minio_access_key=args.minio_access_key,
#             minio_secret_key=args.minio_secret_key,
#             minio_ip_addr=args.minio_addr)
 
#     # Assuming the models are loaded here (sd, clip_text_embedder, model)
#     if torch.cuda.is_available():
#         device = 'cuda'
#     else:
#         device = 'cpu'

#     sd, config, model = get_model(device, steps)

#     # Load the clip embedder model
#     embedder=CLIPTextEmbedder(device=device)
#     embedder.load_submodels()  

#     # Generate the image
#     output_file_path, output_file_hash, img_byte_arr, seed, subseed = img2img(
#         prompt, negative_prompt, sampler_name, batch_size, n_iter, steps, cfg_scale, width, height,
#         mask_blur, inpainting_fill, outpath, styles, init_images, mask, resize_mode,
#         denoising_strength, image_cfg_scale, inpaint_full_res_padding, inpainting_mask_invert,
#         sd=sd, clip_text_embedder=embedder, model=model, device=device)

#     # Display the image
#     img_byte_arr.seek(0)
#     cmd.upload_data(minio_client, 'datasets', OUTPUT_PATH , img_byte_arr) 

# if __name__ == "__main__":
#     main()
