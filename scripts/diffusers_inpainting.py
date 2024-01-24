import os
import argparse

# set args
parser = argparse.ArgumentParser(description="Stable Diffusion Inpainting with command line arguments.")
parser.add_argument('--prompt', type=str, required=True, help='Text prompt for the model.')
parser.add_argument('--image_path', type=str, required=True, help='Path to the initial image.')
parser.add_argument('--mask_path', type=str, required=True, help='Path to the mask image.')
parser.add_argument('--target_path', type=str, required=True, help='Path to save the output image.')
parser.add_argument('--target_size', type=int, nargs=2, default=(512, 512), help='Size to resize images (width, height).')
parser.add_argument('--device_id', type=int, default=0, help='Device ID for CUDA (e.g., 0).')
parser.add_argument('--num_inference_steps', type=int, default=40, help='Number of inference steps.')
parser.add_argument('--strength', type=float, default=0.75, help='Strength of the inpainting.')
parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale for the model.')
parser.add_argument('--use_float32', action='store_true', help='Use float32 precision.')
parser.add_argument('--model_file_path', type=str, required=True, help='Path to the model file.')
parser.add_argument('--config_file_path', type=str, required=True, help='Path to the config file.')
parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer.')
parser.add_argument('--text_model_path', type=str, required=True, help='Path to the text model.')


# get args
args = parser.parse_args()

# set CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)

from diffusers import StableDiffusionInpaintPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np
import torch

# load image & mask
init_image = Image.open(args.image_path).convert("RGB").resize(args.target_size)
mask = Image.open(args.mask_path).convert("RGB").resize(args.target_size)

# Load tokenizer and text encoder
tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
text_encoder = CLIPTextModel.from_pretrained(args.text_model_path, local_files_only=True).cuda().eval()

# load model
torch_dtype = torch.float32 if args.use_float32 else torch.float16
pipe = StableDiffusionInpaintPipeline.from_single_file(
    pretrained_model_link_or_path=args.model_file_path,
    config_files={'v1': args.config_file_path},
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    local_files_only=True, use_safetensors=True, load_safety_checker=False
).to('cuda')

# do inpainting
with torch.no_grad():
    output = pipe(
        prompt=args.prompt, 
        image=init_image, 
        mask_image=mask, 
        num_inference_steps=args.num_inference_steps, 
        strength=args.strength, 
        guidance_scale=args.guidance_scale
    )

# save result
output_image = output.images[0]
output_image.save(args.target_path)
