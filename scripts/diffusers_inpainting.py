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
parser.add_argument('--local_files_only', action='store_true', help='Use local files only.')
parser.add_argument('--cache_dir', type=str, default=None, help='Cache directory for the model.')
parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-inpainting', help='Model ID for the pipeline.')
parser.add_argument('--use_float32', action='store_true', help='Use float32 precision.')

# get args
args = parser.parse_args()

# set CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)

from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
import torch

# load image & mask
init_image = Image.open(args.image_path).convert("RGB").resize(args.target_size)
mask = Image.open(args.mask_path).convert("RGB").resize(args.target_size)

# load model
torch_dtype = torch.float32 if args.use_float32 else torch.float16
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    args.model_id, 
    torch_dtype=torch_dtype, 
    local_files_only=args.local_files_only, 
    cache_dir=args.cache_dir
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
