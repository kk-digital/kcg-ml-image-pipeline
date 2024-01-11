import argparse
import os
import sys
from PIL import Image, ImageDraw
import torch

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from stable_diffusion.model.clip_text_embedder.clip_text_embedder import CLIPTextEmbedder
from worker.image_generation.scripts.inpaint_A1111 import get_model, img2img
from utility.minio import cmd

OUTPUT_PATH="environmental/output/iterative_painting/result.png"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')

    return parser.parse_args()

def main():
    args = parse_args()

    mask = Image.new('L', (1024, 1024), 0)
    square_start_x= 128
    square_start_y= 128
    square = (square_start_x, square_start_y, square_start_x + 512, square_start_y + 512) 
    draw = ImageDraw.Draw(mask)
    draw.rectangle(square, fill=255)
    center_x = square_start_x + 224  # 224 = 512/2 - 64/2
    center_y = square_start_y + 224
    center_size=64
    center_area = (center_x, center_y, center_x + center_size, center_y + center_size)
    draw.rectangle(center_area, fill=0)

    prompt = "A beautiful landscape"  # Example prompt
    negative_prompt = ""  # Negative prompt (can be an empty string if not used)
    sampler_name = "ddim"
    batch_size = 1
    n_iter = 1
    steps = 20
    cfg_scale = 7.0
    width = 512
    height = 512
    mask_blur = 0
    inpainting_fill = 0
    outpath = "output"  # Specify the output path
    styles = None
    init_images = [Image.new("RGBA", (1024, 1024), "white")]
    mask = mask
    resize_mode = 0
    denoising_strength = 0.75
    image_cfg_scale = None
    inpaint_full_res_padding = 0
    inpainting_mask_invert = 0

    minio_client = cmd.get_minio_client(
            minio_access_key=args.minio_access_key,
            minio_secret_key=args.minio_secret_key,
            minio_ip_addr=args.minio_addr)
 
    # Assuming the models are loaded here (sd, clip_text_embedder, model)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    sd, config, model = get_model(device, steps)

    # Load the clip embedder model
    embedder=CLIPTextEmbedder(device=device)
    embedder.load_submodels()  

    # Generate the image
    output_file_path, output_file_hash, img_byte_arr, seed, subseed = img2img(
        prompt, negative_prompt, sampler_name, batch_size, n_iter, steps, cfg_scale, width, height,
        mask_blur, inpainting_fill, outpath, styles, init_images, mask, resize_mode,
        denoising_strength, image_cfg_scale, inpaint_full_res_padding, inpainting_mask_invert,
        sd=sd, clip_text_embedder=embedder, model=model, device=device)

    # Display the image
    img_byte_arr.seek(0)
    cmd.upload_data(minio_client, 'datasets', OUTPUT_PATH , img_byte_arr) 

if __name__ == "__main__":
    main()
