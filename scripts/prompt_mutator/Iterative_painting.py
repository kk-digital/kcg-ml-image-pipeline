import io
import os
import sys
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from stable_diffusion.model.clip_text_embedder.clip_text_embedder import CLIPTextEmbedder
from worker.image_generation.scripts.inpaint_A1111 import get_model, img2img

def display_image(image_byte_array):
    """Display image from a bytes array."""
    image = Image.open(io.BytesIO(image_byte_array))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def main():
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
    steps = 50
    cfg_scale = 7.0
    width = 512
    height = 512
    mask_blur = 4
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

    # Assuming the models are loaded here (sd, clip_text_embedder, model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    display_image(img_byte_arr.getvalue())

if __name__ == "__main__":
    main()
