import os
import sys

from datetime import datetime

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())


from stable_diffusion.utils_image import save_images


def generate_image_from_text(txt2img, clip_text_embedder, positive_prompts, negative_prompts, cfg_strength, seed, image_width, image_height, output_directory):


    embedded_prompts = clip_text_embedder(positive_prompts)
    negative_embedded_prompts = clip_text_embedder(negative_prompts)

    latent = txt2img.generate_images_latent_from_embeddings(
        batch_size=1,
        embedded_prompt=embedded_prompts,
        null_prompt=negative_embedded_prompts,
        uncond_scale=cfg_strength,
        seed=seed,
        w=image_width,
        h=image_height
    )

    images = txt2img.get_image_from_latent(latent)
    output_file_path = output_directory + '/image-' + datetime.now().strftime('%d-%m-%Y-%H-%M-%S') + '.jpg'
    save_images(images, output_file_path)

    return output_file_path

