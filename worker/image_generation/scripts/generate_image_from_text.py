import os
import sys

from datetime import datetime

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())


from stable_diffusion.utils_image import save_images_to_minio, save_image_data_to_minio, save_image_embedding_to_minio, get_image_data


def generate_image_from_text(minio_client, txt2img, clip_text_embedder, job_uuid, dataset, sampler, sampler_steps,
                             positive_prompts, negative_prompts, cfg_strength, seed, image_width, image_height, output_path):
    embedded_prompts = clip_text_embedder(positive_prompts)
    negative_embedded_prompts = clip_text_embedder(negative_prompts)

    prompt_scoring_model = 'N/A'
    prompt_score = 'N/A'
    prompt_generation_policy = 'N/A'
    top_k = 0

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
    output_file_path = output_path
    output_file_hash, img_data = get_image_data(images)

    return output_file_path, output_file_hash, img_data

