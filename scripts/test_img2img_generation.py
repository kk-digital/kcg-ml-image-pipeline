from PIL import Image
import sys
import os

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from kandinsky.models.clip_image_encoder.clip_image_encoder import KandinskyCLIPImageEncoder
from kandinsky_worker.image_generation.img2img_generator import generate_img2img_generation_jobs_with_kandinsky

image_embedder= KandinskyCLIPImageEncoder(device="cuda")
image_embedder.load_submodels()

image= Image.open("input/test_image.jpg")

image_embedding= image_embedder.get_image_features(image)

generate_img2img_generation_jobs_with_kandinsky(image_embedding= image_embedding,
                                                negative_image_embedding=None,
                                                dataset_name="test-generations")


