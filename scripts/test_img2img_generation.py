import argparse
from PIL import Image
import sys
import os

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from kandinsky.models.clip_image_encoder.clip_image_encoder import KandinskyCLIPImageEncoder
from kandinsky_worker.image_generation.img2img_generator import generate_img2img_generation_jobs_with_kandinsky
from utility.minio.cmd import get_minio_client


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--minio-access-key", type=str,
                        help="The minio access key to use so worker can upload files to minio server")
    parser.add_argument("--minio-secret-key", type=str,
                        help="The minio secret key to use so worker can upload files to minio server")

    return parser.parse_args()

def main():
    args = parse_args()

    image_embedder= KandinskyCLIPImageEncoder(device="cuda")
    image_embedder.load_submodels()

    image= Image.open("input/test_image.jpg")

    image_embedding= image_embedder.get_image_features(image)

    minio_client= get_minio_client(args.minio_access_key, args.minio_secret_key)

    generate_img2img_generation_jobs_with_kandinsky(image_embedding= image_embedding,
                                                    negative_image_embedding=None,
                                                    dataset_name="test-generations",
                                                    prompt_generation_policy="img2img_kandinsky",
                                                    minio_client= minio_client)
    
if __name__ == '__main__':
    main()


