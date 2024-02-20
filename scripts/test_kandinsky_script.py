import argparse
import io
import json
import os
import sys
import traceback
import pandas as pd
import requests
from tqdm import tqdm
from PIL import Image

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.minio import cmd
from kandinsky.models.clip_image_encoder.clip_image_encoder import KandinskyCLIPImageEncoder
from kandinsky_worker.image_generation.img2img_generator import generate_img2img_generation_jobs_with_kandinsky

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--num-variants', type=int, required=False, default=1, help='Number of variants for each image')
    return parser.parse_args()

def get_image_paths(images_dir):
    # List of file extensions to consider as images
    image_paths = []

    # Walk through directory and subdirectories
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith('jpeg'):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)

    return image_paths

# store generated prompts in a csv file
def store_csv_file(minio_client, data):
    local_path="output/generated_images.csv"
    pd.DataFrame(data).to_csv(local_path, index=False)
    # Read the contents of the CSV file
    with open(local_path, 'rb') as file:
        csv_content = file.read()

    #Upload the CSV file to Minio
    buffer = io.BytesIO(csv_content)
    buffer.seek(0)

    minio_path=f"environmental/output/synth-boards-pinterest-dataset/character_images.csv"
    cmd.upload_data(minio_client, 'datasets', minio_path, buffer)
    # Remove the temporary file
    os.remove(local_path)

def main():
    args= parse_args()

    # get minio client
    minio_client = cmd.get_minio_client(args.minio_access_key,
                                        args.minio_secret_key,
                                        args.minio_addr)
    
    image_embedder= KandinskyCLIPImageEncoder(device="cuda")
    image_embedder.load_submodels()

    image_paths= get_image_paths()
    images=[]

    for index, path in enumerate(image_paths):
        image= Image.open(path)
        image_embedding=image_embedder.get_image_features(image)
 
        for i in range(args.num_variants):
            try:
                response = generate_img2img_generation_jobs_with_kandinsky(
                            image_embedding= image_embedding,
                            negative_image_embedding=None,
                            dataset_name="test-generations",
                            prompt_generation_policy="img2img_kandinsky",
                            minio_client= minio_client)
                task_uuid = response['uuid']
            except:
                print('Error occured:')
                print(traceback.format_exc())
                task_uuid = -1

            data={
                'task_uuid': task_uuid,
                'image_id': index
            }

            # append prompt data
            images.append(data)
        
    # Output results to CSV files
    images_df = pd.DataFrame(images)
    store_csv_file(minio_client, images_df)
    print("Processing complete!")