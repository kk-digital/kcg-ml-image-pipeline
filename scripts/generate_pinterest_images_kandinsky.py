import argparse
import io
import json
import os
import sys
import traceback
import pandas as pd
from tqdm import tqdm

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.minio import cmd
from optim_utils import download_image
from kandinsky.models.clip_image_encoder.clip_image_encoder import KandinskyCLIPImageEncoder
from kandinsky_worker.image_generation.img2img_generator import generate_img2img_generation_jobs_with_kandinsky


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    return parser.parse_args()

# store generated prompts in a csv file
def store_prompts_in_csv_file(minio_client, data):

    local_path="output/generated_images.csv"
    pd.DataFrame(data).to_csv(local_path, index=False)
    # Read the contents of the CSV file
    with open(local_path, 'rb') as file:
        csv_content = file.read()

    #Upload the CSV file to Minio
    buffer = io.BytesIO(csv_content)
    buffer.seek(0)

    minio_path=f"environmental/output/synth-boards-pinterest-dataset/generated_images.csv"
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

    # Process each line in the JSONL file
    prompts = []
    with open("input/data.jsonl", 'r') as file:
        for line in tqdm(file):
            # Parse JSON line
            data = json.loads(line)
            board_title = data['board_title']

            if board_title != "kcg-environments":
                continue

            image_url = data['image_urls'][0]

            try:
                # Download and process the image
                image = download_image(image_url)
            except Exception as e:
                print(f"Error processing image {image_url}: {e}")
                continue

            image_embedding= image_embedder.get_image_features(image)

            try:
                response = generate_img2img_generation_jobs_with_kandinsky(
                            image_embedding= image_embedding,
                            negative_image_embedding=None,
                            dataset_name="test-generations",
                            prompt_generation_policy="img2img_kandinsky",
                            minio_client= minio_client)
                task_uuid = response['uuid']
                file_path = response['file_path']
            except:
                print('Error occured:')
                print(traceback.format_exc())
                task_uuid = -1

            prompt_data={
                'task_uuid': task_uuid,
                'file_path': file_path,
                'image_url': image_url, 
                'board': board_title
            }

            # append prompt data
            prompts.append(prompt_data)
    
    
    # Output results to CSV files
    prompts_df = pd.DataFrame(prompts)
    store_prompts_in_csv_file(minio_client, prompts_df)
    print("Processing complete!")

if __name__=="__main__":
    main()