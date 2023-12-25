import argparse
import io
import json
import os
import sys
import requests
from PIL import Image
from io import BytesIO
import open_clip
import pandas as pd
import torch
from tqdm import tqdm

base_directory = "./"
sys.path.insert(0, base_directory)

from stable_diffusion.model.clip_text_embedder.clip_text_embedder import CLIPTextEmbedder
from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from training_worker.ab_ranking.model.ab_ranking_linear import ABRankingModel
from utility.hard_prompts_made_easy.optim_utils import *
from utility.minio import cmd

def parse_args():
    parser = argparse.ArgumentParser()

    # Model config
    parser.add_argument('--prompt-len', type=int, default=77, help='Length of the prompt')
    parser.add_argument('--iter', type=int, default=3000, help='Number of iterations')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--prompt-bs', type=int, default=1, help='Prompt batch size')
    parser.add_argument('--loss-weight', type=float, default=1.0, help='Loss weight')
    parser.add_argument('--print-step', type=int, default=100, help='Print step')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--clip-model', default="ViT-L-14", help='CLIP model')
    parser.add_argument('--clip-pretrain', default="openai", help='CLIP pretraining')

    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--input-file', help='JSON file containing all images from pinterest board', default='input/synth-boards-pinterest-dataset.jsonl')
    parser.add_argument('--output-path', help='Folder where csv files containing prompts are stored', default='output/synth-boards-pinterest-dataset')
    parser.add_argument('--config-path', help='Configuration for prompt inversion', default='utility/hard_prompts_made_easy/config.json')
    return parser.parse_args()

# load elm or linear scoring models
def load_model(minio_client, device, embedding_type="positive", scoring_model="linear", input_size=768):
    input_path="environmental/models/ranking/"

    if(scoring_model=="elm"):
        embedding_model = ABRankingELMModel(input_size)
        file_name=f"score-elm-v1-embedding"
    else:
        embedding_model= ABRankingModel(input_size)
        file_name=f"score-linear-embedding"
    
    if(embedding_type=="positive" or embedding_type=="negative"):
        file_name+=f"-{embedding_type}.safetensors"
    else:
        file_name+=".safetensors"

    model_files=cmd.get_list_of_objects_with_prefix(minio_client, 'datasets', input_path)
    most_recent_model = None

    for model_file in model_files:
        if model_file.endswith(file_name):
            most_recent_model = model_file

    if most_recent_model:
        model_file_data =cmd.get_file_from_minio(minio_client, 'datasets', most_recent_model)
    else:
        print("No .safetensors files found in the list.")
        return
    
    print(most_recent_model)

    # Create a BytesIO object and write the downloaded content into it
    byte_buffer = io.BytesIO()
    for data in model_file_data.stream(amt=8192):
        byte_buffer.write(data)
    # Reset the buffer's position to the beginning
    byte_buffer.seek(0)

    embedding_model.load_safetensors(byte_buffer)
    embedding_model.model=embedding_model.model.to(device)

    return embedding_model

# get the clip text embedding of a prompt or a phrase
def get_prompt_embedding(embedder, device, prompt):
    with torch.no_grad():
        embedding= embedder(prompt)

    embedding= embedding.unsqueeze(0)
    embedding=embedding.to(device)

    return embedding

# get linear or elm score of an embedding
def get_prompt_score(positive_scorer, embedding):
    with torch.no_grad():
        prompt_score=positive_scorer.predict_positive_or_negative_only(embedding)
    
    return prompt_score.item()

# Function to download an image from a URL
def download_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return img

# store generated prompts in a csv file
def store_prompts_in_csv_file(minio_client, data):

    local_path="output/generated_prompts.csv"
    pd.DataFrame(data).to_csv(local_path, index=False)
    # Read the contents of the CSV file
    with open(local_path, 'rb') as file:
        csv_content = file.read()

    #Upload the CSV file to Minio
    buffer = io.BytesIO(csv_content)
    buffer.seek(0)

    minio_path=f"environmental/output/synth-boards-pinterest-dataset/generated_prompts.csv"
    cmd.upload_data(minio_client, 'datasets', minio_path, buffer)
    # Remove the temporary file
    os.remove(local_path)

def main():
    args= parse_args()

    # get minio client
    minio_client = cmd.get_minio_client(args.minio_access_key,
                                        args.minio_secret_key,
                                        args.minio_addr)
    
    # get device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    device = torch.device(device)

    # Load the clip embedder model
    embedder=CLIPTextEmbedder(device=device)
    embedder.load_submodels()

    # Load scoring model
    positive_scorer= load_model(minio_client, device)
    # get mean and std values
    mean, std= float(positive_scorer.mean), float(positive_scorer.standard_deviation)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name=args.clip_model, pretrained=args.clip_pretrain, device=device)
    # Initialize CLIP Interrogator
    # ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))

    # Process each line in the JSONL file
    prompts = []
    with open(args.input_file, 'r') as file:
        for line in tqdm(file):
            # Parse JSON line
            data = json.loads(line)
            board_title = data['board_title']
            image_url = data['image_urls'][0]

            # Download and process the image
            image = download_image(image_url)
            #prompt = ci.interrogate(image)
            # optimize prompt
            learned_prompt = optimize_prompt(model, preprocess, args, device, target_images=[image])[0]
            # except Exception as e:
            #     print(f"Error processing image {image_url}: {e}")
            #     continue

            print(learned_prompt) 
            # get text embedding of the prompt
            prompt_embedding=get_prompt_embedding(embedder, device, learned_prompt)
            
            # get prompt score
            score= get_prompt_score(positive_scorer, prompt_embedding)
            sigma_score= (score - mean) / std

            prompt_data={
                'image_url': image_url, 
                'prompt': learned_prompt,
                'board': board_title,
                'score': score,
                'sigma_score': sigma_score
            }

            # append prompt data
            prompts.append(prompt_data)

    # Output results to CSV files
    prompts_df = pd.DataFrame(prompt_data)
    store_prompts_in_csv_file(minio_client, prompts_df)

    print("Processing complete!")

if __name__=="__main__":
    main()