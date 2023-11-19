import argparse
import csv
import io
import os
import sys
import numpy as np
import pandas as pd
import random
import msgpack
import torch


base_directory = "./"
sys.path.insert(0, base_directory)

from training_worker.prompt_mutator.prompt_mutator_model import PromptMutator
from training_worker.prompt_mutator.multiclass_prompt_mutator import MulticlassPromptMutator
from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from stable_diffusion.model.clip_text_embedder.clip_text_embedder import CLIPTextEmbedder
from utility.minio import cmd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="123.176.98.90:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')

    return parser.parse_args()

def load_model(input_size, minio_client, device):
    input_path="environmental/models/ranking/"

    embedding_model = ABRankingELMModel(input_size)

    model_files=cmd.get_list_of_objects_with_prefix(minio_client, 'datasets', input_path)
    most_recent_model = None

    for model_file in model_files:
        if model_file.endswith("score-elm-v1-embedding-positive.pth"):
            most_recent_model = model_file

    if most_recent_model:
        model_file_data =cmd.get_file_from_minio(minio_client, 'datasets', most_recent_model)
    else:
        print("No .pth files found in the list.")
        return

    # Create a BytesIO object and write the downloaded content into it
    byte_buffer = io.BytesIO()
    for data in model_file_data.stream(amt=8192):
        byte_buffer.write(data)
    # Reset the buffer's position to the beginning
    byte_buffer.seek(0)

    embedding_model.load(byte_buffer)
    embedding_model.model=embedding_model.model.to(device)

    return embedding_model

def get_prompt_embedding(device, model, prompt):
    with torch.no_grad():
        embedding= model(prompt)

    embedding= embedding.unsqueeze(0)
    embedding=embedding.to(device)

    return embedding

def get_prompt_score(model, embedding):
    with torch.no_grad():
        prompt_score=model.predict_positive_or_negative_only(embedding)
    
    return prompt_score.item()

def get_mean_pooled_embedding(embedding):
    embedding=torch.mean(embedding, dim=2)
    embedding = embedding.reshape(len(embedding), -1).squeeze(0)

    return embedding.cpu().numpy()

def get_best_substitution_choice(sigma_model, 
                                 prompt_str, 
                                 prompt_score, prompt_embedding, 
                                 phrase_embeddings):
    
    # get mean pooled embedding of prompt for xgboost model
    pooled_prompt_embedding= get_mean_pooled_embedding(prompt_embedding)

    # get number of tokens
    prompt_list = prompt_str.split(',')
    token_number= len(prompt_list)
    # list of delta scores for each substitution
    sigma_scores=[]

    # Randomly select a phrase from the dataset and get an embedding
    for token in range(token_number):
        # Get substitution phrase embedding
        substituted_embedding=phrase_embeddings[token]
        # get full input for xgboost model
        substituted_embedding= get_mean_pooled_embedding(substituted_embedding)
        substitution_input= np.concatenate([pooled_prompt_embedding, substituted_embedding, [token], [prompt_score]])
        # add sigma score to the list of scores
        sigma_score=sigma_model.predict([substitution_input])[0]
        sigma_scores.append(-sigma_score)
    
    tokens_to_substitute=np.argsort(sigma_scores)
    return tokens_to_substitute

def mutate_prompt(device, embedding_model,
                  sigma_model, scoring_model, 
                  prompt_str, phrase_list, 
                  max_iterations=1000, early_stopping=3):
    
    # calculate prompt embedding and score
    prompt_embedding=get_prompt_embedding(device, embedding_model, prompt_str)
    prompt_score= get_prompt_score(scoring_model, prompt_embedding)
    phrase_embeddings=[get_prompt_embedding(device, embedding_model, phrase) for phrase in prompt_str.split(',')]
    
    print(f"prompt str: {prompt_str}")
    print(f"initial score: {prompt_score}")

    # run mutation process iteratively untill score converges
    for i in range(max_iterations):
        print(f"iteration {i}")
        tokens=get_best_substitution_choice(sigma_model, 
                                                        prompt_str,
                                                        prompt_score,
                                                        prompt_embedding, 
                                                        phrase_embeddings)
        
        if len(tokens)==0:
            break

        for token in tokens:
            #Create a modified prompt with the substitution
            prompt_list = prompt_str.split(',')
            substituted_phrase= prompt_list[token]
            prompt_list[token] = random.choice(phrase_list)
            modified_prompt_str = ",".join(prompt_list)

            #calculate modified prompt embedding and score
            modified_prompt_embedding=get_prompt_embedding(device, embedding_model, modified_prompt_str)
            modified_prompt_score= get_prompt_score(scoring_model, modified_prompt_embedding)

            # check if score improves
            if(prompt_score< modified_prompt_score):
                prompt_str= modified_prompt_str
                prompt_embedding= modified_prompt_embedding
                phrase_embeddings[token]= get_prompt_embedding(device, embedding_model, substituted_phrase)
                break

        # check if score increased
        if prompt_score >= modified_prompt_score:
            early_stopping-=1
        else:
            early_stopping=3
        
        prompt_score= modified_prompt_score
        print(f"----mutated prompt str: {prompt_str}")
        print(f"----resulting score: {prompt_score}")
        if early_stopping==0:
            break

def main():
    args = parse_args()

    # get minio client
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)

    # get device
    if torch.cuda.is_available():
            device = 'cuda'
    else:
        device = 'cpu'
    device = torch.device(device)

    # Load the CLIP model
    clip=CLIPTextEmbedder()
    clip.load_submodels()

    # load the elm model
    elm_model= load_model(768, minio_client, device)

    # load the xgboost model
    sigma_model= PromptMutator(minio_client=minio_client, output_type="sigma_score")
    sigma_model.load_model()

    # load the classification model
    # binary_model= MulticlassPromptMutator(minio_client=minio_client)
    # binary_model.load_model('environmental/output/prompt_mutator/binary_prompt_mutator.json')

    # prompt and phrases
    prompt_str="environmental, pixel art, concept art, side scrolling, video game, neo city, (1 girl), white box, puffy lips, cinematic lighting, colorful, steampunk, partially submerged, original, 1girl, night, ribbon choker, see through top, black tissues, a masterpiece, high heel, hand on own crotch"
    phrases_list=pd.read_csv('input/phrase_scores.csv')['phrase'].tolist()

    # mutate prompt
    mutate_prompt(device, 
                  embedding_model=clip, 
                  scoring_model=elm_model, 
                  sigma_model=sigma_model,
                  prompt_str=prompt_str,
                  phrase_list=phrases_list)
    
if __name__ == "__main__":
    main()