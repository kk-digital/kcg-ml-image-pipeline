import argparse
import csv
import io
import os
import sys
from matplotlib import pyplot as plt
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

def mutate_prompt(device, embedding_model, sigma_model, scoring_model, 
                  prompt_str, prompt_embedding, prompt_score, 
                  phrase_embeddings, phrase_list, 
                  max_iterations=800, early_stopping=40):
    
    # early stopping
    early_stopping_iterations=early_stopping
    
    print(f"prompt str: {prompt_str}")
    print(f"initial score: {prompt_score}")

    # boolean for if score increased
    score_increased=True
    # run mutation process iteratively untill score converges
    for i in range(max_iterations):
        print(f"iteration {i}")

        if score_increased:
            tokens=get_best_substitution_choice(sigma_model, 
                                                prompt_str,
                                                prompt_score,
                                                prompt_embedding, 
                                                phrase_embeddings)
        
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
            early_stopping_iterations-=1
            score_increased=False
        else:
            score_increased=True
            prompt_score= modified_prompt_score
            early_stopping_iterations=early_stopping
        
        print(f"----mutated prompt str: {prompt_str}")
        print(f"----resulting score: {prompt_score}")
        if early_stopping_iterations==0:
            break
    
    return prompt_str, prompt_score

def mutate_prompts(prompts, minio_client):
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

    # prompts
    phrases_list=pd.read_csv('input/phrase_scores.csv')['phrase'].tolist()

    # get scores before and after mutation
    original_scores=[]
    mutated_scores=[]
    mutated_prompts=[]

    for prompt_str in prompts:
        # calculate prompt embedding and score
        prompt_embedding=get_prompt_embedding(device, clip, prompt_str)
        prompt_score= get_prompt_score(elm_model, prompt_embedding)
        phrase_embeddings=[get_prompt_embedding(device, clip, phrase) for phrase in prompt_str.split(',')]

        original_scores.append(prompt_score)

        # mutate prompt
        prompt_str, prompt_score= mutate_prompt(device, 
                    embedding_model=clip, 
                    scoring_model=elm_model, 
                    sigma_model=sigma_model,
                    prompt_str=prompt_str,
                    prompt_embedding=prompt_embedding,
                    prompt_score= prompt_score,
                    phrase_embeddings= phrase_embeddings,
                    phrase_list=phrases_list)
        
        mutated_prompts.append(prompt_str)
        mutated_scores.append(prompt_score)
    
    return mutated_prompts, original_scores, mutated_scores
    

def compare_distributions(minio_client,original_scores, mutated_scores):

    fig, axs = plt.subplots(1, 2, figsize=(12, 10))
    
    # plot histogram of original scores
    axs[0][0].hist(original_scores, bins=10, color='blue', alpha=0.7)
    axs[0][0].set_xlabel('Scores')
    axs[0][0].set_ylabel('Frequency')
    axs[0][0].set_title('Scores Before Mutation')

    # plot histogram of mutated scores
    axs[0][1].hist(mutated_scores, bins=10, color='blue', alpha=0.7)
    axs[0][1].set_xlabel('Scores')
    axs[0][1].set_ylabel('Frequency')
    axs[0][1].set_title('Scores After mutation')

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.5)

    plt.savefig("output/prompt_mutator/mutated_scores.png")

    # Save the figure to a file
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # upload the graph report
    cmd.upload_data(minio_client, 'datasets', "environmental/output/prompt_mutator/mutated_scores.png", buf)  

def main():
    args = parse_args()

    # get minio client
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)
    
    prompts=pd.read_csv('input/environment_data.csv')['positive_prompt'].sample(n=1000, random_state=42)
    
    mutated_prompts, original_scores, mutated_scores =mutate_prompts(prompts, minio_client)

    compare_distributions(minio_client, original_scores, mutated_scores)
    
    
if __name__ == "__main__":
    main()