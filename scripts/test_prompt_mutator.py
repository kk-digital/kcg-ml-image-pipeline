import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import io
import os
import sys
import traceback
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
from tqdm import tqdm


base_directory = "./"
sys.path.insert(0, base_directory)

from training_worker.prompt_mutator.prompt_mutator_model import PromptMutator
from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from stable_diffusion.model.clip_text_embedder.clip_text_embedder import CLIPTextEmbedder
from utility.minio import cmd

from worker.prompt_generation.prompt_generator import generate_image_generation_jobs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="123.176.98.90:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')

    return parser.parse_args()

def store_in_csv_file(csv_data, minio_client):
    # Save data to a CSV file
    csv_file = 'output/prompt_mutator/generated_prompts.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Prompt', 'Mutated Pormpt', 'Original Score', 'New Score'])
        # Write the data
        writer.writerows(csv_data)
    
    # Read the contents of the CSV file
    with open(csv_file, 'rb') as file:
        csv_content = file.read()

    #Upload the CSV file to Minio
    buffer = io.BytesIO(csv_content)
    buffer.seek(0)

    model_path = 'environmental' + 'output/prompt_mutator/generated_prompts/prompts.csv'
    cmd.upload_data(minio_client, 'datasets', model_path, buffer)

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

def rank_substitution_choices(top_percentage,   
                                 sigma_model, 
                                 prompt_str, 
                                 prompt_score, prompt_embedding, 
                                 phrase_embeddings):
    
    # get mean pooled embedding of prompt for xgboost model
    pooled_prompt_embedding= get_mean_pooled_embedding(prompt_embedding)

    # get number of tokens
    prompt_list = prompt_str.split(',')
    token_number= len(prompt_list)
    # list of sigma scores for each substitution
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
    
    tokens=np.argsort(sigma_scores)
    top_tokens= tokens[:int(token_number * top_percentage)+1]
    return top_tokens

def mutate_prompt(device, embedding_model, sigma_model, scoring_model, 
                  prompt_str, phrase_list, 
                  max_iterations=100, early_stopping=30):

    # calculate prompt embedding, score and embedding of each phrase
    prompt_embedding=get_prompt_embedding(device, embedding_model, prompt_str)
    prompt_score= get_prompt_score(scoring_model, get_prompt_embedding(device, embedding_model, prompt_str))
    phrase_embeddings= [get_prompt_embedding(device, embedding_model, phrase) for phrase in prompt_str.split(',')]

    # save original score
    original_score=prompt_score 

    # print(f"prompt str: {prompt_str}")
    # print(f"initial score: {prompt_score}")

    # early stopping
    early_stopping_iterations=early_stopping

    # boolean for if score increased
    score_increased=True
    # run mutation process iteratively untill score converges
    for i in range(max_iterations):
        #print(f"iteration {i}")
        if score_increased:
            tokens=rank_substitution_choices(0.3,   
                                                sigma_model, 
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
            if(prompt_score < modified_prompt_score):
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


        # print(f"prompt str: {prompt_str}")
        # print(f"initial score: {prompt_score}")
        if early_stopping_iterations==0:
            break
    
    return prompt_str, original_score, prompt_score

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

    # load the xgboost sigma score model
    sigma_model= PromptMutator(minio_client=minio_client, output_type="sigma_score")
    sigma_model.load_model()

    # prompts
    phrases_list=pd.read_csv('input/phrase_scores.csv')['phrase'].tolist()

    # get scores before and after mutation
    original_scores=[]
    mutated_scores=[]
    mutated_prompts=[]
    csv_data=[]

    index=0

    for prompt_str in prompts:
        #print(f"prompt {index}")

        # mutate prompt
        mutated_str, original_score, mutated_score= mutate_prompt(device, 
                    embedding_model=clip, 
                    scoring_model=elm_model, 
                    sigma_model=sigma_model,
                    prompt_str=prompt_str,
                    phrase_list=phrases_list)
        
        mutated_prompts.append(mutated_str)
        original_scores.append(original_score)
        mutated_scores.append(mutated_score)
        
        # put data in csv
        csv_data.append([
            prompt_str,
            mutated_str,
            original_score,
            mutated_score
        ])

        index+=1

    #store_in_csv_file(csv_data, minio_client)
    return mutated_prompts, original_scores, mutated_scores

def async_mutate_prompts(prompts, minio_client):
    # get device
    if torch.cuda.is_available():
            device = 'cuda'
    else:
        device = 'cpu'
    device = torch.device(device)

    # Load the CLIP model
    clip=CLIPTextEmbedder(device=device)
    clip.load_submodels()

    # load the elm model
    elm_model= load_model(768, minio_client, device)

    # load the xgboost sigma score model
    sigma_model= PromptMutator(minio_client=minio_client, output_type="sigma_score")
    sigma_model.load_model()

    # prompts
    phrases_list=pd.read_csv('input/phrase_scores.csv')['phrase'].tolist()

    # get scores before and after mutation
    original_scores=[]
    mutated_scores=[]
    mutated_prompts=[]
    csv_data=[]

    index=0
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures=[]
        for prompt_str in prompts:
            print(f"Prompt {index} added to queue")
            futures.append(executor.submit(mutate_prompt, device, clip, sigma_model, elm_model,prompt_str, phrases_list))
            index+=1

        for _ in tqdm(as_completed(futures), total=len(prompts)):
                    continue
    
    # Iterate over completed futures
    for future in futures:
        try:
            # Retrieve the result of each task
            mutated_str, original_score, mutated_score = future.result()
            mutated_prompts.append(mutated_str)
            mutated_scores.append(mutated_score)
            original_scores.append(original_score)
            
            # put data in csv
            csv_data.append([
                prompt_str,
                mutated_str,
                original_score,
                mutated_score
            ])

        except Exception as e:
            print(f"Error during mutation: {e}")

    #store_in_csv_file(csv_data, minio_client)
    return mutated_prompts, original_scores, mutated_scores
    
def compare_distributions(minio_client,original_scores, mutated_scores):

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # plot histogram of original scores
    axs[0].hist(original_scores, bins=10, range=[0,5000], color='blue', alpha=0.7)
    axs[0].set_xlabel('Scores')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Scores Before Mutation')

    # plot histogram of mutated scores
    axs[1].hist(mutated_scores, bins=10, range=[0,5000], color='blue', alpha=0.7)
    axs[1].set_xlabel('Scores')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Scores After mutation')

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.3)

    plt.savefig("output/prompt_mutator/mutated_scores.png")

    # Save the figure to a file
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # upload the graph report
    #cmd.upload_data(minio_client, 'datasets', "environmental/output/prompt_mutator/generated_prompts/mutated_scores.png", buf)  

def generate_images():
    generated_prompts=pd.read_csv('output/prompt_mutator/prompts.csv')
    sorted_prompts = generated_prompts.sort_values(by='New Score', ascending=False)
    
    # Get the top 50 prompts
    generated_prompts = sorted_prompts.head(50)
    
    df_data=[]

    for index, prompt in generated_prompts.iterrows():
        try:
            response = generate_image_generation_jobs(
                positive_prompt=prompt['Mutated Pormpt'],
                negative_prompt='',
                prompt_scoring_model='image-pair-ranking-elm-v1',
                prompt_score=prompt['New Score'],
                prompt_generation_policy='greedy-substitution-search-v1',
                top_k='',
                dataset_name='test-generations'
            )
            task_uuid = response['uuid']
            task_time = response['creation_time']
        except:
            print('Error occured:')
            print(traceback.format_exc())
            task_uuid = -1
            task_time = -1

        # data to include to output csv file
        # first 4 fields are standard
        df_data.append({
            'task_uuid': task_uuid,
            'score': prompt['New Score'],
            'generation_policy_string': 'greedy-substitution-search-v1',
            'time': task_time,
            'prompt': prompt['Mutated Pormpt'],
            'seed_elm_score': prompt['Original Score'],
        })

        # save csv at every iteration just in case script crashes while running
        pd.DataFrame(df_data).to_csv('output/generated_prompts.csv', index=False)


def main():
    args = parse_args()

    
    # get minio client
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)
    
    prompts=pd.read_csv('input/environment_data.csv')['positive_prompt'].sample(n=30, random_state=42)
    mutated_prompts, original_scores, mutated_scores =async_mutate_prompts(prompts, minio_client)
    print(original_scores)
    print(mutated_scores)
    #compare_distributions(minio_client, original_scores, mutated_scores)
    
    
if __name__ == "__main__":
    main()