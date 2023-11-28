import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import io
import json
import os
import sys
import tempfile
import time
import traceback
from xmlrpc.client import ResponseError
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
from tqdm import tqdm
import msgpack

base_directory = "./"
sys.path.insert(0, base_directory)

from training_worker.prompt_mutator.prompt_mutator_model import PromptMutator
from training_worker.prompt_mutator.binary_prompt_mutator import BinaryPromptMutator
from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from training_worker.ab_ranking.model.ab_ranking_linear import ABRankingModel
from stable_diffusion.model.clip_text_embedder.clip_text_embedder import CLIPTextEmbedder
from utility.minio import cmd

from worker.prompt_generation.prompt_generator import generate_image_generation_jobs

GENERATION_POLICY="greedy-substitution-search-v1"
DATA_MINIO_DIRECTORY="environmental/data/prompt-generator/substitution"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--csv-phrase', help='CSV containing phrases, must have "phrase str" column', default='input/civitai_phrases_database_v7_no_nsfw.csv')
    parser.add_argument('--n-data', type=int, help='Number of data samples to generate', default=20)
    parser.add_argument('--send-job', action='store_true', default=False)
    parser.add_argument('--update-prompts', action='store_true', default=False)
    parser.add_argument('--dataset-name', default='test-generations')
    parser.add_argument('--ranking-model', help="elm-v1 or linear", default="linear")
    parser.add_argument('--rejection-policy', help="by probability or sigma_score", default="sigma_score")

    return parser.parse_args()

def store_prompts_in_csv_file(data, minio_path, minio_client):
    local_path="output/generated_prompts.csv"
    pd.DataFrame(data).to_csv(local_path, index=False)
    # Read the contents of the CSV file
    with open(local_path, 'rb') as file:
        csv_content = file.read()

    #Upload the CSV file to Minio
    buffer = io.BytesIO(csv_content)
    buffer.seek(0)

    minio_path= minio_path + "/generated_prompts.csv"
    cmd.upload_data(minio_client, 'datasets', minio_path, buffer)
    # Remove the temporary file
    os.remove(local_path)


def get_embedding_paths(minio_client, dataset):
    objects=minio_client.list_objects('datasets', dataset, recursive=True)
    embedding_files = []
    for obj in objects: 
        if obj.object_name.endswith("_embedding.msgpack"):
            embedding_files.append(obj.object_name)
            
    return embedding_files

def load_model(minio_client, device, embedding_type, scoring_model="linear", input_size=768):
    input_path="environmental/models/ranking/"

    if(scoring_model=="elm-v1"):
        embedding_model = ABRankingELMModel(input_size)
        file_name=f"score-elm-v1-embedding"
    else:
        embedding_model= ABRankingModel(input_size)
        file_name=f"score-linear-embedding"
    
    if(embedding_type=="positive" or embedding_type=="negative"):
        file_name+=f"-{embedding_type}.pth"
    else:
        file_name+=".pth"

    model_files=cmd.get_list_of_objects_with_prefix(minio_client, 'datasets', input_path)
    most_recent_model = None

    for model_file in model_files:
        if model_file.endswith(file_name):
            most_recent_model = model_file

    if most_recent_model:
        model_file_data =cmd.get_file_from_minio(minio_client, 'datasets', most_recent_model)
    else:
        print("No .pth files found in the list.")
        return
    
    print(most_recent_model)

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

def rejection_sampling_by_sigma_score(device,
                                 embedding_model,  
                                 xgboost_model, 
                                 prompt_str, 
                                 prompt_score, prompt_embedding, 
                                 phrase_embeddings,
                                 phrase_list, mean, std, threshold=0.3):
    
    # get mean pooled embedding of prompt for xgboost model
    pooled_prompt_embedding= get_mean_pooled_embedding(prompt_embedding)
    prompt_sigma_score= (prompt_score - mean) / std

    # get number of tokens
    prompt_list = prompt_str.split(', ')
    token_number= len(prompt_list)
    # list of sigma scores for each substitution
    sub_phrases=[]
    sub_embeddings=[]
    tokens=[]
    sigma_scores=[]

    # Randomly select a phrase from the dataset and get an embedding
    for token in range(token_number):
        # Get substituted phrase embedding
        substituted_embedding=phrase_embeddings[token]
        # get substitute phrase embedding
        substitute_phrase=random.choice(phrase_list)
        substitute_embedding=get_prompt_embedding(device ,embedding_model, substitute_phrase)
        substitute_embedding= get_mean_pooled_embedding(substitute_embedding)

        substitution_input= np.concatenate([pooled_prompt_embedding, substituted_embedding, substitute_embedding, [token], [prompt_sigma_score]])
        sigma_score=xgboost_model.predict([substitution_input])[0]
        if sigma_score > prompt_sigma_score + threshold:
            sigma_scores.append(-sigma_score)
            tokens.append(token)
            sub_phrases.append(substitute_phrase)
            sub_embeddings.append(substitute_embedding)
        
    token_order= np.argsort(sigma_scores)
    tokens=[tokens[token_pos] for token_pos in token_order]
    sub_phrases=[sub_phrases[token_pos] for token_pos in token_order]
    sub_embeddings=[sub_embeddings[token_pos] for token_pos in token_order]
    
    return tokens, sub_phrases, sub_embeddings

def rejection_sampling_by_probability(device,
                                 embedding_model,  
                                 xgboost_model, 
                                 prompt_str, 
                                 prompt_score, prompt_embedding, 
                                 phrase_embeddings,
                                 phrase_list, mean, std):
    
    # get mean pooled embedding of prompt for xgboost model
    pooled_prompt_embedding= get_mean_pooled_embedding(prompt_embedding)
    prompt_sigma_score= (prompt_score - mean) / std

    # get number of tokens
    prompt_list = prompt_str.split(', ')
    token_number= len(prompt_list)
    # list of sigma scores for each substitution
    sub_phrases=[]
    sub_embeddings=[]
    tokens=[]
    decrease_probs=[]

    # Randomly select a phrase from the dataset and get an embedding
    for token in range(token_number):
        # Get substituted phrase embedding
        substituted_embedding=phrase_embeddings[token]
        # get substitute phrase embedding
        substitute_phrase=random.choice(phrase_list)
        substitute_embedding=get_prompt_embedding(device ,embedding_model, substitute_phrase)
        substitute_embedding= get_mean_pooled_embedding(substitute_embedding)

        substitution_input= np.concatenate([pooled_prompt_embedding, substituted_embedding, substitute_embedding, [token], [prompt_sigma_score]])
        pred=xgboost_model.predict_probs([substitution_input])[0]
        if pred["increase"]>0.66:
            decrease_probs.append(pred['decrease'])
            tokens.append(token)
            sub_phrases.append(substitute_phrase)
            sub_embeddings.append(substitute_embedding)
    
    token_order= np.argsort(decrease_probs)
    tokens=[tokens[token_pos] for token_pos in token_order]
    sub_phrases=[sub_phrases[token_pos] for token_pos in token_order]
    sub_embeddings=[sub_embeddings[token_pos] for token_pos in token_order]
    
    return tokens, sub_phrases, sub_embeddings

def mutate_prompt(device, embedding_model,
                  scoring_model, xgboost_model,
                  prompt_str, phrase_list,
                  prompt_embedding, prompt_score, mean, std, 
                  max_iterations=50, rejection_policy="sigma_score"):

    # calculate embedding of each phrase in the prompt 
    phrase_embeddings= [get_mean_pooled_embedding(get_prompt_embedding(device, embedding_model, phrase)) for phrase in prompt_str.split(', ')]

    # get rejection policy function
    if(rejection_policy=="sigma_score"):
        rejection_func=rejection_sampling_by_sigma_score
    else:
        rejection_func=rejection_sampling_by_probability

    num_attempts=0
    num_success=0

    # run mutation process iteratively untill score converges
    for i in range(max_iterations):
        tokens, sub_phrases, embeddings=rejection_func(device,
                                            embedding_model,
                                            xgboost_model, 
                                            prompt_str,
                                            prompt_score,
                                            prompt_embedding, 
                                            phrase_embeddings,
                                            phrase_list, mean, std)
        
        for token, sub_phrase, embedding in zip(tokens,sub_phrases, embeddings):
            #Create a modified prompt with the substitution
            prompt_list = prompt_str.split(', ')
            prompt_list[token] = sub_phrase
            modified_prompt_str = ", ".join(prompt_list)

            #calculate modified prompt embedding and score
            modified_prompt_embedding=get_prompt_embedding(device, embedding_model, modified_prompt_str)
            modified_prompt_score= get_prompt_score(scoring_model, modified_prompt_embedding)

            num_attempts+=1

            # check if score improves
            if(prompt_score < modified_prompt_score):
                prompt_str= modified_prompt_str
                prompt_embedding= modified_prompt_embedding
                phrase_embeddings[token]= embedding
                prompt_score= modified_prompt_score
                num_success+=1
                break
        
        
    print(f"succeeded {num_success} out of {num_attempts} times")
    
    return prompt_str, prompt_embedding

def compare_distributions(minio_client, minio_path, original_scores, mutated_scores):

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    min_val= min(original_scores)
    max_val= max(mutated_scores)

    # plot histogram of original scores
    axs[0].hist(original_scores, bins=10, range=[min_val,max_val], color='blue', alpha=0.7)
    axs[0].set_xlabel('Scores')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Scores Before Mutation')

    # plot histogram of mutated scores
    axs[1].hist(mutated_scores, bins=10, range=[min_val,max_val], color='blue', alpha=0.7)
    axs[1].set_xlabel('Scores')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Scores After mutation')

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.3)

    plt.savefig("output/mutated_scores.png")

    # Save the figure to a file
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # upload the graph report
    minio_path= minio_path + "/mutated_scores.png"
    cmd.upload_data(minio_client, 'datasets', minio_path, buf)
    # Remove the temporary file
    os.remove("output/mutated_scores.png")

def update_prompt_list(minio_client, device):
    embedding_paths = get_embedding_paths(minio_client, "environmental")
    df_data=[]
    elm_scores=[]
    linear_scores=[]
    positive_elm_scores=[]
    positive_linear_scores=[]

    elm_model= load_model(minio_client, device, 'combined', 'elm-v1', input_size=768*2)
    linear_model= load_model(minio_client, device, 'combined', 'linear', input_size=768*2)
    
    positive_elm_model= load_model(minio_client, device, 'positive', 'elm-v1')
    positive_linear_model= load_model(minio_client, device, 'positive', 'linear')

    for embedding in embedding_paths:
        print(f"updated {embedding}")
        # get prompt embedding
        data = minio_client.get_object('datasets', embedding)
        # Read the content of the msgpack file
        content = data.read()

        # Deserialize the content using msgpack
        msgpack_data = msgpack.loads(content)

        # get positive prompt embedding 
        positive_prompt=msgpack_data['positive_prompt']
        positive_embedding= list(msgpack_data['positive_embedding'].values())
        positive_embedding = torch.tensor(np.array(positive_embedding)).float()
        positive_embedding=positive_embedding.to(device)
       
        # get negative prompt embedding 
        negative_prompt=msgpack_data['negative_prompt']
        negative_embedding= list(msgpack_data['negative_embedding'].values())
        negative_embedding = torch.tensor(np.array(negative_embedding)).float()
        negative_embedding=negative_embedding.to(device)

        # get linear and elm score for each prompt
        elm_score=elm_model.predict(positive_embedding, negative_embedding).item()
        linear_score= linear_model.predict(positive_embedding, negative_embedding).item()

        positive_elm_score=positive_elm_model.predict_positive_or_negative_only(positive_embedding).item()
        positive_linear_score=positive_linear_model.predict_positive_or_negative_only(positive_embedding).item()

        elm_scores.append(elm_score)
        linear_scores.append(linear_score)
        positive_elm_scores.append(positive_elm_score)
        positive_linear_scores.append(positive_linear_score)

        # save data 
        df_data.append({
                'job_uuid':msgpack_data['job_uuid'],
                'creation_time':msgpack_data['creation_time'],
                'dataset':msgpack_data['dataset'],
                'file_path':embedding,
                'positive_prompt':positive_prompt,
                'negative_prompt':negative_prompt,
                'linear_score': linear_score,
                'elm_score': elm_score,
                'positive_linear_score': positive_linear_score,
                'positive_elm_score': positive_elm_score,
            })
    
    # save data locally
    pd.DataFrame(df_data).to_csv('output/initial_prompts.csv', index=False)

    # Read the contents of the CSV file
    with open('output/initial_prompts.csv', 'rb') as file:
        csv_content = file.read()

    #Upload the CSV file to Minio
    buffer = io.BytesIO(csv_content)
    buffer.seek(0)

    minio_path = DATA_MINIO_DIRECTORY + "/input/initial_prompts.csv"
    cmd.upload_data(minio_client, 'datasets', minio_path, buffer)

    # Remove the temporary file
    os.remove('output/initial_prompts.csv')

    # updated mean and std values
    data = {
        'linear_mean': np.mean(linear_scores),
        'linear_std': np.std(linear_scores),
        'elm_mean': np.mean(elm_scores),
        'elm_std': np.std(elm_scores),
        'positive_linear_mean': np.mean(positive_linear_scores),
        'positive_linear_std': np.std(positive_linear_scores),
        'positive_elm_mean': np.mean(positive_elm_scores),
        'positive_elm_std': np.std(positive_elm_scores),
    }

    # Writing to a local JSON file
    with open('output/mean_std_values.json', 'w') as json_file:
        json.dump(data, json_file)
        
    # Read the contents of the JSON file
    with open('output/mean_std_values.json', 'rb') as file:
        json_content = file.read()

    #Upload the Json file to Minio
    buffer = io.BytesIO(json_content)
    buffer.seek(0)

    minio_path = DATA_MINIO_DIRECTORY + "/input/mean_std_values.json"
    cmd.upload_data(minio_client, 'datasets', minio_path, buffer)

    # Remove the temporary file
    os.remove('output/mean_std_values.json')

def get_initial_prompts(minio_client, n_data):
    try:
        # Get the CSV file as BytesIO object
        minio_path = DATA_MINIO_DIRECTORY + "/input/initial_prompts.csv"
        data = minio_client.get_object('datasets', minio_path)
        csv_data = io.BytesIO(data.read())

        # Read the CSV into a DataFrame
        df = pd.read_csv(csv_data)

        # Filter the DataFrame based on the condition
        filtered_df = df[df['positive_prompt'].str.split(', ').apply(len)>=10]

  
        # get sample prompts
        sampled_df = filtered_df.sample(n=n_data)

        return sampled_df

    except ResponseError as err:
        print(f"Error: {err}")
        return None

def get_mean_std_values(minio_client, ranking_model):
    minio_path = DATA_MINIO_DIRECTORY + "/input/mean_std_values.json"
    json_file_data =cmd.get_file_from_minio(minio_client, 'datasets', minio_path)

    # Parse JSON data
    data = json.loads(json_file_data.read().decode('utf-8'))

    if(ranking_model=="elm-v1"):
        return data['elm_mean'], data['elm_std'], data['positive_elm_mean'], data['positive_elm_std']
    else:
        return data['linear_mean'], data['linear_std'], data['positive_linear_mean'], data['positive_linear_std']


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
    clip=CLIPTextEmbedder(device=device)
    clip.load_submodels()

    # load the elm model
    positive_model= load_model(minio_client, device, 'positive', args.ranking_model)
    combined_model= load_model(minio_client, device, 'combined', args.ranking_model, input_size=768*2)
    
    # load the xgboost binary model
    if(args.rejection_policy=="sigma_score"):
        if(args.ranking_model=="elm-v1"):
            xgboost_model= PromptMutator(minio_client=minio_client)
        else:
            xgboost_model= PromptMutator(minio_client=minio_client, ranking_model="linear")
    else:
        if(args.ranking_model=="elm-v1"):
            xgboost_model= BinaryPromptMutator(minio_client=minio_client)
        else:
            xgboost_model= BinaryPromptMutator(minio_client=minio_client, ranking_model="linear")

    xgboost_model.load_model()

    # update list of prompts if necessary
    if(args.update_prompts):
        update_prompt_list(minio_client, device)

    # get mean and std values
    #mean, std, positive_mean, positive_std= get_mean_std_values(minio_client,args.ranking_model)
    mean, std= combined_model.mean, combined_model.standard_deviation
    positive_mean, positive_std= positive_model.mean, positive_model.standard_deviation

    # get phrase list for substitutions
    phrase_list=pd.read_csv(args.csv_phrase)['phrase str'].tolist()
    
    # get initial prompts
    prompt_list = get_initial_prompts(minio_client, args.n_data)

    df_data=[]
    original_scores=[]
    mutated_scores=[]
    index=0
 
    start=time.time()
    for i, prompt in prompt_list.iterrows():

        #getting negative and positive prompts
        positive_prompt=prompt['positive_prompt'] 
        negative_prompt=prompt['negative_prompt']
        
        # get prompt embedding
        data = minio_client.get_object('datasets', prompt['file_path'])
        # Read the content of the msgpack file
        content = data.read()

        # Deserialize the content using msgpack
        msgpack_data = msgpack.loads(content)
        positive_embedding= list(msgpack_data['positive_embedding'].values())
        positive_embedding = torch.tensor(np.array(positive_embedding)).float()
        positive_embedding=positive_embedding.to(device)
        
        negative_embedding= list(msgpack_data['negative_embedding'].values())
        negative_embedding = torch.tensor(np.array(negative_embedding)).float()
        negative_embedding=negative_embedding.to(device)

        seed_score=combined_model.predict(positive_embedding, negative_embedding).item()
        positive_score=positive_model.predict_positive_or_negative_only(positive_embedding).item()

        seed_sigma_score=(seed_score - mean) / std
        original_scores.append(seed_sigma_score)

        #mutate positive prompt
        mutated_positive_prompt, mutated_positive_embedding= mutate_prompt(device=device,
                        embedding_model=clip,
                        xgboost_model=xgboost_model, 
                        scoring_model=positive_model,
                        prompt_str=positive_prompt, 
                        phrase_list=phrase_list,
                        prompt_embedding=positive_embedding,
                        prompt_score=positive_score,
                        mean=positive_mean, std=positive_std, rejection_policy=args.rejection_policy)

        # calculating new score
        score=combined_model.predict(mutated_positive_embedding, negative_embedding).item()

        sigma_score=(score - mean) / std
        mutated_scores.append(sigma_score)

        print(f"prompt {index} mutated.")
        print(f"----initial score: {seed_score}.")
        print(f"----final score: {score}.")

        if args.send_job:
            try:
                response = generate_image_generation_jobs(
                    positive_prompt=mutated_positive_prompt,
                    negative_prompt=negative_prompt,
                    prompt_scoring_model=f'image-pair-ranking-{args.ranking_model}',
                    prompt_score=score,
                    prompt_generation_policy=GENERATION_POLICY,
                    top_k='',
                    dataset_name=args.dataset_name
                )
                task_uuid = response['uuid']
                task_time = response['creation_time']
            except:
                print('Error occured:')
                print(traceback.format_exc())
                task_uuid = -1
                task_time = -1

            df_data.append({
                'seed_score': seed_score,
                'seed_sigma_score': seed_sigma_score,
                'score': score,
                'sigma_score': sigma_score,
                'positive_prompt': mutated_positive_prompt,
                'negative_prompt': negative_prompt,
                'seed_prompt': positive_prompt,
                'generation_policy_string': GENERATION_POLICY,
                'task_uuid': task_uuid,
                'time': task_time
            })
        
        index+=1

    end=time.time()

    print(f"time taken for {args.n_data} prompts is {end - start:.2f} seconds")

    current_date=datetime.now().strftime("%Y-%m-%d-%H:%M")
    generation_path=DATA_MINIO_DIRECTORY + f"/generated-images/{current_date}-generated-data"
    # save csv and histogram
    if args.send_job:
        store_prompts_in_csv_file(df_data, generation_path, minio_client)

    compare_distributions(minio_client, generation_path, original_scores, mutated_scores)
    
    
if __name__ == "__main__":
    main()