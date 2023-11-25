import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
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
    parser.add_argument('--csv-initial-prompts', help='CSV containing initial prompts', default='input/environment_data.csv')
    parser.add_argument('--n-data', type=int, help='Number of data samples to generate', default=20)
    parser.add_argument('--send-job', action='store_true', default=False)
    parser.add_argument('--update-prompts', action='store_true', default=False)
    parser.add_argument('--dataset-name', default='test-generations')
    parser.add_argument('--ranking-model', help="elm-v1 or linear", default="linear")

    return parser.parse_args()

def store_prompts_in_csv_file(csv_path, minio_client):
    
    # Read the contents of the CSV file
    with open(csv_path, 'rb') as file:
        csv_content = file.read()

    #Upload the CSV file to Minio
    buffer = io.BytesIO(csv_content)
    buffer.seek(0)

    minio_path = DATA_MINIO_DIRECTORY + "/generated_prompts.csv"
    cmd.upload_data(minio_client, 'datasets', minio_path, buffer)

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

def rank_substitution_choices(device,
                                 embedding_model,  
                                 binary_model, 
                                 prompt_str, 
                                 prompt_score, prompt_embedding, 
                                 phrase_embeddings,
                                 phrase_list, mean, std):
    
    # get mean pooled embedding of prompt for xgboost model
    pooled_prompt_embedding= get_mean_pooled_embedding(prompt_embedding)

    # get number of tokens
    prompt_list = prompt_str.split(', ')
    token_number= len(prompt_list)
    # list of sigma scores for each substitution
    sub_phrases=[]
    sub_embeddings=[]
    tokens=[]

    # Randomly select a phrase from the dataset and get an embedding
    for token in range(token_number):
        # Get substituted phrase embedding
        substituted_embedding=phrase_embeddings[token]
        # get substitute phrase embedding
        substitute_phrase=random.choice(phrase_list)
        substitute_embedding=get_prompt_embedding(device ,embedding_model, substitute_phrase)
        substitute_embedding= get_mean_pooled_embedding(substitute_embedding)

        substitution_input= np.concatenate([pooled_prompt_embedding, substituted_embedding, substitute_embedding, [token], [prompt_score]])
        # sigma_score=sigma_model.predict([substitution_input])[0]
        # sigma_scores.append(-sigma_score)
        pred=binary_model.predict_probs([substitution_input])[0]
        if pred["increase"]>0.66:
            tokens.append(token)
            sub_phrases.append(substitute_phrase)
            sub_embeddings.append(substitute_embedding)
    
    return tokens, sub_phrases, sub_embeddings

def mutate_prompt(device, embedding_model,
                  scoring_model, binary_model,
                  prompt_str, phrase_list,
                  prompt_embedding, prompt_score,
                  mean, std, 
                  max_iterations=50):

    # calculate prompt embedding, score and embedding of each phrase
    #prompt_embedding=get_prompt_embedding(device, embedding_model, prompt_str)
    prompt_score= get_prompt_score(scoring_model, prompt_embedding)
    phrase_embeddings= [get_mean_pooled_embedding(get_prompt_embedding(device, embedding_model, phrase)) for phrase in prompt_str.split(', ')]

    # run mutation process iteratively untill score converges
    for i in range(max_iterations):
        tokens, sub_phrases, embeddings=rank_substitution_choices(device,
                                                embedding_model,
                                                binary_model, 
                                                prompt_str,
                                                prompt_score,
                                                prompt_embedding, 
                                                phrase_embeddings,
                                                phrase_list,
                                                mean,std)
        
        modified_prompt_score=prompt_score
        for token, sub_phrase, embedding in zip(tokens,sub_phrases, embeddings):
            #Create a modified prompt with the substitution
            prompt_list = prompt_str.split(', ')
            prompt_list[token] = sub_phrase
            modified_prompt_str = ", ".join(prompt_list)

            #calculate modified prompt embedding and score
            modified_prompt_embedding=get_prompt_embedding(device, embedding_model, modified_prompt_str)
            modified_prompt_score= get_prompt_score(scoring_model, modified_prompt_embedding)

            # check if score improves
            if(prompt_score < modified_prompt_score):
                prompt_str= modified_prompt_str
                prompt_embedding= modified_prompt_embedding
                phrase_embeddings[token]= embedding
                prompt_score= modified_prompt_score
                break
    
    return prompt_str, prompt_embedding

def compare_distributions(minio_client,original_scores, mutated_scores):

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # plot histogram of original scores
    axs[0].hist(original_scores, bins=10, range=[10000,30000], color='blue', alpha=0.7)
    axs[0].set_xlabel('Scores')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Scores Before Mutation')

    # plot histogram of mutated scores
    axs[1].hist(mutated_scores, bins=10, range=[10000,30000], color='blue', alpha=0.7)
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
    minio_path = DATA_MINIO_DIRECTORY + "/generated_prompts.png"
    cmd.upload_data(minio_client, 'datasets', minio_path, buf)  

def update_prompt_list(minio_client, device, embedding_model):
    embedding_paths = get_embedding_paths(minio_client, "environmental")
    embedding_paths=embedding_paths[:10]
    df_data=[]
    elm_scores=[]
    linear_scores=[]

    elm_model= load_model(minio_client, device, 'combined', 'elm-v1', input_size=768*2)
    linear_model= load_model(minio_client, device, 'combined', 'linear', input_size=768*2)

    for embedding in embedding_paths:
        print(embedding)
        # get prompt embedding
        data = minio_client.get_object('datasets', embedding)
        # Read the content of the msgpack file
        content = data.read()

        # Deserialize the content using msgpack
        msgpack_data = msgpack.loads(content)

        # get positive prompt embedding 
        positive_prompt=msgpack_data['positive_prompt']
        positive_embedding= get_prompt_embedding(device, embedding_model, positive_prompt)
        # positive_embedding= list(msgpack_data['positive_embedding'].values())
        # positive_embedding = torch.tensor(np.array(positive_embedding)).float()
        # positive_embedding=positive_embedding.to(device)
       
        # get negative prompt embedding 
        negative_prompt=msgpack_data['negative_prompt']
        negative_embedding= get_prompt_embedding(device, embedding_model, negative_prompt)
        # negative_embedding= list(msgpack_data['negative_embedding'].values())
        # negative_embedding = torch.tensor(np.array(negative_embedding)).float()
        # negative_embedding=negative_embedding.to(device)

        # get linear and elm score for each prompt
        elm_score=elm_model.predict(positive_embedding, negative_embedding).item()
        linear_score= linear_model.predict(positive_embedding, negative_embedding).item()

        elm_scores.append(elm_score)
        linear_scores.append(linear_score)

        # save data 
        df_data.append({
                'job_uuid':msgpack_data['job_uuid'],
                'creation_time':msgpack_data['creation_time'],
                'dataset':msgpack_data['dataset'],
                'file_path':msgpack_data['file_path'],
                'file_hash':msgpack_data['file_hash'],
                'positive_prompt':positive_prompt,
                'negative_prompt':negative_prompt,
                'linear_score': linear_score,
                'elm_score': elm_score
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

        # Read CSV using pandas
        df = pd.read_csv(csv_data).sample(n=n_data)

        return df

    except ResponseError as err:
        print(f"Error: {err}")
        return None

def get_mean_std_values(minio_client, ranking_model):
    minio_path = DATA_MINIO_DIRECTORY + "/input/mean_std_values.json"
    json_file_data =cmd.get_file_from_minio(minio_client, 'datasets', minio_path)

    # Parse JSON data
    data = json.loads(json_file_data.read().decode('utf-8'))

    if(ranking_model=="elm-v1"):
        return data['elm_mean'], data['elm_std']
    else:
        return data['linear_mean'], data['linear_std']


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
    if(args.ranking_model=="elm-v1"):
        binary_model= BinaryPromptMutator(minio_client=minio_client)
    else:
        binary_model= BinaryPromptMutator(minio_client=minio_client, ranking_model="linear")

    binary_model.load_model()

    # update list of prompts if necessary
    if(args.update_prompts):
        update_prompt_list(minio_client, device, clip)

    # get mean and std values
    mean, std= get_mean_std_values(minio_client,args.ranking_model)
    print(mean,std)

    phrase_list=pd.read_csv(args.csv_phrase)['phrase str'].tolist()
    prompt_list = get_initial_prompts(minio_client, args.n_data)
    df_data=[]
    original_scores=[]
    mutated_scores=[]
    index=0

    return

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

        #negative_embedding= get_prompt_embedding(device, clip, negative_prompt)
        #positive_embedding= get_prompt_embedding(device, clip, positive_prompt)

        #calculating combined score
        #seed_score=combined_model.predict(positive_embedding, negative_embedding).item()

        if(args.ranking_model=="elm-v1"):
            seed_score=prompt['elm_score']
        else:
            seed_score=prompt['linear_score']

        original_scores.append(seed_score)

        #mutate positive prompt
        mutated_positive_prompt, mutated_positive_embedding= mutate_prompt(device=device,
                        embedding_model=clip,
                        binary_model=binary_model, 
                        scoring_model=positive_model,
                        prompt_str=positive_prompt, 
                        phrase_list=phrase_list,
                        prompt_embedding=positive_embedding,
                        mean=mean, std=std)

        # calculating new score
        score=combined_model.predict(mutated_positive_embedding, negative_embedding).item()

        mutated_scores.append(score)

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
                'seed_sigma_score': (seed_score - mean) / std,
                'score': score,
                'sigma_score': (score - mean) / std,
                'positive_prompt': mutated_positive_prompt,
                'negative_prompt': negative_prompt,
                'seed_prompt': positive_prompt,
                'generation_policy_string': GENERATION_POLICY,
                'task_uuid': task_uuid,
                'time': task_time
            })
        
        index+=1

    # save csv
    if args.send_job:
        path="output/generated_prompts.csv"
        pd.DataFrame(df_data).to_csv(path, index=False)
        store_prompts_in_csv_file(path, minio_client)

    compare_distributions(minio_client, original_scores, mutated_scores)
    
    
if __name__ == "__main__":
    main()