import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import io
import os
import sys
import time
import traceback
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
from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from training_worker.ab_ranking.model.ab_ranking_linear import ABRankingModel
from stable_diffusion.model.clip_text_embedder.clip_text_embedder import CLIPTextEmbedder
from utility.minio import cmd

from worker.prompt_generation.prompt_generator import generate_image_generation_jobs

DATA_MINIO_DIRECTORY="environmental/data/prompt-generator/substitution"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--csv-phrase', help='CSV containing phrases, must have "phrase str" column', default='input/civitai_phrases_database_v7_no_nsfw.csv')
    parser.add_argument('--csv-initial-prompts', help='CSV containing initial prompts', default='input/environment_data.csv')
    parser.add_argument('--n-data', type=int, help='Number of data samples to generate', default=20)
    parser.add_argument('--send-job', action='store_true', default=True)
    parser.add_argument('--dataset-name', default='test-generations')
    parser.add_argument('--ranking-model', help="elm or linear", default="elm")

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

def load_model(input_size, minio_client, device, scoring_model, embedding_type):
    input_path="environmental/models/ranking/"

    if(scoring_model=="elm-v1"):
        embedding_model = ABRankingELMModel(input_size)
    else:
        embedding_model= ABRankingModel(input_size)

    model_files=cmd.get_list_of_objects_with_prefix(minio_client, 'datasets', input_path)
    most_recent_model = None

    for model_file in model_files:
        if model_file.endswith(f"score-{scoring_model}-embedding-{embedding_type}.pth"):
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

def get_substitute_embedding(minio_client, phrase):
    index=phrase['index'].values[0]
    data = minio_client.get_object('datasets', DATA_MINIO_DIRECTORY + f"/phrase_embeddings/{index}_phrase.msgpack")
    # Read the content of the msgpack file
    content = data.read()

    # Deserialize the content using msgpack
    msgpack_data = msgpack.loads(content)
    return msgpack_data['embedding']

def rank_substitution_choices(device,
                                 embedding_model,  
                                 sigma_model, 
                                 prompt_str, 
                                 prompt_score, prompt_embedding, 
                                 phrase_embeddings,
                                 phrase_list):
    
    # get mean pooled embedding of prompt for xgboost model
    pooled_prompt_embedding= get_mean_pooled_embedding(prompt_embedding)

    # get number of tokens
    prompt_list = prompt_str.split(',')
    token_number= len(prompt_list)
    # list of sigma scores for each substitution
    sigma_scores=[]
    sub_phrases=[]

    # Randomly select a phrase from the dataset and get an embedding
    for token in range(token_number):
        # Get substituted phrase embedding
        substituted_embedding=phrase_embeddings[token]
        # get substitute phrase embedding
        substitute_phrase=random.choice(phrase_list)
        substitute_embedding=get_prompt_embedding(device ,embedding_model, substitute_phrase)
        substitute_embedding= get_mean_pooled_embedding(substitute_embedding)

        substitution_input= np.concatenate([pooled_prompt_embedding, substituted_embedding, substitute_embedding, [token], [prompt_score]])
        # add sigma score to the list of scores
        sigma_score=sigma_model.predict([substitution_input])[0]
        sigma_scores.append(-sigma_score)
        sub_phrases.append(substitute_phrase)
    
    tokens=np.argsort(sigma_scores)
    sub_phrases=[sub_phrases[token] for token in tokens]
    return tokens, sub_phrases

def mutate_prompt(device, embedding_model, sigma_model, scoring_model, 
                  prompt_str, phrase_list, 
                  max_iterations=30, early_stopping=30):

    # calculate prompt embedding, score and embedding of each phrase
    prompt_embedding=get_prompt_embedding(device, embedding_model, prompt_str)
    prompt_score= get_prompt_score(scoring_model, prompt_embedding)
    phrase_embeddings= [get_mean_pooled_embedding(get_prompt_embedding(device, embedding_model, phrase)) for phrase in prompt_str.split(',')]

    # save original score
    original_score=prompt_score 

    print(f"prompt str: {prompt_str}")
    print(f"initial score: {prompt_score}")

    # early stopping
    early_stopping_iterations=early_stopping

    # run mutation process iteratively untill score converges
    for i in range(max_iterations):
        print(f"iteration {i}")
        tokens, sub_phrases=rank_substitution_choices(device,
                                                embedding_model,
                                                sigma_model, 
                                                prompt_str,
                                                prompt_score,
                                                prompt_embedding, 
                                                phrase_embeddings,
                                                phrase_list)
        
        num_attempts=0
        
        for token, sub_phrase in zip(tokens,sub_phrases):
            #Create a modified prompt with the substitution
            prompt_list = prompt_str.split(',')
            substituted_phrase= prompt_list[token]
            prompt_list[token] = sub_phrase
            modified_prompt_str = ",".join(prompt_list)

            #calculate modified prompt embedding and score
            modified_prompt_embedding=get_prompt_embedding(device, embedding_model, modified_prompt_str)
            modified_prompt_score= get_prompt_score(scoring_model, modified_prompt_embedding)

            num_attempts+=1

            # check if score improves
            if(prompt_score < modified_prompt_score):
                prompt_str= modified_prompt_str
                prompt_embedding= modified_prompt_embedding
                phrase_embeddings[token]= get_mean_pooled_embedding(get_prompt_embedding(device, embedding_model, substituted_phrase))
                break

        print(f"failed {num_attempts} times")
        # check if score increased
        if prompt_score >= modified_prompt_score:
            early_stopping_iterations-=1
        else:
            prompt_score= modified_prompt_score
            early_stopping_iterations=early_stopping


        print(f"prompt str: {prompt_str}")
        print(f"initial score: {prompt_score}")
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

    # phrases
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


def store_phrase_embeddings(minio_client, phrases_path):
    phrases_df=pd.read_csv(phrases_path)
    
    # get device
    if torch.cuda.is_available():
            device = 'cuda'
    else:
        device = 'cpu'
    device = torch.device(device)

    # Load the CLIP model
    clip=CLIPTextEmbedder(device=device)
    clip.load_submodels()

    for i, phrases in phrases_df.iterrows():
        phrase_str=phrases['phrase str']
        index=phrases['index']

        phrase_embedding=get_prompt_embedding(device, clip, phrase_str)
        pooled_embedding= get_mean_pooled_embedding(phrase_embedding)

        prompt_data={
            'embedding': pooled_embedding.tolist(),
        }

        packed_data = msgpack.packb(prompt_data, use_single_float=True)

        # Define the local directory path for embedding
        local_directory = 'output/prompt_mutator/phrase_embeddings/'

        # Ensure the local directory exists, create it if necessary
        os.makedirs(local_directory, exist_ok=True)

        # Create a local file with the packed data
        local_file_path = local_directory + f"{index}_phrase.msgpack"
        with open(local_file_path, 'wb') as local_file:
            local_file.write(packed_data)
        
        # Read the contents of the CSV file
        with open(local_file_path, 'rb') as file:
            content = file.read()

        # Upload the local file to MinIO
        buffer = io.BytesIO(content)
        buffer.seek(0)

        minio_path=DATA_MINIO_DIRECTORY + f"/phrase_embeddings/{index}_phrase.msgpack"
        cmd.upload_data(minio_client, 'datasets',minio_path, buffer)

def main():
    args = parse_args()
    # get minio client
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)
    
    prompt_str="environmental, pixel art, concept art, side scrolling, video game, neo city, (1 girl), white box, puffy lips, cinematic lighting, colorful, steampunk, partially submerged, original, 1girl, night, ribbon choker, see through top, black tissues, a masterpiece, high heel, hand on own crotch"

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
    elm_model= load_model(768, minio_client, device, 'linear', 'positive')

    # load the xgboost sigma score model
    sigma_model= PromptMutator(minio_client=minio_client, output_type="sigma_score", ranking_model="linear")
    sigma_model.load_model()

    phrase_list=pd.read_csv(args.csv_phrase)['phrase str'].tolist()

    start=time.time()
    print(mutate_prompt(device=device,
                        embedding_model=clip, 
                        sigma_model=sigma_model, 
                        scoring_model=elm_model,
                        prompt_str=prompt_str, 
                        phrase_list=phrase_list))
    end=time.time()
    print(f'Time taken for mutating a prompt is: {end - start:.2f} seconds')


    #store_phrase_embeddings(minio_client, args.csv_phrase)
    
    # prompts=pd.read_csv(args.csv_initial_prompts)

    # positive_prompts=prompts['positive_prompt'].sample(n=args.n_data, random_state=42)
    # negative_prompts=prompts['negative_prompt'].sample(n=args.n_data, random_state=42)
    
    # mutated_prompts, original_scores, mutated_scores =async_mutate_prompts(prompts, minio_client)
    
    
if __name__ == "__main__":
    main()