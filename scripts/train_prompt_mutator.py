import argparse
import csv
import io
import json
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
from training_worker.prompt_mutator.binary_prompt_mutator import BinaryPromptMutator
from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from training_worker.ab_ranking.model.ab_ranking_linear import ABRankingModel
from stable_diffusion.model.clip_text_embedder.clip_text_embedder import CLIPTextEmbedder
from utility.minio import cmd

DATA_MINIO_DIRECTORY="environmental/data/prompt-generator/substitution"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--csv-phrase', help='CSV containing phrases, must have "phrase str" column', default='input/civitai_phrases_database_v7_no_nsfw.csv')
    parser.add_argument('--embedding-type', help='type of embedding, positive or negative', default='positive')
    parser.add_argument('--create-dataset', help='whether to create a new dataset or load existing one', default=False)
    args = parser.parse_args()
    return args

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

def get_embedding_paths(minio_client, dataset):
    objects=minio_client.list_objects('datasets', dataset, recursive=True)
    embedding_files = []
    for obj in objects: 
        if obj.object_name.endswith("_embedding.msgpack"):
            embedding_files.append(obj.object_name)
            
    return embedding_files

def get_self_training_paths(minio_client):
    # get minio paths for existing self training data
    dataset_path=DATA_MINIO_DIRECTORY + f"/self_training/"
    dataset_files=minio_client.list_objects('datasets', prefix=dataset_path, recursive=True)
    dataset_files= [file.object_name for file in dataset_files]
    
    return dataset_files

def store_in_csv_file(csv_data, minio_client, embedding_type):
    # Save data to a CSV file
    csv_file = 'output/prompt_substitution_dataset.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Prompt', 'Substitute Phrase', 'Substituted Phrase', 
                         'Substitution Position', 'Original_elm_score', 'New_elm_score',
                         'Original_linear_score', 'New_linear_score'])
        # Write the data
        writer.writerows(csv_data)
    
    # Read the contents of the CSV file
    with open(csv_file, 'rb') as file:
        csv_content = file.read()

    #Upload the CSV file to Minio
    buffer = io.BytesIO(csv_content)
    buffer.seek(0)

    minio_path = DATA_MINIO_DIRECTORY+ f'/{embedding_type}_dataset.csv'
    cmd.upload_data(minio_client, 'datasets', minio_path, buffer)

def store_in_msgpack_file(prompt_data, index, minio_client, embedding_type):
    packed_data = msgpack.packb(prompt_data, use_single_float=True)

    # Define the local directory path for embedding
    local_directory = 'output/prompt_mutator/data/'

    # Ensure the local directory exists, create it if necessary
    os.makedirs(local_directory, exist_ok=True)

    # Create a local file with the packed data
    local_file_path = local_directory + f"{str(index).zfill(6)}_substitution.msgpack"
    with open(local_file_path, 'wb') as local_file:
        local_file.write(packed_data)
    
    # Read the contents of the CSV file
    with open(local_file_path, 'rb') as file:
        content = file.read()

    # Upload the local file to MinIO
    buffer = io.BytesIO(content)
    buffer.seek(0)

    minio_path=DATA_MINIO_DIRECTORY + f"/{embedding_type}_prompts/{str(index).zfill(6)}_substitution.msgpack"
    cmd.upload_data(minio_client, 'datasets',minio_path, buffer)
    

def create_dataset(minio_client, device, csv_path, embedding_type):
    # Load the CLIP model
    clip=CLIPTextEmbedder(device=device)
    clip.load_submodels()

    # get dataset of phrases
    phrases_df = pd.read_csv(csv_path)
    # get ranking mondel
    elm_model= load_model(768,minio_client, device, 'elm-v1', embedding_type)
    linear_model= load_model(768,minio_client, device, 'linear', embedding_type)
    # get mean and std values
    elm_mean, elm_std= elm_model.mean, elm_model.standard_deviation
    linear_mean, linear_std= linear_model.mean, linear_model.standard_deviation

    print(f"elm mean: {elm_mean}, elm std {elm_std}")
    print(f"linear mean: {linear_mean}, linear std {linear_std}")

    # get minio paths for embeddings
    embedding_paths = get_embedding_paths(minio_client, "environmental")

    prompt_index=1
    csv_data = []

    for embedding in embedding_paths:
        print(f"prompt {prompt_index}")

        # get prompt embedding
        data = minio_client.get_object('datasets', embedding)
        # Read the content of the msgpack file
        content = data.read()

        # Deserialize the content using msgpack
        msgpack_data = msgpack.loads(content)

        # get prompt embedding 
        prompt_str=msgpack_data[f'{embedding_type}_prompt']
        prompt_embedding= list(msgpack_data[f'{embedding_type}_embedding'].values())
        prompt_embedding = torch.tensor(np.array(prompt_embedding)).float()
        prompt_embedding=prompt_embedding.to(device)

        #Randomly select a phrase from the dataset and get an embedding
        substitute_phrase = random.choice(phrases_df['phrase str'].tolist())
        with torch.no_grad():
                substitute_embedding= clip.forward(substitute_phrase).unsqueeze(0)
        
        prompt_list = prompt_str.split(', ')
        # Choose a random position to substitute in the prompt
        position_to_substitute = random.randint(0, len(prompt_list) - 1)

        # Create a modified prompt with the substitution and get embedding of substituted phrase
        substituted_phrase=prompt_list[position_to_substitute]
        with torch.no_grad():
            substituted_embedding= clip.forward(substituted_phrase).unsqueeze(0)

        prompt_list[position_to_substitute] = substitute_phrase
        modified_prompt = ", ".join(prompt_list)

        # Get embedding of mutated prompt
        with torch.no_grad():
            modified_embedding= clip.forward(modified_prompt)
            modified_embedding= modified_embedding.unsqueeze(0)
            modified_embedding=modified_embedding.to(device)

        # get score before and after substitution
        with torch.no_grad():
            elm_prompt_score=elm_model.predict_positive_or_negative_only(prompt_embedding).item()
            elm_modified_pormpt_score= elm_model.predict_positive_or_negative_only(modified_embedding).item()
            
            linear_prompt_score=linear_model.predict_positive_or_negative_only(prompt_embedding).item()
            linear_modified_pormpt_score= linear_model.predict_positive_or_negative_only(modified_embedding).item()
        
        # mean pooling
        pooled_prompt_embedding=torch.mean(prompt_embedding, dim=2)
        #flattening embedding
        pooled_prompt_embedding = pooled_prompt_embedding.reshape(len(pooled_prompt_embedding), -1).squeeze(0)
        
        # mean pooling
        pooled_substituted_embedding=torch.mean(substituted_embedding, dim=2)
        #flattening embedding
        pooled_substituted_embedding = pooled_substituted_embedding.reshape(len(pooled_substituted_embedding), -1).squeeze(0)
        
        # mean pooling
        pooled_substitute_embedding=torch.mean(substitute_embedding, dim=2)
        #flattening embedding
        pooled_substitute_embedding = pooled_substitute_embedding.reshape(len(pooled_substitute_embedding), -1).squeeze(0)

        # calculate sigma scores
        sigma_elm_score=(elm_prompt_score - elm_mean) / elm_std
        modified_sigma_elm_score= (elm_modified_pormpt_score - elm_mean) / elm_std
        sigma_linear_score= (linear_prompt_score - linear_mean) / linear_std
        modified_sigma_linear_score= (linear_modified_pormpt_score - linear_mean) / linear_std 

        # Append to the CSV data list
        csv_data.append([
            prompt_str,  # Prompt string
            substitute_phrase,        # Substitute phrase string
            substituted_phrase,  # Substituted phrase string
            position_to_substitute,   # Substitution position
            sigma_elm_score, # elm score before substitution
            modified_sigma_elm_score, # elm score after substitution
            sigma_linear_score, # linear score before substitution
            modified_sigma_linear_score # linear score after substitution
        ])

        # Append to the msgpack data list
        prompt_data={
        'input': torch.cat([pooled_prompt_embedding, pooled_substituted_embedding, pooled_substitute_embedding], dim=0).tolist(),
        'position_encoding': position_to_substitute,
        'elm_score_encoding': sigma_elm_score,
        'elm_output': modified_sigma_elm_score,
        'linear_score_encoding': sigma_linear_score,
        'linear_output': modified_sigma_linear_score
        }

        store_in_msgpack_file(prompt_data, prompt_index, minio_client, embedding_type)
        prompt_index+=1

    store_in_csv_file(csv_data, minio_client, embedding_type)

def load_dataset(minio_client, embedding_type):
    dataset_path=DATA_MINIO_DIRECTORY + f"/{embedding_type}_prompts/"
    dataset_files=minio_client.list_objects('datasets', prefix=dataset_path, recursive=True)
    dataset_files= [file.object_name for file in dataset_files]
    print(len(dataset_files))

    self_training_data= get_self_training_paths(minio_client)
    print(len(self_training_data))

    dataset_files= dataset_files + self_training_data
    
    elm_inputs=[]
    linear_inputs=[]
    
    elm_sigma_outputs=[]
    elm_binary_outputs=[]

    linear_sigma_outputs=[]
    linear_binary_outputs=[]

    for file in dataset_files:
        print(file)
        # get prompt embedding
        data = minio_client.get_object('datasets', file)
        # Read the content of the msgpack file
        content = data.read()

        # Deserialize the content using msgpack
        msgpack_data = msgpack.loads(content)

        if(msgpack_data["elm_output"]!=""):
            # get elm input
            elm_inputs.append(np.concatenate([msgpack_data['input'],
                                            [msgpack_data['position_encoding']],
                                        [msgpack_data['elm_score_encoding']]]))
            
            # get sigma output
            elm_sigma_outputs.append(msgpack_data['elm_output'])

            # get binary input
            if msgpack_data['elm_score_encoding']> msgpack_data['elm_output'] :
                binary_elm_output="decrease"
            else:
                binary_elm_output="increase"

            elm_binary_outputs.append(binary_elm_output)

        if(msgpack_data["linear_output"]!=""):
            # get linear input
            linear_inputs.append(np.concatenate([msgpack_data['input'],
                                                [msgpack_data['position_encoding']],
                                        [msgpack_data['linear_score_encoding']]]))
            
            # get sigma output
            linear_sigma_outputs.append(msgpack_data['linear_output'])

            # get binary input
            if msgpack_data['linear_score_encoding']> msgpack_data['linear_output'] :
                binary_linear_output="decrease"
            else:
                binary_linear_output="increase"

            linear_binary_outputs.append(binary_linear_output)
        

    return elm_inputs, linear_inputs, elm_sigma_outputs, elm_binary_outputs, linear_sigma_outputs, linear_binary_outputs     


def fix_dataset(minio_client):
    self_training_data= get_self_training_paths(minio_client)
    self_training_data= self_training_data[122539:]
    index=1
    for file in self_training_data:
        print(file)
        # get prompt embedding
        data = minio_client.get_object('datasets', file)
        # Read the content of the msgpack file
        content = data.read()

        # Deserialize the content using msgpack
        msgpack_data = msgpack.loads(content)

        print(len(msgpack_data['input'][0]))

        # Append to the msgpack data list
        prompt_data={
            'input': msgpack_data['input'][0],
            'position_encoding': msgpack_data['position_encoding'],
            'elm_score_encoding': msgpack_data['elm_score_encoding'],
            'elm_output': msgpack_data['elm_output'],
            'linear_score_encoding': msgpack_data['linear_score_encoding'],
            'linear_output': msgpack_data['linear_output']
        }

        packed_data = msgpack.packb(prompt_data, use_single_float=True)

        # Define the local directory path for embedding
        local_directory = 'output/prompt_mutator/data/'

        # Ensure the local directory exists, create it if necessary
        os.makedirs(local_directory, exist_ok=True)

        # Create a local file with the packed data
        local_file_path = local_directory + f"{str(index).zfill(6)}_substitution.msgpack"
        with open(local_file_path, 'wb') as local_file:
            local_file.write(packed_data)
        
        # Read the contents of the CSV file
        with open(local_file_path, 'rb') as file:
            content = file.read()

        # Upload the local file to MinIO
        buffer = io.BytesIO(content)
        buffer.seek(0)

        minio_path=DATA_MINIO_DIRECTORY + f"/self_training/{str(index).zfill(6)}_substitution.msgpack"
        cmd.upload_data(minio_client, 'datasets',minio_path, buffer)

        # Remove the temporary file
        os.remove(local_file_path)

        index+=1

def main():
    args = parse_args()

    # get device
    if torch.cuda.is_available():
            device = 'cuda'
    else:
        device = 'cpu'
    device = torch.device(device)

    # get minio client
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)
    
    if args.create_dataset:
        create_dataset(minio_client, device, args.csv_phrase, args.embedding_type)

    fix_dataset(minio_client)

    # elm_inputs, linear_inputs, elm_sigma_outputs, elm_binary_outputs, linear_sigma_outputs, linear_binary_outputs =load_dataset(minio_client, args.embedding_type)

    # # prompt mutator for predicting binary classes (increase, decrease) wth elm scores and linear scores
    # elm_binary_mutator= BinaryPromptMutator(minio_client=minio_client)
    # elm_binary_mutator.train(elm_inputs, elm_binary_outputs)
    # elm_binary_mutator.save_model()
    
    # linear_binary_mutator= BinaryPromptMutator(minio_client=minio_client, ranking_model="linear")
    # linear_binary_mutator.train(linear_inputs, linear_binary_outputs)
    # linear_binary_mutator.save_model()

    # #prompt mutator for predicting sigma scores for elm and linear scores
    # elm_sigma_mutator= PromptMutator(minio_client=minio_client)
    # elm_sigma_mutator.train(elm_inputs, elm_sigma_outputs)
    # elm_sigma_mutator.save_model()
    
    # linear_sigma_mutator= PromptMutator(minio_client=minio_client, ranking_model="linear")
    # linear_sigma_mutator.train(linear_inputs, linear_sigma_outputs)
    # linear_sigma_mutator.save_model()

if __name__ == "__main__":
    main()
