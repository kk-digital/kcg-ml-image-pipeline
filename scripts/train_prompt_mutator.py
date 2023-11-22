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

DATA_MINIO_DIRECTORY="environmental/data/prompt-generator/substitution"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="123.176.98.90:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--csv-phrase', help='CSV containing phrases, must have "phrase str" column', default='input/civitai_phrases_database_v7_no_nsfw.csv')

    args = parser.parse_args()
    return args

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

def get_embedding_paths(minio_client, dataset):
    objects=minio_client.list_objects('datasets', dataset, recursive=True)
    embedding_files = []
    for obj in objects: 
        if obj.object_name.endswith("_embedding.msgpack"):
            embedding_files.append(obj.object_name)
            
    return embedding_files

def store_in_csv_file(csv_data, minio_client):
    # Save data to a CSV file
    csv_file = 'output/prompt_substitution_dataset.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Prompt', 'Substitute Phrase', 'Substituted Phrase', 'Substitution Position', 'Original Score', 'New Score'])
        # Write the data
        writer.writerows(csv_data)
    
    # Read the contents of the CSV file
    with open(csv_file, 'rb') as file:
        csv_content = file.read()

    #Upload the CSV file to Minio
    buffer = io.BytesIO(csv_content)
    buffer.seek(0)

    model_path = DATA_MINIO_DIRECTORY+ '/data_csv/dataset.csv'
    cmd.upload_data(minio_client, 'datasets', model_path, buffer)

def store_in_msgpack_file(prompt_data, index, minio_client):
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

    cmd.upload_data(minio_client, 'datasets', DATA_MINIO_DIRECTORY, buffer)

# function to randomly select one phrase from each bin
def random_phrase_from_each_bin(group):
    return random.choice(group.tolist())

def create_dataset(minio_client, device, csv_path):
    # Load the CLIP model
    clip=CLIPTextEmbedder(device=device)
    clip.load_submodels()

    # get dataset of phrases
    phrases_df = pd.read_csv(csv_path)
    # get minio paths for embeddings
    embedding_paths = get_embedding_paths(minio_client, "environmental")
    # get ranking mondel
    elm_model= load_model(768,minio_client, device)

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
        prompt_str=msgpack_data['positive_prompt']
        prompt_embedding= list(msgpack_data['positive_embedding'].values())
        prompt_embedding = torch.tensor(np.array(prompt_embedding)).float()
        prompt_embedding=prompt_embedding.to(device)

        #Randomly select a phrase from the dataset and get an embedding
        substitute_phrase = random.choice(phrases_df['phrase str'].tolist())
        with torch.no_grad():
                substitute_embedding= clip.forward(substitute_phrase).unsqueeze(0)

        # Choose a random position to substitute in the prompt
        position_to_substitute = random.randint(0, len(prompt_str.split(',')) - 1)

        # Create a modified prompt with the substitution and get embedding of substituted phrase
        prompt_list = prompt_str.split(', ')
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
            prompt_score=elm_model.predict_positive_or_negative_only(prompt_embedding)
            modified_pormpt_score= elm_model.predict_positive_or_negative_only(modified_embedding)
        
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

        # Append to the CSV data list
        csv_data.append([
            prompt_str,  # Prompt string
            substitute_phrase,        # Substitute phrase string
            substituted_phrase,  # Substituted phrase string
            position_to_substitute,   # Substitution position
            prompt_score.item(), # score before substitution
            modified_pormpt_score.item() # score after substitution
        ])

        # Append to the msgpack data list
        prompt_data={
        'input': torch.cat([pooled_prompt_embedding, pooled_substituted_embedding, pooled_substitute_embedding], dim=0).tolist(),
        'position_encoding': position_to_substitute,
        'score_encoding': prompt_score.item(),
        'output': modified_pormpt_score.item()
        }

        store_in_msgpack_file(prompt_data, prompt_index, minio_client)
        prompt_index+=1

    store_in_csv_file(csv_data, minio_client)

def load_dataset(minio_client):
    dataset_files=minio_client.list_objects('datasets', prefix=DATA_MINIO_DIRECTORY)
    dataset_files= [file.object_name for file in dataset_files]

    inputs=[]
    outputs=[]
    score_encoding= []
    position_encoding= []

    for file in dataset_files:
        print(file)
        # get prompt embedding
        data = minio_client.get_object('datasets', file)
        # Read the content of the msgpack file
        content = data.read()

        # Deserialize the content using msgpack
        msgpack_data = msgpack.loads(content)

        # get input and output
        inputs.append(msgpack_data['input'])
        position_encoding.append(msgpack_data['position_encoding'])
        score_encoding.append(msgpack_data['score_encoding'])
        outputs.append(msgpack_data['output'])
        

    #compute sigma scores for initial score encoding
    sigma_mean=np.mean(score_encoding)
    sigma_std=np.std(score_encoding)
    score_encoding = [(x - sigma_mean) / sigma_std for x in score_encoding]
    #compute sigma scores for output
    outputs = [(x - sigma_mean) / sigma_std for x in outputs]

    return inputs, position_encoding, score_encoding, outputs

def load_classification_dataset(minio_client):
    dataset_files=minio_client.list_objects('datasets', prefix=DATA_MINIO_DIRECTORY)
    dataset_files= [file.object_name for file in dataset_files]

    inputs=[]
    outputs=[]

    for file in dataset_files:
        # get prompt embedding
        data = minio_client.get_object('datasets', file)
        # Read the content of the msgpack file
        content = data.read()

        # Deserialize the content using msgpack
        msgpack_data = msgpack.loads(content)

        # get input and output
        if msgpack_data['score_encoding']> msgpack_data['output'] :
            output="decrease"
        else:
            output="increase"
        
        outputs.append(output)
        inputs.append(np.concatenate([msgpack_data['input'], 
                                      [msgpack_data['position_encoding']],
                                       [msgpack_data['score_encoding']]]))

        print(file)
        

    return inputs, outputs        
        

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
    
    create_dataset(minio_client, device, args.csv_phrase)

    inputs, position_encoding, score_encoding, outputs =load_dataset(minio_client)
    binary_inputs, binary_outputs =load_classification_dataset(minio_client)

    # prompt mutator for predicting binary classes (increase, decrease)
    binary_mutator= MulticlassPromptMutator(minio_client=minio_client)
    binary_mutator.train(binary_inputs, binary_outputs)
    binary_mutator.save_model()

    #prompt mutator with both position encoding and initial score encoding
    sigma_mutator= PromptMutator(minio_client=minio_client, output_type="sigma_score")
    sigma_mutator.train(inputs, 
                        position_encoding, 
                        score_encoding,
                        outputs
                        )
    sigma_mutator.save_model()
    
    # input, delta_output, sigma_output, binary_output = load_dataset(minio_client, device)

    # prompt mutator for predicting delta scores
    # delta_mutator= PromptMutator(minio_client=minio_client, output_type="delta_score")
    # delta_mutator.train(inputs, 
    #                     position_encoding,
    #                     score_encoding,
    #                     outputs)
    # delta_mutator.save_model()
    
    
    # grid search for hyperparameters
    #last params {'objective': 'reg:squarederror', 'alpha': 0.0, 'lambda': 0.0, 'max_depth': 7, 'min_child_weight': 1, 'gamma': 0.0, 'subsample': 1, 'colsample_bytree': 1, 'eta': 0.05}
    # params = {
    # 'max_depth': [5,7,10],
    # 'min_child_weight': [1],
    # 'gamma': [0.0, 0.01, 0.05, 0.1],
    # 'eta': [0.05],
    # }

    # best_params, best_score= mutator.grid_search(X_train=input, y_train=output, param_grid=params)
    # print("Best Parameters: ", best_params)
    # print("Best Score: ", best_score)

if __name__ == "__main__":
    main()
