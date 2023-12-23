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

DATA_MINIO_DIRECTORY="environmental/data/prompt-generator/"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--csv-phrase', help='CSV containing phrases, must have "phrase str" column', default='input/civitai_phrases_database_v7_no_nsfw.csv')
    parser.add_argument('--embedding-type', help='type of embedding, positive or negative', default='positive')
    parser.add_argument('--create-dataset', help='whether to create a new dataset or load existing one', default=False)
    parser.add_argument('--operation', help='operation to train mutator on (substitution, permutation..)', default="substitution")
    parser.add_argument('--output-type', help='type of output for the prompt mutator model', default="sigma_score")
    parser.add_argument('--scoring-model', help="scoring model to do self training on (elm,linear etc..)", default="linear")
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

    embedding_model.load_pth(byte_buffer)
    embedding_model.model=embedding_model.model.to(device)

    return embedding_model

def get_embedding_paths(minio_client, dataset):
    objects=minio_client.list_objects('datasets', dataset, recursive=True)
    embedding_files = []
    for obj in objects: 
        if obj.object_name.endswith("_embedding.msgpack"):
            embedding_files.append(obj.object_name)
            
    return embedding_files

def get_self_training_paths(minio_client, operation):
    # get minio paths for existing self training data
    dataset_path=DATA_MINIO_DIRECTORY + f"{operation}/self_training/"
    dataset_files=minio_client.list_objects('datasets', prefix=dataset_path, recursive=True)
    dataset_files= [file.object_name for file in dataset_files]
    
    return dataset_files

def store_in_csv_file(csv_data, minio_client, embedding_type, operation):
    # Save data to a CSV file
    csv_file = f'output/prompt_{operation}_dataset.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        if(operation=="substitution"):
            writer.writerow(['Prompt', 'Substitute Phrase', 'Substituted Phrase', 
                         'Substitution Position', 'Original_elm_score', 'New_elm_score',
                         'Original_linear_score', 'New_linear_score'])
        elif(operation=="permutation"):
            writer.writerow(['Prompt', 'First Phrase', 'Second Phrase', 
                         'First Position', 'Second Position', 'Original_elm_score', 'New_elm_score',
                         'Original_linear_score', 'New_linear_score'])
        elif(operation=="addition"):
            writer.writerow(['Prompt', 'Added Phrase','Addition Position', 
                            'Original_elm_score', 'New_elm_score',
                            'Original_linear_score', 'New_linear_score'])
        # Write the data
        writer.writerows(csv_data)
    
    # Read the contents of the CSV file
    with open(csv_file, 'rb') as file:
        csv_content = file.read()

    #Upload the CSV file to Minio
    buffer = io.BytesIO(csv_content)
    buffer.seek(0)

    minio_path = DATA_MINIO_DIRECTORY+ f'{operation}/{embedding_type}_dataset.csv'
    cmd.upload_data(minio_client, 'datasets', minio_path, buffer)

def store_in_msgpack_file(prompt_data, index, minio_client, embedding_type, operation):
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

    minio_path=DATA_MINIO_DIRECTORY + f"{operation}/{embedding_type}_prompts/{str(index).zfill(6)}_substitution.msgpack"
    cmd.upload_data(minio_client, 'datasets',minio_path, buffer)

def create_permutation_dataset(minio_client, device, embedding_type):
    # Load the CLIP model
    clip=CLIPTextEmbedder(device=device)
    clip.load_submodels()

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

        
        prompt_list = prompt_str.split(', ')
        if(len(prompt_list)<2):
            continue
        # Choose two random positions in the prompt to do permutation
        random_numbers = random.sample(range(len(prompt_list)), 2)

        position1, position2= random_numbers[0], random_numbers[1]

        print(position1, position2)

        # Calculate text embedding of the two phrases to be permutated
        phrase1=prompt_list[position1]
        with torch.no_grad():
            phrase1_embedding= clip.forward(phrase1).unsqueeze(0)
        
        phrase2=prompt_list[position2]
        with torch.no_grad():
            phrase2_embedding= clip.forward(phrase2).unsqueeze(0)

        prompt_list[position1] = phrase2
        prompt_list[position2] = phrase1
        modified_prompt = ", ".join(prompt_list)

        # Get embedding of mutated prompt
        with torch.no_grad():
            modified_embedding= clip.forward(modified_prompt)
            modified_embedding= modified_embedding.unsqueeze(0)
            modified_embedding=modified_embedding.to(device)

        # get score before and after permutation
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
        pooled_phrase1_embedding=torch.mean(phrase1_embedding, dim=2)
        #flattening embedding
        pooled_phrase1_embedding = pooled_phrase1_embedding.reshape(len(pooled_phrase1_embedding), -1).squeeze(0)
        
        # mean pooling
        pooled_phrase2_embedding=torch.mean(phrase2_embedding, dim=2)
        #flattening embedding
        pooled_phrase2_embedding = pooled_phrase2_embedding.reshape(len(pooled_phrase2_embedding), -1).squeeze(0)

        # calculate sigma scores
        sigma_elm_score=(elm_prompt_score - elm_mean) / elm_std
        modified_sigma_elm_score= (elm_modified_pormpt_score - elm_mean) / elm_std
        sigma_linear_score= (linear_prompt_score - linear_mean) / linear_std
        modified_sigma_linear_score= (linear_modified_pormpt_score - linear_mean) / linear_std 

        # Append to the CSV data list
        csv_data.append([
            prompt_str,  # Prompt string
            phrase1,        # first phrase string
            phrase2,  # second phrase string
            position1,   # posiiton of first phrase
            position2,   # posiiton of second phrase
            sigma_elm_score, # elm score before permutation
            modified_sigma_elm_score, # elm score after permutation
            sigma_linear_score, # linear score before permutation
            modified_sigma_linear_score # linear score after permutation
        ])

        # Append to the msgpack data list
        prompt_data={
        'input': torch.cat([pooled_prompt_embedding, pooled_phrase1_embedding, pooled_phrase2_embedding], dim=0).tolist(),
        'first_position': position1,
        'second_position': position2,
        'elm_score_encoding': sigma_elm_score,
        'elm_output': modified_sigma_elm_score,
        'linear_score_encoding': sigma_linear_score,
        'linear_output': modified_sigma_linear_score
        }

        store_in_msgpack_file(prompt_data, prompt_index, minio_client, embedding_type, 'permutation')
        prompt_index+=1

    store_in_csv_file(csv_data, minio_client, embedding_type, "permutation")  

def create_substitution_dataset(minio_client, device, csv_path, embedding_type):
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

        store_in_msgpack_file(prompt_data, prompt_index, minio_client, embedding_type, "substitution")
        prompt_index+=1

    store_in_csv_file(csv_data, minio_client, embedding_type, "substitution")

def create_addition_dataset(minio_client, device, csv_path, embedding_type):
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
        added_phrase = random.choice(phrases_df['phrase str'].tolist())
        with torch.no_grad():
                added_embedding= clip.forward(added_phrase).unsqueeze(0)
        
        prompt_list = prompt_str.split(', ')
        # Choose a random position to add to the prompt
        addition_position = random.randint(0, len(prompt_list))

        prompt_list.insert(addition_position, added_phrase)
        modified_prompt = ", ".join(prompt_list)

        # Get embedding of mutated prompt
        with torch.no_grad():
            modified_embedding= clip.forward(modified_prompt)
            modified_embedding= modified_embedding.unsqueeze(0)
            modified_embedding=modified_embedding.to(device)

        # get score before and after addition
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
        pooled_added_embedding=torch.mean(added_embedding, dim=2)
        #flattening embedding
        pooled_added_embedding = pooled_added_embedding.reshape(len(pooled_added_embedding), -1).squeeze(0)

        # calculate sigma scores
        sigma_elm_score=(elm_prompt_score - elm_mean) / elm_std
        modified_sigma_elm_score= (elm_modified_pormpt_score - elm_mean) / elm_std
        sigma_linear_score= (linear_prompt_score - linear_mean) / linear_std
        modified_sigma_linear_score= (linear_modified_pormpt_score - linear_mean) / linear_std 

        # Append to the CSV data list
        csv_data.append([
            prompt_str,  # Prompt string
            added_phrase,        # added phrase string
            addition_position,   # addition position
            sigma_elm_score, # elm score before addition
            modified_sigma_elm_score, # elm score after addition
            sigma_linear_score, # linear score before addition
            modified_sigma_linear_score # linear score after addition
        ])

        # Append to the msgpack data list
        prompt_data={
        'input': torch.cat([pooled_prompt_embedding, pooled_added_embedding], dim=0).tolist(),
        'position_encoding': addition_position/len(prompt_list),
        'elm_score_encoding': sigma_elm_score,
        'elm_output': modified_sigma_elm_score,
        'linear_score_encoding': sigma_linear_score,
        'linear_output': modified_sigma_linear_score
        }

        store_in_msgpack_file(prompt_data, prompt_index, minio_client, embedding_type, "addition")
        prompt_index+=1

    store_in_csv_file(csv_data, minio_client, embedding_type, "addition")    

def load_dataset(minio_client, embedding_type, output_type, scoring_model, operation):
    dataset_path=DATA_MINIO_DIRECTORY + f"{operation}/{embedding_type}_prompts/"
    dataset_files=minio_client.list_objects('datasets', prefix=dataset_path, recursive=True)
    dataset_files= [file.object_name for file in dataset_files]
    
    inputs=[]
    outputs=[]

    for file in dataset_files:
        print(file)
        # get prompt embedding
        data = minio_client.get_object('datasets', file)
        # Read the content of the msgpack file
        content = data.read()

        # Deserialize the content using msgpack
        msgpack_data = msgpack.loads(content)

        # get input
        if operation=="substitution":
            input=np.concatenate([msgpack_data['input'],
                                            [msgpack_data['position_encoding']],
                                        [msgpack_data[f'{scoring_model}_score_encoding']]])
        elif operation=="addition":
            input=np.concatenate([msgpack_data['input'], [msgpack_data['position_encoding']]])
        elif operation=="permutation":
            input=np.concatenate([msgpack_data['input'],
                                            [msgpack_data['first_position']],
                                            [msgpack_data['second_position']],
                                    [msgpack_data[f'{scoring_model}_score_encoding']]])
            
        inputs.append(input)
        
        if(output_type=="binary"):
            # get binary output
            if msgpack_data[f'{scoring_model}_score_encoding']> msgpack_data[f'{scoring_model}_output'] :
                binary_linear_output="decrease"
            else:
                binary_linear_output="increase"

            outputs.append(binary_linear_output)
        
        elif(output_type=="sigma_score"):
            # get sigma output
            sigma_score=msgpack_data[f'{scoring_model}_output']
            outputs.append(sigma_score)
        elif(output_type=="delta_score"):
            # get delta output
            delta_score= msgpack_data[f'{scoring_model}_output'] - msgpack_data[f'{scoring_model}_score_encoding']
            outputs.append(delta_score)
        
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
    
    if args.create_dataset:
        if args.operation=="substitution":
            create_substitution_dataset(minio_client, device, args.csv_phrase, args.embedding_type)
        elif args.operation=="permutation":
            create_permutation_dataset(minio_client, device, args.embedding_type)
        elif args.operation=="addition":
            create_addition_dataset(minio_client, device, args.csv_phrase, args.embedding_type)

    inputs, outputs= load_dataset(minio_client=minio_client, 
                                  embedding_type=args.embedding_type, 
                                  output_type=args.output_type,
                                  scoring_model=args.scoring_model,
                                  operation=args.operation)

    if(args.output_type=="binary"):
        model= BinaryPromptMutator(minio_client=minio_client, ranking_model=args.scoring_model, operation=args.operation, prompt_type=args.embedding_type)
    else:
        model= PromptMutator(minio_client=minio_client, output_type=args.output_type, ranking_model=args.scoring_model, operation=args.operation, prompt_type=args.embedding_type)

    model.train(inputs, outputs)
    model.save_model()

if __name__ == "__main__":
    main()
