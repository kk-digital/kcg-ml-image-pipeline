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

    parser.add_argument('--alpha', required=False, default=0.8, help='regularisation term')
    parser.add_argument('--learning-rate', required=False, default=0.1, help='learning rate for model')
    parser.add_argument('--max-depth', required=False, default=10, help='max depth of decision trees')
    parser.add_argument('--min-child_weight', required=False, default=1, help='controls minimum weight of features')
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

def load_dataset(minio_client, device, output_type="delta_score"):
    # Load the CLIP model
    clip=CLIPTextEmbedder()
    clip.load_submodels()

    # Load the dataset of phrases
    phrases_df = pd.read_csv('input/environment_phrase_scores.csv')  # Change this to your phrases dataset
    prompts_df = pd.read_csv('input/environment_data.csv')
    
    elm_model= load_model(768,minio_client, device)

    # Create empty lists to store the generated data
    input_features = []
    output_scores = []
    csv_data = []

    for index, prompt in prompts_df.iterrows():
        print(f"prompt {index}")

        # get prompt embedding
        data = minio_client.get_object('datasets', prompt['file_path'])
        # Read the content of the msgpack file
        content = data.read()

        # Deserialize the content using msgpack
        msgpack_data = msgpack.loads(content)

        # get prompt embedding 
        prompt_embedding= list(msgpack_data['positive_embedding'].values())
        prompt_embedding = torch.tensor(np.array(prompt_embedding)).float()
        prompt_embedding=prompt_embedding.to(device)

        # Randomly select a phrase from the dataset and get an embedding
        substitute_phrase = random.choice(phrases_df['phrase'].tolist())

        # Choose a random position to substitute in the prompt
        position_to_substitute = random.randint(0, len(prompt['positive_prompt'].split(',')) - 1)

        # Create a modified prompt with the substitution
        prompt_list = prompt['positive_prompt'].split(',')
        substituted_phrase=prompt_list[position_to_substitute]
        with torch.no_grad():
            sub_phrase_embedding= clip(substituted_phrase).unsqueeze(0)

        prompt_list[position_to_substitute] = substitute_phrase
        modified_prompt = " ".join(prompt_list)

        # Get embeddings of mutated promp
        with torch.no_grad():
            modified_embedding= clip(modified_prompt)
            modified_embedding= modified_embedding.unsqueeze(0)
            modified_embedding=modified_embedding.to(device)

        # Calculate the delta score
        with torch.no_grad():
            prompt_score=elm_model.predict_positive_or_negative_only(prompt_embedding)
            modified_pormpt_score= elm_model.predict_positive_or_negative_only(modified_embedding)

        if(output_type=="delta_score"):
            delta_score= modified_pormpt_score - prompt_score
        elif(output_type=="multi_class"):
            category=get_category(modified_pormpt_score, prompt_score)
        
        prompt_embedding=torch.mean(prompt_embedding, dim=2)
        prompt_embedding = prompt_embedding.reshape(len(prompt_embedding), -1).squeeze(0)
        
        sub_phrase_embedding=torch.mean(sub_phrase_embedding, dim=2)
        sub_phrase_embedding = sub_phrase_embedding.reshape(len(sub_phrase_embedding), -1).squeeze(0)

        # Convert the position to a tensor
        position_tensor = torch.tensor([position_to_substitute]).float().to(device)

        # Append to the input and output lists
        input_features.append(torch.cat([prompt_embedding, sub_phrase_embedding, position_tensor], dim=0).detach().cpu().numpy())
        if(output_type=="delta_score"):
            output_scores.append(delta_score.item())
        elif(output_type=="multi_class"):
            output_scores.append(category)


        # Append to the CSV data list
        csv_data.append([
            prompt['positive_prompt'],  # Prompt string
            substitute_phrase,        # Substitute phrase string
            substituted_phrase,  # Substituted phrase string
            position_to_substitute,   # Substitution position
            delta_score.item()        # Delta score
        ])
    
    # Save data to a CSV file
    csv_file = 'output/prompt_substitution_dataset.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Prompt', 'Substitute Phrase', 'Substituted Phrase', 'Substitution Position', 'Delta Score'])
        # Write the data
        writer.writerows(csv_data)
    
    # Read the contents of the CSV file
    with open(csv_file, 'rb') as file:
        csv_content = file.read()

    #Upload the CSV file to Minio
    buffer = io.BytesIO(csv_content)
    buffer.seek(0)

    if(output_type=="delta_score"):
        path='output/prompt_mutator/dataset.csv'
    elif(output_type=="multi_class"):
        path='output/prompt_mutator/multi_class_dataset.csv'

    model_path = os.path.join('environmental', path)
    cmd.upload_data(minio_client, 'datasets', model_path, buffer)

    return np.array(input_features), np.array(output_scores)

def get_category(modified_score, old_score):
    change= modified_score/old_score

    if(change>0.99 and change<1.01):
        category="no change"
    elif(change>0.9 and change<1.1):
        category="low increase" if change>0 else "low decrease"
    else:
        category="high increase" if change>0 else "high decrease"
    
    return category

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
    
    input, output = load_dataset(minio_client, device, output_type="multi_class")

    mutator= MulticlassPromptMutator(minio_client=minio_client)
    mutator.train(input, output)
    mutator.save_model(local_path="output/multiclass_prompt_mutator.json" , 
                       minio_path="environmental/output/prompt_mutator/multiclass_prompt_mutator.json")

    # mutator= PromptMutator(minio_client=minio_client)
    
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

    # mutator.train(input, output,
    #               gamma=best_params['gamma'], 
    #               max_depth=best_params['max_depth'],
    #               min_child_weight=best_params['min_child_weight'],
    #               eta= best_params['eta']
    #             )
    # mutator.save_model(local_path="output/prompt_mutator.json" , 
    #                    minio_path="environmental/output/prompt_mutator/prompt_mutator.json")

if __name__ == "__main__":
    main()
