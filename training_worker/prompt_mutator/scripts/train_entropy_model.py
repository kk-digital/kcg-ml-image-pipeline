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
from stable_diffusion.model.clip_text_embedder.clip_text_embedder import CLIPTextEmbedder
from utility.ensemble.ensemble_helpers import Binning, SigmaScoresWithEntropy
from utility.minio import cmd

DATA_MINIO_DIRECTORY="environmental/data/prompt-generator/entropy/"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--csv-phrase', help='CSV containing phrases, must have "phrase str" column', default='input/civitai_phrases_database_v7_no_nsfw.csv')
    parser.add_argument('--embedding-type', help='type of embedding, positive or negative', default='positive')
    parser.add_argument('--create-dataset', action='store_true', help='whether to create a new dataset or load existing one', default=False)
    parser.add_argument('--operation', help='operation to train mutator on (substitution, permutation..)', default="substitution")
    parser.add_argument('--output-type', help='type of output for the prompt mutator model', default="entropy")
    args = parser.parse_args()
    return args

class EntropyDatasetLoader:
    def __init__(
        self,
        minio_access_key,
        minio_secret_key,
        minio_ip_addr,
        csv_phrase,
        embedding_type,
        operation,
        output_type,
        create_dataset,
        step=1,
        bins=8
    ):
        
        # get minio client
        self.minio_client = cmd.get_minio_client(minio_access_key,
                                            minio_secret_key,
                                            minio_ip_addr)

        self.csv_phrase=csv_phrase
        self.embedding_type= embedding_type
        self.operation= operation
        self.output_type= output_type
        self.create_dataset= create_dataset
        self.step= step
        self.bins= bins

        # get device
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        self.device = torch.device(device)

        # Load the clip embedder model
        self.embedder=CLIPTextEmbedder(device=device)
        self.embedder.load_submodels()

        # get ensemble elm models
        self.ensemble_models=self.get_ensemble_models()

    def get_ensemble_models(self):
        input_path = "environmental/models/ranking/"

        model_class = ABRankingELMModel

        # Get all model files
        model_files = cmd.get_list_of_objects_with_prefix(self.minio_client, 'datasets', input_path)

        # Filter relevant model files
        relevant_models = [
            model_file for model_file in model_files
            if model_file.endswith(f"score-elm-v1-embedding-{self.embedding_type}.pth")
        ]

        # Sort the model files by timestamp (assuming the file names include a timestamp)
        relevant_models=np.flip(relevant_models)

        # Load the latest num_models models
        loaded_models = []
        for i in range(min(16, len(relevant_models))):
            most_recent_model = relevant_models[i]

            # Get the model file data
            model_file_data = cmd.get_file_from_minio(self.minio_client, 'datasets', most_recent_model)

            # Create a BytesIO object and write the downloaded content into it
            byte_buffer = io.BytesIO()
            for data in model_file_data.stream(amt=8192):
                byte_buffer.write(data)
            # Reset the buffer's position to the beginning
            byte_buffer.seek(0)

            # Load the model
            embedding_model = model_class(768)
            embedding_model.load_pth(byte_buffer)
            embedding_model.model=embedding_model.model.to(self.device)

            loaded_models.append(embedding_model)

        return loaded_models

    def store_in_csv_file(self, data):
        # Save data to a CSV file
        csv_file = f'output/prompt_{self.operation}_dataset.csv'
        pd.DataFrame(data).to_csv(csv_file, index=False)
        
        # Read the contents of the CSV file
        with open(csv_file, 'rb') as file:
            csv_content = file.read()

        #Upload the CSV file to Minio
        buffer = io.BytesIO(csv_content)
        buffer.seek(0)

        minio_path = DATA_MINIO_DIRECTORY+ f'{self.operation}/{self.embedding_type}_dataset.csv'
        cmd.upload_data(self.minio_client, 'datasets', minio_path, buffer)

    def store_batch_in_msgpack_file(self, batch, index):
        file_path=f"{self.operation}/{self.embedding_type}_prompts/{str(index).zfill(4)}_substitution.msgpack"
        packed_data = msgpack.packb(batch, use_single_float=True)

        local_file_path = f"output/temporary_file.msgpack"
        with open(local_file_path, 'wb') as local_file:
            local_file.write(packed_data)

        with open(local_file_path, 'rb') as file:
            content = file.read()

        buffer = io.BytesIO(content)
        buffer.seek(0)

        minio_path = DATA_MINIO_DIRECTORY + file_path
        cmd.upload_data(self.minio_client, 'datasets', minio_path, buffer)

        os.remove(local_file_path)

    def get_embedding_paths(self, dataset):
        objects=self.minio_client.list_objects('datasets', dataset, recursive=True)
        embedding_files = []
        for obj in objects: 
            if obj.object_name.endswith("_embedding.msgpack"):
                embedding_files.append(obj.object_name)
                
        return embedding_files

    def get_prompt_embedding(self, prompt):
        with torch.no_grad():
            embedding= self.embedder(prompt)

        embedding= embedding.unsqueeze(0)
        embedding=embedding.to(self.device)

        return embedding

    def get_ensemble_sigma_scores(self, embedding):
        sigma_scores=[]
        for model in self.ensemble_models:
            mean=model.mean
            std=model.standard_deviation
            with torch.no_grad():
                score=model.predict_positive_or_negative_only(embedding).item()
                score=(score - mean)/std
            
            sigma_scores.append(score)
        
        return np.array(sigma_scores)
        
    def get_prompt_entropy(self, embedding):
        sigma_scores=self.get_ensemble_sigma_scores(embedding)

        # get entropy classes
        binning= Binning(start=-6,count=self.bins,step=self.step)
        entropy_data=SigmaScoresWithEntropy(sigma_scores, binning)

        # get entropy, variance and average
        entropy= entropy_data.entropy
        variance= entropy_data.variance
        mean= entropy_data.mean

        return entropy, variance, mean

    def create_substitution_dataset(self):
        # get dataset of phrases
        phrases_df = pd.read_csv(self.csv_phrase)
        # get minio paths for embedding
        embedding_paths = self.get_embedding_paths("environmental")

        prompt_index=1
        csv_data = []
        batch=[]

        for embedding in embedding_paths:
            print(f"prompt {prompt_index}")

            # get prompt embedding
            data = self.minio_client.get_object('datasets', embedding)
            # Read the content of the msgpack file
            content = data.read()

            # Deserialize the content using msgpack
            msgpack_data = msgpack.loads(content)

            # get prompt embedding 
            prompt_str=msgpack_data[f'{self.embedding_type}_prompt']
            prompt_embedding= list(msgpack_data[f'{self.embedding_type}_embedding'].values())
            prompt_embedding = torch.tensor(np.array(prompt_embedding)).float()
            prompt_embedding=prompt_embedding.to(self.device)

            #Randomly select a phrase from the dataset and get an embedding
            substitute_phrase = random.choice(phrases_df['phrase str'].tolist())
            substitute_embedding= self.get_prompt_embedding(substitute_phrase)
            
            prompt_list = prompt_str.split(', ')
            # Choose a random position to substitute in the prompt
            position_to_substitute = random.randint(0, len(prompt_list) - 1)

            # Create a modified prompt with the substitution and get embedding of substituted phrase
            substituted_phrase=prompt_list[position_to_substitute]
            substituted_embedding= self.get_prompt_embedding(substituted_phrase)

            prompt_list[position_to_substitute] = substitute_phrase
            modified_prompt = ", ".join(prompt_list)

            # Get embedding of mutated prompt
            modified_embedding= self.get_prompt_embedding(modified_prompt)

            # get entropy before and after substitution
            entropy= self.get_prompt_entropy(prompt_embedding)
            modified_entropy= self.get_prompt_entropy(modified_embedding)
            
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
            csv_data.append({
                'prompt_str':prompt_str,  # Prompt string
                'substitute phrase': substitute_phrase,        # Substitute phrase string
                'substituted phrase':substituted_phrase,  # Substituted phrase string
                'position':position_to_substitute,   # Substitution position
                'entropy': entropy,
                'new entropy': modified_entropy
            })

            # Append to the msgpack data list
            batch.append({
            'input': torch.cat([pooled_prompt_embedding, pooled_substituted_embedding, pooled_substitute_embedding], dim=0).tolist(),
            'position_encoding': position_to_substitute,
            'initial_entropy':entropy,
            'output':modified_entropy,
            })
            
            if len(batch) == 10000:
                self.store_batch_in_msgpack_file(batch, prompt_index)
                prompt_index+=1
                batch = []  # Reset the batch for the next file

        self.store_in_csv_file(csv_data)

    def load_substitution_dataset(self):
        # get self training data
        minio_path = DATA_MINIO_DIRECTORY + f"{self.operation}/{self.embedding_type}_prompts/"
        files = self.minio_client.list_objects('datasets', prefix=minio_path, recursive=True)
        files = [file.object_name for file in files]

        inputs=[]
        outputs=[]
        
        for file in files:
            # get data
            data = self.minio_client.get_object('datasets', file)
            # Read the content of the msgpack file
            content = data.read()

            # Deserialize the content using msgpack
            entropy_data = msgpack.loads(content)
    
            for d in entropy_data:
                input=np.concatenate([d['input'], [d['position_encoding']], [d['initial_entropy']]])
                inputs.append(input)
                outputs.append(d['output'])
        
        return inputs, outputs

def main():
    args = parse_args()

    DatasetLoader= EntropyDatasetLoader(minio_access_key=args.minio_access_key,
                                minio_secret_key=args.minio_secret_key,
                                minio_ip_addr=args.minio_addr,
                                csv_phrase=args.csv_phrase,
                                operation=args.operation,
                                embedding_type=args.embedding_type,
                                output_type=args.output_type,
                                create_dataset=args.create_dataset)

    if args.create_dataset:
        if args.operation=="substitution":
            DatasetLoader.create_substitution_dataset()
        # other operations will be added here later
        else:
            pass

    inputs, outputs= DatasetLoader.load_substitution_dataset()

    # train model
    model= PromptMutator(minio_client=DatasetLoader.minio_client, output_type=args.output_type, ranking_model="ensemble", operation=args.operation, prompt_type=args.embedding_type)

    model.train(inputs, outputs)
    model.save_model()

if __name__ == "__main__":
    main()