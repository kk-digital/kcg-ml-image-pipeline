import argparse
import io
import os
import sys
import numpy as np
import pandas as pd
import torch

base_directory = "./"
sys.path.insert(0, base_directory)

from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from utility.ensemble.ensemble_helpers import Binning, SigmaScoresWithEntropy
from stable_diffusion.model.clip_text_embedder.clip_text_embedder import CLIPTextEmbedder
from utility.minio import cmd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--dataset', default="environmental")
    parser.add_argument('--policy', default="greedy-substitution-search-v1")
    parser.add_argument('--bins', help="number of bins", default=8)
    parser.add_argument('--csv-path', help="directory where csv files are stored", default="environmental/output/generated-prompts-csv/")
    parser.add_argument('--output-path', help="directory where csv files are stored", default="environmental/data/prompt-generator/substitution/")

    return parser.parse_args()

class PromptEntropy:
    def __init__(
        self,
        minio_access_key,
        minio_secret_key,
        minio_ip_addr,
        dataset,
        policy,
        bins,
        csv_path,
        output_path
    ):
        # get minio client
        self.minio_client = cmd.get_minio_client(minio_access_key,
                                            minio_secret_key,
                                            minio_ip_addr)
        
        self.dataset=dataset
        self.policy=policy
        self.bins=bins
        self.step=1

        self.csv_path=csv_path
        self.output_path=output_path

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
            if model_file.endswith(f"score-elm-v1-embedding.pth")
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
            embedding_model = model_class(768*2)
            embedding_model.load_pth(byte_buffer)
            embedding_model.model=embedding_model.model.to(self.device)

            loaded_models.append(embedding_model)

        return loaded_models
    
    # get the clip text embedding of a prompt or a phrase
    def get_prompt_embedding(self, prompt):
        with torch.no_grad():
            embedding= self.embedder(prompt)

        embedding= embedding.unsqueeze(0)
        embedding=embedding.to(self.device)

        return embedding

    def get_ensemble_sigma_scores(self, positive_embedding, negative_embedding):
        sigma_scores=[]
        for model in self.ensemble_models:
            mean=model.mean
            std=model.standard_deviation
            with torch.no_grad():
                score=model.predict(positive_embedding,negative_embedding).item()
                score=(score - mean)/std
            
            sigma_scores.append(score)
        
        return np.array(sigma_scores)
        

    def get_prompt_entropy(self, positive_prompt, negative_prompt):
        positive_embedding= self.get_prompt_embedding(positive_prompt)
        negative_embedding= self.get_prompt_embedding(negative_prompt)
        sigma_scores=self.get_ensemble_sigma_scores(positive_embedding, negative_embedding)

        # get entropy classes
        binning= Binning(start=-2,count=self.bins,step=self.step)
        entropy_data=SigmaScoresWithEntropy(sigma_scores, binning)

        # get entropy, variance and average
        entropy= entropy_data.entropy
        variance= entropy_data.variance
        mean= entropy_data.mean

        return entropy, variance, mean


    def get_generated_prompts_csv_data(self):
        prompt_data=[]
        # get minio paths for csv files
        csv_paths = self.get_csv_paths()

        for path in csv_paths:
            print(path)
            data = self.minio_client.get_object('datasets', path)
            csv_data = io.BytesIO(data.read())

            # Read the CSV into a DataFrame
            df = pd.read_csv(csv_data)
            # Filter out rows where 'negative_prompt' is missing or empty
            df = df[df['negative_prompt'].notna() & (df['negative_prompt'] != '')]
            # get entropy fields
            df['entropy'], df['variance'], df['mean'] = zip(*df.apply(lambda row: self.get_prompt_entropy(row['positive_prompt'], row['negative_prompt']), axis=1))
            # Append the DataFrame to the list
            prompt_data.append(df)

        # Concatenate all DataFrames into a single DataFrame
        combined_df = pd.concat(prompt_data, ignore_index=True)

        return combined_df

    def get_csv_paths(self):
        objects=self.minio_client.list_objects('datasets', self.csv_path, recursive=True)
        csv_files = []
        for obj in objects: 
            if obj.object_name.endswith(f"-{self.policy}-{self.dataset}.csv"):
                csv_files.append(obj.object_name)
                
        return csv_files
    
    def save_csv_file(self):
        csv_data=self.get_generated_prompts_csv_data()

        local_path="output/generated_prompts.csv"
        pd.DataFrame(csv_data).to_csv(local_path, index=False)
        # Read the contents of the CSV file
        with open(local_path, 'rb') as file:
            csv_content = file.read()

        #Upload the CSV file to Minio
        buffer = io.BytesIO(csv_content)
        buffer.seek(0)

        minio_path=self.output_path + f"{self.policy}-environmental-entropy.csv"
        cmd.upload_data(self.minio_client, 'datasets', minio_path, buffer)
        # Remove the temporary file
        os.remove(local_path)

def main():
    args = parse_args()
    prompts_entropy= PromptEntropy(minio_access_key=args.minio_access_key,
                                  minio_secret_key=args.minio_secret_key,
                                  minio_ip_addr=args.minio_addr,
                                  dataset=args.dataset,
                                  policy=args.policy,
                                  bins=args.bins,
                                  csv_path=args.csv_path,
                                  output_path=args.output_path)
    
    prompts_entropy.save_csv_file()
    
if __name__ == "__main__":
    main()
        