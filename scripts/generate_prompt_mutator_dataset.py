import sys
sys.path.append('.')

import argparse
import torch
import random
import transformers
import io
import os
import msgpack
import tqdm

import pandas as pd

from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from utility.minio import cmd
from transformers import CLIPTokenizer, CLIPTextModel

transformers.logging.set_verbosity_error()

class PromptMutatorDatasetGenerator:
    def __init__(
        self,
        clip_model_path,
        clip_tokenizer_path,
        minio_access_key,
        minio_secret_key,
        minio_ip_addr
    ):
        
        self.minio_client = cmd.get_minio_client(
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            minio_ip_addr=minio_ip_addr
        )

        self.scorer = self.load_model(768, device='cuda')
        self.clip_model, self.tokenizer = self.load_clip(clip_model_path, clip_tokenizer_path)


    def load_clip(self, model_path, tokenizer_path):
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        model = CLIPTextModel.from_pretrained(model_path).eval().to('cuda')

        return model, tokenizer

    def load_model(self, input_size, device='cuda'):
        input_path = "environmental/models/ranking/"

        embedding_model = ABRankingELMModel(input_size)

        model_files = cmd.get_list_of_objects_with_prefix(self.minio_client, 'datasets', input_path)
        most_recent_model = None

        for model_file in model_files:
            if model_file.endswith("score-elm-v1-embedding-positive.pth"):
                most_recent_model = model_file

        if most_recent_model:
            model_file_data =cmd.get_file_from_minio(self.minio_client, 'datasets', most_recent_model)
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

    def embed(self, prompt,):
        with torch.no_grad():
            token_encoding = self.tokenizer(prompt, return_length=True, return_tensors='pt')
            embedding = self.clip_model(input_ids=token_encoding['input_ids'].to('cuda')).last_hidden_state[0]

        return embedding

    def score_prompt(self, prompt):
        embedding = self.embed(prompt)
        embedding = embedding.unsqueeze(0).permute(0, 2, 1)
        score = self.scorer.predict_positive_or_negative_only(embedding).item()

        return score

    def get_token_length(self, prompt):
        token_encoding = self.tokenizer(prompt, return_length=True, return_tensors='pt')

        return token_encoding['length'].item()
    
    def get_tokens(self, prompt):
        token_encoding = self.tokenizer(prompt, return_length=True, return_tensors='pt')

        return token_encoding['input_ids'].cpu().numpy()[0].tolist()

    def create_remove_datapoint(self, prompt):
        original_score = self.score_prompt(prompt)
        original_length = self.get_token_length(prompt)
        original_embedding = self.embed(prompt).cpu().numpy().tolist()

        # remove random phrase
        prompt_phrase = prompt.split(', ')
        random_index = random.randrange(len(prompt_phrase))
        removed_phrase = prompt_phrase.pop(random_index)
        removed_prompt = ', '.join((prompt_phrase))
        removed_length = self.get_token_length(removed_prompt)
        removed_score = self.score_prompt(removed_prompt)
        removed_embedding = self.embed(removed_prompt).cpu().numpy().tolist()

        # compute positional encoding
        original_token = self.get_tokens(prompt)
        removed_token = self.get_tokens(removed_prompt)
        positional_encoding = [1 if token in original_token and token not in removed_token else 0 for token in original_token]
        positional_encoding = positional_encoding + [0] * (77 - len(positional_encoding))

        return {
            'original_prompt': prompt,
            'original_length': original_length,
            'original_score': original_score,
            'original_embedding': original_embedding,
            'removed_prompt': removed_prompt,
            'removed_phrase': removed_phrase,
            'removed_length': removed_length,
            'removed_score': removed_score,
            'removed_embedding': removed_embedding,
            'positional_encoding': positional_encoding
        }

    def create_add_datapoint(self, prompt, df_phrase):
        original_length = self.get_token_length(prompt)

        # truncate prompt by removing last phrase 
        # if prompt length is longer than 60
        if original_length > 60:
            prompt_phrase = prompt.split(', ')
            prompt = ', '.join(prompt_phrase[:-1])
            original_length = self.get_token_length(prompt)
            
        avail_length = 75 - original_length
        original_score = self.score_prompt(prompt)
        original_embedding = self.embed(prompt).cpu().numpy().tolist()
        
        df_sample = df_phrase[df_phrase['token size'] <= avail_length].sample().iloc[0]
        add_phrase = df_sample['phrase str']
        add_prompt = f'{add_phrase}, {prompt}'
        add_length = self.get_token_length(add_prompt)
        add_score = self.score_prompt(add_prompt)
        add_embedding = self.embed(add_prompt).cpu().numpy().tolist()

        # compute positional encoding
        original_token = self.get_tokens(prompt)
        add_token = self.get_tokens(add_prompt)
        positional_encoding = [1 if token in add_token and token not in original_token else 0 for token in add_token]
        positional_encoding = positional_encoding + [0] * (77 - len(positional_encoding))

        return {
            'original_prompt': prompt,
            'original_length': original_length,
            'original_score': original_score,
            'original_embedding': original_embedding,
            'add_prompt': add_prompt,
            'add_phrase': add_phrase,
            'add_length': add_length,
            'add_score': add_score,
            'add_embedding': add_embedding,
            'positional_encoding': positional_encoding
        }
    
    def upload_msgpack_to_minio(self, data, upload_path):
        buffer = io.BytesIO()
        encoder = msgpack.Packer()
        encoded_data = encoder.pack(data)
        buffer.write(encoded_data)
        buffer.seek(0)
        cmd.upload_data(self.minio_client, 'users', upload_path, buffer)

    def upload_csv_to_minio(self, data, upload_path):
        csv_buffer = io.StringIO()
        data.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        cmd.upload_data(self.minio_client, 'users', upload_path, csv_buffer)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default='192.168.3.5:9000')
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--df_phrase_path', required=True, help='CSV containing phrases, must have "phrase str" column')
    parser.add_argument('--df_seed_path', required=True, help='CSV containing prompts, must have "positive_prompt" column')
    parser.add_argument('--clip_model_path', help='Path to CLIP text model', default='input/model/clip/txt_emb_model/')
    parser.add_argument('--clip_tokenizer_path', help='Path to CLIP tokenizer', default='input/model/clip/txt_emb_tokenizer/')
    parser.add_argument('--n_data', type=int, help='Number of data samples to generate')
    parser.add_argument('--minio_upload_path', help='Minio upload folder path')
    args = parser.parse_args()

    return args

def main(
    clip_model_path,
    clip_tokenizer_path,
    minio_access_key,
    minio_secret_key,
    minio_ip_addr,
    df_phrase_path,
    df_seed_path,
    n_data,
    minio_upload_path
):
    df_phrase = pd.read_csv(df_phrase_path)
    df_seed = pd.read_csv(df_seed_path)

    seed_prompt = df_seed['positive_prompt'].sample().iloc[0]

    dataset_generator = PromptMutatorDatasetGenerator(
        clip_model_path=clip_model_path,
        clip_tokenizer_path=clip_tokenizer_path,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_ip_addr=minio_ip_addr
    )

    # folder number to save data
    folder = 0
    df_data_removed = []
    df_data_add = []
    for i in tqdm.tqdm(range(n_data)):
        removed_data = dataset_generator.create_remove_datapoint(seed_prompt)
        add_data = dataset_generator.create_add_datapoint(removed_data['removed_prompt'], df_phrase)

        if ((i + 1) % 10000) == 0:
            df_data_removed = pd.DataFrame(df_data_removed)
            df_data_add = pd.DataFrame(df_data_add)

            df_data_removed = df_data_removed.drop(
                ['original_embedding', 'removed_embedding', 'positional_encoding'], axis=1
            )
            df_data_add = df_data_add.drop(
                ['original_embedding', 'add_embedding', 'positional_encoding'], axis=1
            )
            removed_path = os.path.join(
                minio_upload_path,
                str(folder).zfill(6),
                'prompt_removal',
                f'data_{str(i).zfill(5)}.csv'
            )
            add_path = os.path.join(
                minio_upload_path,
                str(folder).zfill(6),
                'prompt_addition',
                f'data_{str(i).zfill(5)}.csv'
            )
            dataset_generator.upload_csv_to_minio(df_data_removed, removed_path)
            dataset_generator.upload_csv_to_minio(df_data_add, add_path)

            folder += 1
            df_data_removed = []
            df_data_add = []

        removed_path = os.path.join(
            minio_upload_path,
            str(folder).zfill(6),
            'prompt_removal',
            f'{str(i).zfill(5)}.msgpack'
        )
        add_path = os.path.join(
            minio_upload_path,
            str(folder).zfill(6),
            'prompt_addition',
            f'{str(i).zfill(5)}.msgpack'
        )
        df_data_removed.append(removed_data)
        df_data_add.append(add_data)
        dataset_generator.upload_to_minio(removed_data, removed_path)
        dataset_generator.upload_to_minio(add_data, add_path)

if __name__ == '__main__':
    args = parse_args()
    main(
        clip_model_path=args.clip_model_path,
        clip_tokenizer_path=args.clip_tokenizer_path,
        minio_access_key=args.minio_access_key,
        minio_secret_key=args.minio_secret_key,
        minio_ip_addr=args.minio_addr,
        df_phrase_path=args.df_phrase_path,
        df_seed_path=args.df_seed_path,
        n_data=args.n_data,
        minio_upload_path=args.minio_upload_path
    )