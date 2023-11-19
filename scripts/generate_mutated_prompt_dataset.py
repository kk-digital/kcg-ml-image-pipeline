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
import time
import threading
import traceback
import json

import pandas as pd

from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from utility.minio import cmd
from transformers import CLIPTokenizer, CLIPTextModel
from worker.prompt_generation.prompt_generator import load_base_prompts, generate_image_generation_jobs

transformers.logging.set_verbosity_error()

class PromptMutatorDatasetGenerator:
    def __init__(
        self,
        clip_model_path,
        clip_tokenizer_path,
        minio_access_key,
        minio_secret_key,
        minio_ip_addr,
        csv_base_prompts,
        csv_phrase
    ):
        self.minio_client = cmd.get_minio_client(
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            minio_ip_addr=minio_ip_addr
        )
        self.csv_base_prompts = csv_base_prompts
        self.df_phrase = pd.read_csv(csv_phrase)
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
        # removed_embedding is the embedding of the removed phrase
        prompt_phrase = prompt.split(', ')
        random_index = random.randrange(len(prompt_phrase))
        removed_phrase = prompt_phrase.pop(random_index)
        removed_prompt = ', '.join((prompt_phrase))
        removed_length = self.get_token_length(removed_prompt)
        removed_score = self.score_prompt(removed_prompt)
        removed_embedding = self.embed(removed_phrase).cpu().numpy().tolist()

        return {
            'original_prompt': prompt,
            'original_length': original_length,
            'original_score': original_score,
            'original_embedding': original_embedding,
            'removed_prompt': removed_prompt,
            'removed_phrase': removed_phrase,
            'removed_length': removed_length,
            'removed_score': removed_score,
            'removed_embedding': removed_embedding
        }

    def create_add_datapoint(self, prompt, df_phrase):
        original_length = self.get_token_length(prompt)

        # truncate prompt by removing last phrase 
        # if prompt length is longer than 60
        if original_length > 60:
            prompt_phrase = prompt.split(', ')
            prompt = ', '.join(prompt_phrase[:-1])
            original_length = self.get_token_length(prompt)
        
        # use smaller number (68) to get available space
        # the phrase list uses tiktoken and it is not accurate
        # it may exceed length
        avail_length = 68 - original_length
        original_score = self.score_prompt(prompt)
        original_embedding = self.embed(prompt).cpu().numpy().tolist()
        
        # sample a phrase to add
        # add_embedding is the embedding of the phrase to add
        df_sample = df_phrase[df_phrase['token size'] <= avail_length].sample().iloc[0]
        add_phrase = df_sample['phrase str']
        add_prompt = f'{add_phrase}, {prompt}'
        add_length = self.get_token_length(add_prompt)
        add_score = self.score_prompt(add_prompt)
        add_embedding = self.embed(add_phrase).cpu().numpy().tolist()


        return {
            'original_prompt': prompt,
            'original_length': original_length,
            'original_score': original_score,
            'original_embedding': original_embedding,
            'add_prompt': add_prompt,
            'add_phrase': add_phrase,
            'add_length': add_length,
            'add_score': add_score,
            'add_embedding': add_embedding
        }
    
    def generate_seed_prompt(self):
        base_prompt_population = load_base_prompts(self.csv_base_prompts)
        random.shuffle(base_prompt_population)

        seed_prompt = ''
        for phrase in base_prompt_population:
            seed_prompt += f'{phrase},'
            seed_length = self.get_token_length(seed_prompt)
            if seed_length >= 60:
                break

        return seed_prompt
    
    def sample_datapoint(self, seed_prompt=None, n_mutation=1000):
        if seed_prompt is None:
            seed_prompt = self.generate_seed_prompt()

        modified_prompt = seed_prompt
        print(f'Mutating prompt for {n_mutation} iterations')
        for i in tqdm.tqdm(range(n_mutation)):
            add_data = self.create_add_datapoint(modified_prompt, self.df_phrase)
            # keep prompt with higher score
            modified_prompt = add_data['original_prompt'] \
                if add_data['original_score'] > add_data['add_score'] else add_data['add_prompt']
            # print(self.score_prompt(modified_prompt))
            
            remove_data = self.create_remove_datapoint(modified_prompt)
            # keep prompt with higher score
            modified_prompt = remove_data['original_prompt'] \
                if remove_data['original_score'] > remove_data['removed_score'] else remove_data['removed_prompt']
            
        seed_score = self.score_prompt(seed_prompt)
        modified_score = self.score_prompt(modified_prompt)

        print(f'Prompt: {modified_prompt}  Score: {modified_score:.3f}  Base Score: {seed_score:.3f}')
            
        return modified_prompt, modified_score, seed_score
    
    def upload_msgpack_to_minio(self, data, upload_path):
        buffer = io.BytesIO()
        encoder = msgpack.Packer()
        encoded_data = encoder.pack(data)
        buffer.write(encoded_data)
        buffer.seek(0)
        cmd.upload_data(self.minio_client, 'users', upload_path, buffer)

    def upload_csv_to_minio(self, data, upload_path):
        csv_buffer = io.BytesIO()
        data.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        cmd.upload_data(self.minio_client, 'users', upload_path, csv_buffer)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default='192.168.3.5:9000')
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--csv_phrase', help='CSV containing phrases, must have "phrase str" column', default='input/civitai_phrases_database_v7_no_nsfw.csv')
    parser.add_argument('--clip_model_path', help='Path to CLIP text model', default='input/model/clip/txt_emb_model/')
    parser.add_argument('--clip_tokenizer_path', help='Path to CLIP tokenizer', default='input/model/clip/txt_emb_tokenizer/')
    parser.add_argument('--n_data', type=int, help='Number of data samples to generate', default=20)
    parser.add_argument(
        '--csv_base_prompts', help='CSV containing base prompts', 
        default='input/dataset-config/environmental/base-prompts-environmental.csv'
    )
    parser.add_argument('--csv_save_path', help='CSV path to save job info', default='tmp/greedy_output.csv')
    parser.add_argument('--send_job', action='store_true', default=False)
    parser.add_argument('--dataset_name', default='test-generations')
    parser.add_argument('--n_mutation', type=int, default=800)
    args = parser.parse_args()

    return args

def main(
    clip_model_path,
    clip_tokenizer_path,
    minio_access_key,
    minio_secret_key,
    minio_ip_addr,
    csv_phrase,
    n_data,
    csv_save_path,
    csv_base_prompts,
    send_job,
    dataset_name,
    n_mutation
):
    
    dataset_generator = PromptMutatorDatasetGenerator(
        clip_model_path=clip_model_path,
        clip_tokenizer_path=clip_tokenizer_path,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_ip_addr=minio_ip_addr,
        csv_base_prompts=csv_base_prompts,
        csv_phrase=csv_phrase
    )

    df_data = []
    for i in range(n_data):
        print(f'Generation prompt {i+1}')
        prompt, score, seed_score = dataset_generator.sample_datapoint(None, n_mutation)
        if send_job:
            try:
                generate_image_generation_jobs(
                    positive_prompt=prompt,
                    negative_prompt='',
                    prompt_scoring_model=dataset_generator.scorer.model_type,
                    prompt_score=score,
                    prompt_generation_policy='greedy-search',
                    top_k='',
                    dataset_name=dataset_name
                )
                response = json.loads(response)
                task_uuid = response['uuid']
                task_time = response['creation_time']
            except:
                print('Error occured:')
                print(traceback.format_exc())
                task_uuid = -1
                task_time = -1
                continue

        df_data.append({
            'prompt': prompt, 'elm_score': score, 'seed_elm_score': seed_score,
            'task_uuid': task_uuid, 'task_time': task_time
        })

    df_data = pd.DataFrame(df_data)
    df_data.to_csv(csv_save_path, index=False)

if __name__ == '__main__':
    args = parse_args()
    start = time.time()
    main(
        clip_model_path=args.clip_model_path,
        clip_tokenizer_path=args.clip_tokenizer_path,
        minio_access_key=args.minio_access_key,
        minio_secret_key=args.minio_secret_key,
        minio_ip_addr=args.minio_addr,
        csv_phrase=args.csv_phrase,
        n_data=args.n_data,
        csv_save_path=args.csv_save_path,
        csv_base_prompts=args.csv_base_prompts,
        send_job=args.send_job,
        dataset_name=args.dataset_name,
        n_mutation=args.n_mutation
    )
    end = time.time()

    print(f'Time taken: {end - start:.2f} seconds')