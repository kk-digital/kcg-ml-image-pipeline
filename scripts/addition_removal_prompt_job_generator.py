import sys
sys.path.append('.')

from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from worker.prompt_generation.prompt_generator import load_base_prompts, generate_image_generation_jobs
from utility.minio import cmd
from transformers import CLIPTokenizer, CLIPTextModel

import tqdm
import io
import argparse
import torch
import random
import json
import traceback

import xgboost as xgb
import pandas as pd
import numpy as np

class AdditionRemovalPromptGenerator:
    def __init__(
        self,
        clip_model_path,
        clip_tokenizer_path,
        addition_model_path,
        removal_model_path,
        minio_access_key,
        minio_secret_key,
        minio_addr,
        csv_base_prompts,
        csv_phrase
    ):
        self.csv_base_prompts = csv_base_prompts
        self.df_phrase = pd.read_csv(csv_phrase)
        self.addition_model_path = addition_model_path
        self.removal_model_path = removal_model_path

        self.minio_client = cmd.get_minio_client(
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            minio_ip_addr=minio_addr
        )

        self.scorer = self.load_model(768, device='cuda')
        self.clip_model, self.tokenizer = self.load_clip(clip_model_path, clip_tokenizer_path)
        self.addition_model, self.removal_model = self.load_prompt_addition_removal_models(
            addition_model_path, removal_model_path
        )

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
    
    def sample_prompt_by_score(self, original_prompt, modified_prompt, score):
        if score < 0:
            rand_num = random.random()  # Generates a random float between 0 and 1

            if rand_num < 0.33:
                return modified_prompt
            else:
                return original_prompt
        elif score > 0:
            rand_num = random.random()

            if rand_num < 0.66:
                return modified_prompt
            else:
                return original_prompt

    def load_prompt_addition_removal_models(self):
        removal_model = xgb.Booster()
        removal_model.load_model(self.removal_model_path)

        addition_model = xgb.Booster()
        addition_model.load_model(self.addition_model_path)

        return addition_model, removal_model
    
    def remove_op(self, prompt):
        original_embedding = self.embed(prompt).cpu().numpy()

        # remove random phrase
        # removed_embedding is the embedding of the removed phrase
        prompt_phrase = prompt.split(', ')
        random_index = random.randrange(len(prompt_phrase))
        removed_phrase = prompt_phrase.pop(random_index)
        removed_prompt = ', '.join((prompt_phrase))
        removed_embedding = self.embed(removed_phrase).cpu().numpy()

        original_embedding = np.mean(original_embedding, 0).reshape(-1)
        removed_embedding = np.mean(removed_embedding, 0).reshape(-1)
        model_input = np.concatenate([original_embedding, removed_embedding]).reshape(1, -1)
        score = self.removal_model(model_input)
        chosen_prompt = self.sample_prompt_by_score(prompt, removed_prompt, score)

        return chosen_prompt

    def add_op(self, prompt, df_phrase):
        original_length = self.get_token_length(prompt)

        # truncate prompt by removing last phrase 
        # if prompt length is longer than 60
        if original_length > 60:
            prompt_phrase = prompt.split(', ')
            prompt = ', '.join(prompt_phrase[:-1])
            original_length = self.get_token_length(prompt)
            
        avail_length = 70 - original_length
        original_embedding = self.embed(prompt).cpu().numpy()
        
        # sample a phrase to add
        # add_embedding is the embedding of the phrase to add
        df_sample = df_phrase[df_phrase['token size'] <= avail_length].sample().iloc[0]
        add_phrase = df_sample['phrase str']
        add_prompt = f'{add_phrase}, {prompt}'
        add_embedding = self.embed(add_phrase).cpu().numpy()

        original_embedding = np.mean(original_embedding, 0).reshape(-1)
        add_embedding = np.mean(add_embedding, 0).reshape(-1)
        model_input = np.concatenate([original_embedding, add_embedding]).reshape(1, -1)
        score = self.addition_model(model_input)
        chosen_prompt = self.sample_prompt_by_score(prompt, add_prompt, score)

        return chosen_prompt
    
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
    
    def prompt_mutate_op(self, seed_prompt, iteration=100):
        for i in range(iteration):
            added_prompt = self.add_op(seed_prompt, self.df_phrase)
            removed_prompt = self.remove_op(added_prompt)

            seed_prompt = removed_prompt

        score = self.score_prompt(seed_prompt)

        return seed_prompt, score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default='192.168.3.5:9000')
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--csv_save_path', help='CSV path to save job info', default=None)
    parser.add_argument(
        '--csv_phrase', help='CSV containing phrases, must have "phrase str" column',
        default='input/civitai_phrases_database_v7_no_nsfw.csv'
    )
    parser.add_argument(
        '--csv_base_prompts', help='CSV containing base prompts', 
        default='input/dataset-config/environmental/base-prompts-environmental.csv'
    )
    parser.add_argument('--clip_model_path', help='Path to CLIP text model', default='input/model/clip/txt_emb_model/')
    parser.add_argument('--clip_tokenizer_path', help='Path to CLIP tokenizer', default='input/model/clip/txt_emb_tokenizer/')
    parser.add_argument('--n_prompts', type=int, help='Top number of prompts to send to job queue', default=100)
    parser.add_argument('--n_generations', type=int, help='Number of prompts to generate', default=10000)
    parser.add_argument('--addition_model_path', help='Path to add xgb model', default='input/model/prompt_mutator/xgb_add.json')
    parser.add_argument('--removal_model_path', help='Path to remove xgb model', default='input/model/prompt_mutator/xgb_remove.json')
    parser.add_argument('--n_optimization', help='Number of iterations to apply add/remove', type=int, default=500)
    parser.add_argument('--dataset_name', help='Dataset job name to add to', default='test-generations')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    prompt_generator = AdditionRemovalPromptGenerator(
        minio_addr=args.minio_addr,
        minio_access_key=args.minio_access_key,
        minio_secret_key=args.minio_secret_key,
        csv_base_prompts=args.csv_base_prompts,
        csv_phrase=args.csv_phrase,
        clip_model_path=args.clip_model_path,
        clip_tokenizer_path=args.clip_tokenizer_path,
        addition_model_path=args.addition_model_path,
        removal_model_path=args.removal_model_path,
    )

    print('Generating prompts')
    generated_prompts = []
    for _ in tqdm.tqdm(range(args.n_generations)):
        seed_prompt = prompt_generator.generate_seed_prompt()
        prompt, score = prompt_generator.prompt_mutate_op(seed_prompt, args.n_optimization)
        generated_prompts.append({'prompt': prompt, 'score': score})
    df_generated_prompts = pd.DataFrame(generated_prompts)
    df_generated_prompts = df_generated_prompts.sort_values(by='score', ascending=False)
    df_top = df_generated_prompts.head(args.n_prompts)
    if args.save_path:
        df_top.to_csv(args.save_path, index=False)

    task_uuid = []
    task_time = []
    print('Sending prompts to image generation job')
    pbar = tqdm.tqdm(df_top.iterrows(), total=df_top.shape[0])
    for idx, row in pbar:
        try:
            response = generate_image_generation_jobs(
                positive_prompt=row['prompt'],
                negative_prompt='',
                prompt_scoring_model=prompt_generator.scorer.model_type,
                prompt_score=row['score'],
                prompt_generation_policy='top-k',
                top_k=args.n_prompts,
                dataset_name=args.dataset_name,
            )
            response = json.loads(response)
            task_uuid.append(response['uuid'])
            task_time.append(response['creation_time'])
            pbar.update(1)
        except:
            print('Error occured:')
            print(traceback.format_exc())

    # add null values to task_uuid list if not all jobs were successfully sent
    not_sent = df_top.shape[0] - len(task_uuid)
    if not_sent > 0:
        task_uuid += [None] * not_sent
        task_time += [None] * not_sent
    df_top.loc[:, 'uuid'] = task_uuid
    df_top.loc[:, 'creation_time'] = task_time
    if args.save_path:
        df_top.to_csv(args.save_path, index=False)

    print(f'\n{len(task_uuid)} prompt jobs sent')
    


if __name__ == '__main__':
    main()