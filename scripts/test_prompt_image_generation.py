import json
import sys
sys.path.append('.')

import argparse
import time
import tqdm
import traceback
import torch

import pandas as pd

from transformers import CLIPTokenizer, CLIPTextModel
from worker.prompt_generation.prompt_generator import generate_image_generation_jobs
from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel

# initialize models
def load_models(tokenizer_path, model_path, scorer_path):
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
    model = CLIPTextModel.from_pretrained(model_path)
    scorer = ABRankingELMModel(768)
    scorer.load_pth(scorer_path)

    return tokenizer, model, scorer

def get_embedding(text, tokenizer, model):
    with torch.no_grad():
        token_encoding = tokenizer(text, return_length=True, return_tensors='pt')
        embedding = model(input_ids=token_encoding['input_ids']).last_hidden_state[0]

    embedding = embedding.unsqueeze(0).permute(0, 2, 1).float().to('cuda')

    return embedding

def sample_prompt(df):
    df_shuffle = df.sample(frac=1)
    prompt = []
    token_length = 0
    for idx, row in df_shuffle.iterrows():
        if token_length >= 60:
            break
        prompt.append(row['phrase str'])
        token_length += row['token size']

    prompt = ', '.join(prompt)

    return prompt, token_length


def main(
    data_csv_path,
    n_prompts,
    top_k,
    tokenizer_path,
    model_path,
    scorer_path,
    dataset_name='test-generations',
    save_path=None,
):
    df = pd.read_csv(data_csv_path)

    # load models
    tokenizer, model, scorer = load_models(
        tokenizer_path,
        model_path,
        scorer_path
    )

    # generate prompts
    df_generated_prompts = []
    print('Generating prompts')
    for i in tqdm.tqdm(range(n_prompts)):
        prompt, token_length = sample_prompt(df)
        embedding = get_embedding(prompt, tokenizer, model)
        score = scorer.predict_positive_or_negative_only(embedding).item()
        df_generated_prompts.append({
            'prompt': prompt,
            'score': score,
            'token_length': token_length
        })
    df_generated_prompts = pd.DataFrame(df_generated_prompts)

    # sort prompts
    df_generated_prompts = df_generated_prompts.sort_values(by='score', ascending=False)
    df_top = df_generated_prompts.head(int(top_k * df_generated_prompts.shape[0]))
    if save_path:
        df_top.to_csv(save_path, index=False)

    # send prompts to generation job
    print('Sending prompts to image generation job')
    pbar = tqdm.tqdm(df_top.iterrows(), total=df_top.shape[0])
    task_uuid = []
    task_time = []
    for idx, row in pbar:
        try:
            response = generate_image_generation_jobs(
                positive_prompt=row['prompt'],
                negative_prompt='',
                prompt_scoring_model=scorer.model_type,
                prompt_score=row['score'],
                prompt_generation_policy='top-k',
                top_k=top_k,
                dataset_name=dataset_name,
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
    if save_path:
        df_top.to_csv(save_path, index=False)

    print(f'\n{len(task_uuid)} prompt jobs sent')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_csv_path', type=str)
    ap.add_argument('--n_prompts', type=int)
    ap.add_argument('--tokenizer_path', type=str, default='../kcg-ml-n-grams//weights/txt_emb_tokenizer')
    ap.add_argument('--model_path', type=str, default='../kcg-ml-n-grams/weights/txt_emb_model/')
    ap.add_argument('--scorer_path', type=str, default='../kcg-ml-n-grams/weights/2023-11-03-00-score-elm-v1-embedding-positive.pth')
    ap.add_argument('--top_k', type=float, default=0.05)
    ap.add_argument('--dataset_name', type=str, default='test-generations')
    ap.add_argument('--save_path', type=str, default=None)
    args = ap.parse_args()

    start = time.time()
    main(
        data_csv_path=args.data_csv_path,
        n_prompts=args.n_prompts,
        top_k=args.top_k,
        dataset_name=args.dataset_name,
        save_path=args.save_path,
        tokenizer_path=args.tokenizer_path,
        model_path=args.model_path,
        scorer_path=args.scorer_path
    )
    end = time.time()
    print(f'Time taken: {end - start:.2f} seconds')


