import argparse
import sys
import io
from datetime import datetime

import torch.cuda
from tqdm import tqdm

base_directory = "./"
sys.path.insert(0, base_directory)

from stable_diffusion import CLIPTextEmbedder
from stable_diffusion.model_paths import (SDconfigs, CLIPconfigs)
from configs.model_config import ModelPathConfig
from worker.prompt_generation.prompt_generator import (initialize_prompt_list_from_csv)
from worker.prompt_generation.prompt_generator import generate_prompts_proportional_selection, generate_base_prompts, load_base_prompts
from utility.minio import cmd
from utility.path import separate_bucket_and_file_path
from training_worker.ab_ranking.model.ab_ranking_efficient_net import ABRankingEfficientNetModel
from training_worker.ab_ranking.model.ab_ranking_linear import ABRankingModel
from worker.prompt_generation.prompt_generator import (generate_inpainting_job,
                                                       generate_image_generation_jobs)


class ScoredPrompt:
    def __init__(self, score, positive_prompt, negative_prompt):
        self.score = score
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt


class PromptBatch:
    def __init__(self, positive_prompt_list, negative_prompt_list):
        self.positive_prompt_list = positive_prompt_list
        self.negative_prompt_list = negative_prompt_list



def generate_prompts(clip_text_embedder, scoring_model,
                     base_prompt_population, current_index,
                     prompt_count, prompts, prompt_multiplier, batch_size):

    current_index_in_batch = 0
    positive_prompt_batch = []
    negative_prompt_batch = []
    batch_list = []

    for index in range(0, prompt_multiplier):

        prompt = prompts[current_index + index]
        # N Base Prompt Phrases
        # Hard coded probability of choose 0,1,2,3,4,5, etc base prompt phrases
        # Chance for 0 base prompt phrases should be 30%
        # choose_probability = [0.3, 0.3, 0.2, 0.2, 0.2]
        choose_probability = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        if base_prompt_population is not None:
            base_prompt_list = generate_base_prompts(base_prompt_population, choose_probability)
        else:
            base_prompt_list = []

        base_prompts = ''

        for base_prompt in base_prompt_list:
            base_prompts = base_prompts + base_prompt + ', '

        positive_text_prompt = base_prompts + prompt.positive_prompt_str
        negative_text_prompt = prompt.negative_prompt_str

        positive_prompt_batch.append(positive_text_prompt)
        negative_prompt_batch.append(negative_text_prompt)

        current_index_in_batch = current_index_in_batch + 1

        if (current_index_in_batch % batch_size == 0 or
                current_index_in_batch == prompt_multiplier):

            this_batch = PromptBatch(positive_prompt_batch, negative_prompt_batch)
            batch_list.append(this_batch)
            positive_prompt_batch = []
            negative_prompt_batch = []


    print('Scoring Generated Prompts ')
    scored_prompts = []
    batch_index = 0
    for batch in batch_list:
        batch_index = batch_index + 1
        print(f'batch {batch_index} out of {len(batch_list)}')
        positive_prompt_embeddings_list = clip_text_embedder(batch.positive_prompt_list)
        negative_prompt_embeddings_list = clip_text_embedder(batch.negative_prompt_list)

        for index in range(len(batch.positive_prompt_list)):
            positive_prompt = batch.positive_prompt_list[index]
            negative_prompt = batch.negative_prompt_list[index]
            positive_prompt_embeddings = positive_prompt_embeddings_list[index]
            negative_prompt_embeddings = negative_prompt_embeddings_list[index]

            prompt_score = 0
            if scoring_model is not None and clip_text_embedder is not None:
                prompt_score = scoring_model.predict(positive_prompt_embeddings,
                                                     negative_prompt_embeddings).item()

            scored_prompt = ScoredPrompt(prompt_score, positive_prompt,
                                         negative_prompt)
            scored_prompts.append(scored_prompt)

        del positive_prompt_embeddings_list
        del negative_prompt_embeddings_list
        torch.cuda.empty_cache()

    # Sort the list based on the maximize_int1 function
    sorted_scored_prompts = sorted(scored_prompts, key=maximize_score)

    chosen_scored_prompts = sorted_scored_prompts[:prompt_count]

    return chosen_scored_prompts


def maximize_score(scored_prompt):
    return -scored_prompt.score


def load_linear_model(minio_client, dataset_bucket, model_path):

    linear_model = ABRankingModel(768*2)

    model_file_data = cmd.get_file_from_minio(minio_client, dataset_bucket, model_path)

    if model_file_data is None:
        return

    # Create a BytesIO object and write the downloaded content into it
    byte_buffer = io.BytesIO()
    for data in model_file_data.stream(amt=8192):
        byte_buffer.write(data)
    # Reset the buffer's position to the beginning
    byte_buffer.seek(0)

    linear_model.load(byte_buffer)

    return linear_model

def generate_character_generation_jobs(scored_prompt):

    dataset_name = "character"
    init_img_path = "./test/test_inpainting/white_512x512.jpg"
    mask_path = "./test/test_inpainting/character_mask.png"

    if scored_prompt is None:
        return

    positive_prompt = scored_prompt.positive_prompt
    negative_prompt = scored_prompt.negative_prompt

    generate_inpainting_job(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        dataset_name=dataset_name,
        init_img_path=init_img_path,
        mask_path=mask_path,

    )


def generate_waifu_generation_jobs(scored_prompt):

    dataset_name = "waifu"

    if scored_prompt is None:
        return

    positive_prompt = scored_prompt.positive_prompt
    negative_prompt = scored_prompt.negative_prompt

    generate_image_generation_jobs(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        dataset_name=dataset_name,
    )


def generate_propaganda_poster_generation_jobs(scored_prompt):

    dataset_name = "propaganda-poster"

    if scored_prompt is None:
        return

    positive_prompt = scored_prompt.positive_prompt
    negative_prompt = scored_prompt.negative_prompt

    print(f'spawning {dataset_name} job')

    generate_image_generation_jobs(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        dataset_name=dataset_name,
    )


def generate_environmental_image_generation_jobs(scored_prompt):

    dataset_name = 'environmental'

    if scored_prompt is None:
        return

    positive_prompt = scored_prompt.positive_prompt
    negative_prompt = scored_prompt.negative_prompt

    generate_image_generation_jobs(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        dataset_name=dataset_name,
    )

def parse_args():
    parser = argparse.ArgumentParser(description="generate prompts")

    # Required parameters
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--dataset", type=str, default='environmental')
    parser.add_argument("--top_k", type=float, default=0.1)
    parser.add_argument("--prompt_count", type=int, default=1)
    parser.add_argument("--csv_dataset_path", type=str, default='input/civitai_phrases_database_v6.csv')
    parser.add_argument("--csv_base_prompts", type=str,
                        default='')

    parser.add_argument("--model_path", type=str,
                        default='')

    parser.add_argument("--minio_access_key", type=str, default='v048BpXpWrsVIHUfdAix')
    parser.add_argument("--minio_secret_key", type=str, default='4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu')


    parser.add_argument("--batch_size", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device
    dataset = args.dataset
    csv_base_prompts = args.csv_base_prompts
    top_k = args.top_k
    csv_dataset_path = args.csv_dataset_path
    prompt_count = args.prompt_count
    minio_secret_key = args.minio_secret_key
    minio_access_key = args.minio_access_key
    model_path = args.model_path
    batch_size = args.batch_size

    clip_text_embedder = CLIPTextEmbedder(device=device)
    config = ModelPathConfig()
    clip_text_embedder.load_submodels(
        tokenizer_path=config.get_model_folder_path(CLIPconfigs.TXT_EMB_TOKENIZER),
        transformer_path=config.get_model_folder_path(CLIPconfigs.TXT_EMB_TEXT_MODEL)
    )

    minio_client = cmd.get_minio_client(minio_access_key, minio_secret_key)

    if minio_client is not None:
        scoring_model = load_linear_model(minio_client, 'datasets', model_path)
    else:
        scoring_model = None

    print(f'generating {prompt_count} prompts for dataset {dataset}')

    begin_time = datetime.now()

    prompt_multiplier = int(1.0 / top_k)

    total_prompt_count = prompt_count * prompt_multiplier

    total_prompt_count = int(total_prompt_count)

    phrases, phrases_token_size, positive_count_list, negative_count_list = initialize_prompt_list_from_csv(
        csv_dataset_path, 0)

    prompts = generate_prompts_proportional_selection(phrases,
                                                      phrases_token_size,
                                                      positive_count_list,
                                                      negative_count_list,
                                                      total_prompt_count,
                                                      '')

    if csv_base_prompts != '' and csv_base_prompts is not None:
        base_prompt_population = load_base_prompts(csv_base_prompts)
    else:
        base_prompt_population = None

    current_index_in_batch = 0
    positive_prompt_batch = []
    negative_prompt_batch = []
    batch_list = []

    for index in range(0, total_prompt_count):

        prompt = prompts[index]
        # N Base Prompt Phrases
        # Hard coded probability of choose 0,1,2,3,4,5, etc base prompt phrases
        # Chance for 0 base prompt phrases should be 30%
        # choose_probability = [0.3, 0.3, 0.2, 0.2, 0.2]
        choose_probability = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        if base_prompt_population is not None:
            base_prompt_list = generate_base_prompts(base_prompt_population, choose_probability)
        else:
            base_prompt_list = []

        base_prompts = ''

        for base_prompt in base_prompt_list:
            base_prompts = base_prompts + base_prompt + ', '

        positive_text_prompt = base_prompts + prompt.positive_prompt_str
        negative_text_prompt = prompt.negative_prompt_str

        positive_prompt_batch.append(positive_text_prompt)
        negative_prompt_batch.append(negative_text_prompt)

        current_index_in_batch = current_index_in_batch + 1

        if (current_index_in_batch % batch_size == 0 or
                current_index_in_batch == prompt_multiplier):
            this_batch = PromptBatch(positive_prompt_batch, negative_prompt_batch)
            batch_list.append(this_batch)
            positive_prompt_batch = []
            negative_prompt_batch = []

    print('Scoring Generated Prompts ')
    scored_prompts = []
    batch_index = 0
    for batch in batch_list:
        batch_index = batch_index + 1
        print(f'batch {batch_index} out of {len(batch_list)}')
        positive_prompt_embeddings_list = clip_text_embedder(batch.positive_prompt_list)
        negative_prompt_embeddings_list = clip_text_embedder(batch.negative_prompt_list)

        for index in range(len(batch.positive_prompt_list)):
            positive_prompt = batch.positive_prompt_list[index]
            negative_prompt = batch.negative_prompt_list[index]
            positive_prompt_embeddings = positive_prompt_embeddings_list[index]
            negative_prompt_embeddings = negative_prompt_embeddings_list[index]

            prompt_score = 0
            if scoring_model is not None and clip_text_embedder is not None:
                prompt_score = scoring_model.predict(positive_prompt_embeddings,
                                                     negative_prompt_embeddings).item()

            scored_prompt = ScoredPrompt(prompt_score, positive_prompt,
                                         negative_prompt)
            scored_prompts.append(scored_prompt)

        del positive_prompt_embeddings_list
        del negative_prompt_embeddings_list
        torch.cuda.empty_cache()

    # Sort the list based on the maximize_int1 function
    sorted_scored_prompts = sorted(scored_prompts, key=maximize_score)
    chosen_scored_prompts = sorted_scored_prompts[:prompt_count]


    for index in range(0, prompt_count):
        print('generating ', index ,' out of ', prompt_count)

        prompt = chosen_scored_prompts[index]

        if dataset == 'environmental':
            generate_environmental_image_generation_jobs(prompt)
        elif dataset == 'character':
            generate_character_generation_jobs(prompt)
        elif dataset == 'waifu':
            generate_waifu_generation_jobs(prompt)
        elif dataset == 'propaganda-poster':
            generate_propaganda_poster_generation_jobs(prompt)

    end_time = datetime.now()
    elapsed_time = end_time - begin_time

    print(f"Execution time: {elapsed_time}")

if __name__ == '__main__':
    main()



