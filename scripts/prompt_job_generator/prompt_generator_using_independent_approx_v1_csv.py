import os
import sys
import argparse
import io
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
import random
import math

base_directory = "./"
sys.path.insert(0, base_directory)

from scripts.image_scorer import ImageScorer
from training_worker.http import request
from utility.minio import cmd
from data_loader.phrase_scores_loader import PhraseScoresLoader

# find the first element, whose cumulative prob is more than the random float
def find_first_element_binary_search(cumulative_prob_arr, random_float):
    low = 0
    high = len(cumulative_prob_arr) - 1
    mid = 0

    while low <= high:
        mid = (high + low) / 2
        mid = math.floor(mid)

        # If random_float is greater, ignore left half
        if cumulative_prob_arr[mid] < random_float:
            low = mid + 1
        # If random_float is smaller, ignore right half
        elif cumulative_prob_arr[mid] > random_float:
            high = mid - 1
        # else:
        #     return mid

        # use this index since sometimes the exact
        # random num is not in the list
        if low == high:
            # assert cumulative_prob_arr[low-1] < random_float
            # assert cumulative_prob_arr[low] >= random_float, "{} >= {}, next index val={}".format(cumulative_prob_arr[low], random_float, cumulative_prob_arr[low+1])
            assert round(cumulative_prob_arr[low], 4) >= 0.0, "val={}".format(cumulative_prob_arr[low])
            assert round(cumulative_prob_arr[low], 4) <= 1.0, "val={}".format(cumulative_prob_arr[low])

            return low

    # If we reach here, then the element was not present
    return -1

def generate_prompt(positive_phrase_scores_loader,
                     positive_phrase_origin_indexes,
                     positive_cumulative_probability_arr,
                     negative_phrase_scores_loader,
                     negative_phrase_origin_indexes,
                     negative_cumulative_probability_arr,
                    ):
    max_token_size = 75
    comma_token_size = 1
    positive_total_cumulative = int(positive_cumulative_probability_arr[-1])
    negative_total_cumulative = int(negative_cumulative_probability_arr[-1])

    positive_prompt_total_token_size = 0
    negative_prompt_total_token_size = 0
    positive_prompt = []
    negative_prompt = []
    positive_used_phrase_dict = {}
    negative_used_phrase_dict = {}

    # positive prompt
    while positive_prompt_total_token_size < max_token_size:
        random_float = random.uniform(0, positive_total_cumulative)
        random_index = find_first_element_binary_search(positive_cumulative_probability_arr, random_float)
        if random_index in positive_used_phrase_dict:
            continue

        prompt_index = positive_phrase_origin_indexes[random_index]
        random_phrase = positive_phrase_scores_loader.get_phrase(prompt_index)

        chosen_phrase_size = positive_phrase_scores_loader.get_token_size(prompt_index)
        sum_token_size = positive_prompt_total_token_size + chosen_phrase_size + comma_token_size
        if sum_token_size < max_token_size:
            # update used array
            positive_used_phrase_dict[random_index] = 1
            positive_prompt.append(random_phrase)
            positive_prompt_total_token_size = sum_token_size
        else:
            break

    # negative prompt
    while negative_prompt_total_token_size < max_token_size:
        random_float = random.uniform(0, negative_total_cumulative)
        random_index = find_first_element_binary_search(negative_cumulative_probability_arr, random_float)
        if random_index in negative_used_phrase_dict:
            continue

        prompt_index = negative_phrase_origin_indexes[random_index]
        random_phrase = negative_phrase_scores_loader.get_phrase(prompt_index)

        chosen_phrase_size = negative_phrase_scores_loader.get_token_size(prompt_index)
        sum_token_size = negative_prompt_total_token_size + chosen_phrase_size + comma_token_size
        if sum_token_size < max_token_size:
            # update used array
            negative_used_phrase_dict[random_index] = 1
            negative_prompt.append(random_phrase)
            negative_prompt_total_token_size = sum_token_size
        else:
            break

    positive_prompt_str = ', '.join([prompt for prompt in positive_prompt])
    negative_prompt_str = ', '.join([prompt for prompt in negative_prompt])

    prompt = (positive_prompt_str, negative_prompt_str)

    return prompt

def generate_prompts(positive_phrase_scores_loader,
                     positive_phrase_origin_indexes,
                     positive_cumulative_probability_arr,
                     negative_phrase_scores_loader,
                     negative_phrase_origin_indexes,
                     negative_cumulative_probability_arr,
                     prompt_count):
    generated_prompts = []

    print("Generating {} prompts...".format(prompt_count))
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(prompt_count):
            futures.append(executor.submit(generate_prompt,
                                           positive_phrase_scores_loader=positive_phrase_scores_loader,
                                           positive_phrase_origin_indexes=positive_phrase_origin_indexes,
                                           positive_cumulative_probability_arr=positive_cumulative_probability_arr,
                                           negative_phrase_scores_loader=negative_phrase_scores_loader,
                                           negative_phrase_origin_indexes=negative_phrase_origin_indexes,
                                           negative_cumulative_probability_arr=negative_cumulative_probability_arr))

        for future in tqdm(as_completed(futures), total=len(futures)):
            prompt = future.result()
            positive_prompt = prompt[0]
            negative_prompt = prompt[1]
            print("positive prompt=", positive_prompt)
            print("negative prompt=", negative_prompt)
            print("---------------------------------------------------------------")

    return generated_prompts

def get_cumulative_probability_arr(index_phrase_score_data):
    scores_arr = []
    for index, data in index_phrase_score_data.items():
        score = data.score
        scores_arr.append(score)

    scores_np_arr = np.array(scores_arr)

    # get boltzman prob
    k = 1.0
    temperature = 0.8
    probability_arr = np.exp(-(scores_np_arr/(k*temperature)))

    # normalize
    normalized_probability_arr = probability_arr/np.sum(probability_arr)
    assert round(np.sum(normalized_probability_arr), 4) == 1.0, "sum={}".format(np.sum(normalized_probability_arr))

    # sort
    sorted_probability_arr = []
    sorted_indexes = sorted(range(len(normalized_probability_arr)), key=lambda x: normalized_probability_arr[x], reverse=True)
    for i in sorted_indexes:
        sorted_probability_arr.append(normalized_probability_arr[i])
    sorted_probability_arr = np.array(sorted_probability_arr)

    # get cumulative
    cumulative_probability_arr = sorted_probability_arr.cumsum()

    print("scores=", scores_np_arr)
    print("prob=", probability_arr)
    print("normalized=", normalized_probability_arr)
    print("sorted prob=", sorted_probability_arr)
    print("cumulative=", cumulative_probability_arr)

    return sorted_indexes, cumulative_probability_arr


def run_prompt_generator(minio_client,
                         dataset_name,
                         positive_phrase_scores_csv,
                         negative_phrase_scores_csv,
                         prompt_count):
    positive_phrase_scores_loader = PhraseScoresLoader(dataset_name=dataset_name,
                                                       phrase_scores_csv=positive_phrase_scores_csv,
                                                       minio_client=minio_client,
                                                       )
    positive_phrase_scores_loader.load_dataset()
    positive_phrase_origin_indexes, positive_cumulative_probability_arr = get_cumulative_probability_arr(positive_phrase_scores_loader.index_phrase_score_data)

    negative_phrase_scores_loader = PhraseScoresLoader(dataset_name=dataset_name,
                                                       phrase_scores_csv=negative_phrase_scores_csv,
                                                       minio_client=minio_client,
                                                       )

    negative_phrase_scores_loader.load_dataset()
    negative_phrase_origin_indexes, negative_cumulative_probability_arr = get_cumulative_probability_arr(negative_phrase_scores_loader.index_phrase_score_data)

    generate_prompts(positive_phrase_scores_loader,
                     positive_phrase_origin_indexes,
                     positive_cumulative_probability_arr,
                     negative_phrase_scores_loader,
                     negative_phrase_origin_indexes,
                     negative_cumulative_probability_arr,
                     prompt_count)


def parse_args():
    parser = argparse.ArgumentParser(description="Prompt Job Generator using Independent Approx V1 csv results")
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--dataset-name', required=True, help='Name of the dataset to generate prompt jobs')
    parser.add_argument('--positive-phrase-scores-csv', required=True, help='Filename of the positive phrase scores csv')
    parser.add_argument('--negative-phrase-scores-csv', required=True, help='Filename of the negative phrase scores csv')
    parser.add_argument('--prompt-count', required=True, type=int, help='Number of prompt jobs to generate')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    dataset_name = args.dataset_name
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)
    if dataset_name != "all":
        run_prompt_generator(minio_client,
                             args.dataset_name,
                             args.positive_phrase_scores_csv,
                             args.negative_phrase_scores_csv,
                             args.prompt_count)
    else:
        # if all, do for all existing datasets
        # get dataset name list
        dataset_names = request.http_get_dataset_names()
        print("dataset names=", dataset_names)
        for dataset in dataset_names:
            try:
                run_prompt_generator(minio_client,
                                     args.dataset_name,
                                     args.positive_phrase_scores_csv,
                                     args.negative_phrase_scores_csv,
                                     args.prompt_count)
            except Exception as e:
                print("Error running prompt generator for {}: {}".format(dataset, e))


if __name__ == "__main__":
    main()
