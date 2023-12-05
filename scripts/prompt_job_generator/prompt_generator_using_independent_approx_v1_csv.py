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
from datetime import datetime
from pytz import timezone
base_directory = "./"
sys.path.insert(0, base_directory)

from scripts.image_scorer import ImageScorer
from training_worker.http import request
from utility.minio import cmd
from utility.boltzman.boltzman_phrase_scores_loader import BoltzmanPhraseScoresLoader
from worker.prompt_generation.prompt_generator import generate_image_generation_jobs_with_temperature, generate_inpainting_job_with_temperature

def all_same(items):
    return np.all(x == items[0] for x in items)

# find the first element, whose cumulative prob is more than the random float
def find_first_element_binary_search(cumulative_prob_arr, random_float):
    low = 0
    high = len(cumulative_prob_arr) - 1
    mid = 0

    loop_count = 0
    while low < high:
        loop_count += 1
        assert loop_count < 32, "Error: binary search loop count is more than 32"

        mid = (high + low) / 2
        mid = math.floor(mid)

        # If random_float is greater, ignore left half
        if cumulative_prob_arr[mid] < random_float:
            low = mid + 1
        # If random_float is smaller, ignore right half
        elif cumulative_prob_arr[mid] >= random_float:
            high = mid - 1

        # use this index since sometimes the exact
        # random num is not in the list
        if low == high:
            # assert cumulative_prob_arr[low-1] < random_float
            # assert cumulative_prob_arr[low] >= random_float, "{} >= {}, next index val={}".format(cumulative_prob_arr[low], random_float, cumulative_prob_arr[low+1])
            # assert round(cumulative_prob_arr[low], 4) >= 0.0, "val={}".format(cumulative_prob_arr[low])
            # assert round(cumulative_prob_arr[low], 4) <= 1.0, "val={}".format(cumulative_prob_arr[low])

            return low


    # If we reach here, then the element was not present
    return -1


def upload_prompt_generation_data_to_csv(minio_client,
                                         dataset_name,
                                         prompt_generation_data,
                                         boltzman_temperature,
                                         boltzman_k):
    print("Saving prompt generation data to csv...")
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow((["job_uuid", "positive_prompt", "negative_prompt", "boltzman temperature", "boltzman k"]))

    for data in prompt_generation_data:
        job_uuid = data["job_uuid"]
        positive_prompt = data["positive_prompt"]
        negative_prompt = data["negative_prompt"]
        writer.writerow([job_uuid, positive_prompt, negative_prompt, boltzman_temperature, boltzman_k])

    bytes_buffer = io.BytesIO(bytes(csv_buffer.getvalue(), "utf-8"))

    date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%Y-%m-%d')
    # get final filename
    sequence = 0
    # if exist, increment sequence
    while True:
        filename = "{}-independent-approx-v1-{:02}-{}.csv".format(date_now, sequence, dataset_name)
        csv_path = os.path.join(dataset_name, "output/generated-prompts-csv", filename)

        exists = cmd.is_object_exists(minio_client, 'datasets', csv_path)
        if not exists:
            break

        sequence += 1

    # upload the csv
    cmd.upload_data(minio_client, 'datasets', csv_path, bytes_buffer)


def upload_score_probability_data_to_csv(minio_client,
                                         dataset_name,
                                         index_phrase_score_data,
                                         probability_arr,
                                         normalized_probability_arr,
                                         sorted_indexes,
                                         cumulative_probability_arr,
                                         boltzman_temperature,
                                         boltzman_k,
                                         type="positive"):
    print("Saving prompt generation data to csv...")
    # sort cumulative by original index
    sorted_cumulative_prob = [None] * len(cumulative_probability_arr)
    for i in range(len(cumulative_probability_arr)):
        index = sorted_indexes[i]
        sorted_cumulative_prob[index] = cumulative_probability_arr[i]

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow((["index", "phrase", "occurrences", "token length", "boltzman_temperature", "boltzman k", "score", "boltzman probability", "normalized probability", "cumulative probability"]))

    for index, data in index_phrase_score_data.items():
        phrase = data.phrase
        score = "{:f}".format(data.score)
        occurrences = data.occurrences
        token_length = data.token_length
        boltzman_prob = probability_arr[index]
        normalized_prob = normalized_probability_arr[index]
        cumulative_prob = sorted_cumulative_prob[index]
        writer.writerow([index, phrase, occurrences, token_length, boltzman_temperature, boltzman_k, score, boltzman_prob, normalized_prob, cumulative_prob])

    bytes_buffer = io.BytesIO(bytes(csv_buffer.getvalue(), "utf-8"))

    date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%Y-%m-%d')
    # get final filename
    sequence = 0
    # if exist, increment sequence
    while True:
        filename = "{}-phrase-scores-probability-{:02}-{}-{}.csv".format(date_now, sequence, dataset_name, type)
        csv_path = os.path.join(dataset_name, "output/generated-phrases-scores-probability-csv", filename)

        exists = cmd.is_object_exists(minio_client, 'datasets', csv_path)
        if not exists:
            break

        sequence += 1

    # upload the csv
    cmd.upload_data(minio_client, 'datasets', csv_path, bytes_buffer)


def generate_prompt(positive_phrase_scores_loader,
                    positive_phrase_origin_indexes,
                    positive_cumulative_probability_arr,
                    negative_phrase_scores_loader,
                    negative_phrase_origin_indexes,
                    negative_cumulative_probability_arr,
                    ):
    max_token_size = 75
    comma_token_size = 1

    positive_prompt_total_token_size = 0
    negative_prompt_total_token_size = 0
    positive_prompt = []
    negative_prompt = []
    positive_used_phrase_dict = {}
    negative_used_phrase_dict = {}

    positive_cumulative_probability_arr_min = positive_cumulative_probability_arr.min()
    positive_cumulative_probability_arr_max = positive_cumulative_probability_arr.max()
    # positive prompt
    while positive_prompt_total_token_size < max_token_size:
        random_float = random.uniform(positive_cumulative_probability_arr_min,
                                      positive_cumulative_probability_arr_max)
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

    negative_cumulative_probability_arr_min = negative_cumulative_probability_arr.min()
    negative_cumulative_probability_arr_max = negative_cumulative_probability_arr.max()
    # negative prompt
    while negative_prompt_total_token_size < max_token_size:
        random_float = random.uniform(negative_cumulative_probability_arr_min,
                                      negative_cumulative_probability_arr_max)
        random_index = find_first_element_binary_search(negative_cumulative_probability_arr, random_float)
        if random_index in negative_used_phrase_dict:
            # print("float={} index={}", random_float, random_index)
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


def generate_prompts(minio_client,
                     dataset_name,
                     positive_phrase_scores_loader,
                     positive_phrase_origin_indexes,
                     positive_cumulative_probability_arr,
                     negative_phrase_scores_loader,
                     negative_phrase_origin_indexes,
                     negative_cumulative_probability_arr,
                     prompt_count,
                     boltzman_temperature,
                     boltzman_k):
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
            if dataset_name in ["environmental", "propaganda-poster", "waifu", "test-generations"]:
                response = generate_image_generation_jobs_with_temperature(positive_prompt=positive_prompt,
                                                                           negative_prompt=negative_prompt,
                                                                           prompt_scoring_model="n/a",
                                                                           prompt_score=0.0,
                                                                           prompt_generation_policy="independent_approx_v1",
                                                                           top_k=0.0,
                                                                           dataset_name=dataset_name,
                                                                           boltzman_temperature=boltzman_temperature,
                                                                           boltzman_k=boltzman_k)
            elif dataset_name in ["character", "mech", "icons"]:
                mask_path = "./test/test_inpainting/icon_mask.png"
                if dataset_name == "character":
                    mask_path = "./test/test_inpainting/character_mask.png"
                elif dataset_name == "mech":
                    sizes = ["1x1", "1x2", "2x1", "2x2", "2x3", "3x2", "3x3"]
                    chosen_size = random.randint(0, len(sizes)-1)
                    size_str = sizes[chosen_size]
                    mask_path = "./input/mask/mech/mech_mask_{}.png".format(size_str)

                response = generate_inpainting_job_with_temperature(positive_prompt=positive_prompt,
                                                                    negative_prompt=negative_prompt,
                                                                    prompt_scoring_model="n/a",
                                                                    prompt_score=0.0,
                                                                    prompt_generation_policy="independent_approx_v1",
                                                                    top_k=0.0,
                                                                    dataset_name=dataset_name,
                                                                    boltzman_temperature=boltzman_temperature,
                                                                    boltzman_k=boltzman_k,
                                                                    init_img_path="./test/test_inpainting/white_512x512.jpg",
                                                                    mask_path=mask_path)
            else:
                raise Exception("dataset unsupported")

            job_uuid = response['uuid']
            data = {"job_uuid": job_uuid,
                    "positive_prompt": positive_prompt,
                    "negative_prompt": negative_prompt}
            generated_prompts.append(data)

        upload_prompt_generation_data_to_csv(minio_client=minio_client,
                                             dataset_name=dataset_name,
                                             prompt_generation_data=generated_prompts,
                                             boltzman_temperature=boltzman_temperature,
                                             boltzman_k=boltzman_k)


def get_cumulative_probability_arr(minio_client,
                                   dataset_name,
                                   index_phrase_score_data,
                                   boltzman_temperature,
                                   boltzman_k,
                                   type="positive"
                                   ):
    scores_arr = []
    for index, data in index_phrase_score_data.items():
        score = data.score
        scores_arr.append(score)

    scores_np_arr = np.array(scores_arr)

    probability_arr = np.exp(-(scores_np_arr/(boltzman_k*boltzman_temperature)))

    # normalize
    normalized_probability_arr = probability_arr/np.linalg.norm(probability_arr)
    # assert round(np.sum(normalized_probability_arr), 4) == 1.0, "sum={}".format(np.sum(normalized_probability_arr))

    # cumulative
    sorted_probability_arr = []
    sorted_indexes = sorted(range(len(normalized_probability_arr)), key=lambda x: normalized_probability_arr[x],
                                   reverse=True)
    for i in sorted_indexes:
        sorted_probability_arr.append(normalized_probability_arr[i])
    sorted_probability_arr = np.array(sorted_probability_arr)

    # get cumulative
    cumulative_probability_arr = sorted_probability_arr.cumsum()

    print("-------------------------------------------------------------------------------------")
    print("scores=", scores_np_arr)
    print("prob=", probability_arr)
    print("normalized=", normalized_probability_arr)


    upload_score_probability_data_to_csv(minio_client,
                                         dataset_name,
                                         index_phrase_score_data,
                                         probability_arr,
                                         normalized_probability_arr,
                                         sorted_indexes,
                                         cumulative_probability_arr,
                                         boltzman_temperature,
                                         boltzman_k,
                                         type)

    return sorted_indexes, cumulative_probability_arr


def run_prompt_generator(minio_client,
                         dataset_name,
                         positive_phrase_scores_csv,
                         negative_phrase_scores_csv,
                         prompt_count,
                         boltzman_temperature,
                         boltzman_k,
                         ):
    positive_phrase_scores_loader = BoltzmanPhraseScoresLoader(dataset_name=dataset_name,
                                                       phrase_scores_csv=positive_phrase_scores_csv,
                                                       minio_client=minio_client,
                                                       )
    positive_phrase_scores_loader.load_dataset()
    positive_phrase_origin_indexes, positive_cumulative_probability_arr = get_cumulative_probability_arr(minio_client=minio_client,
                                                                                                         dataset_name=dataset_name,
                                                                                                         index_phrase_score_data=positive_phrase_scores_loader.index_phrase_score_data,
                                                                                                         boltzman_temperature=boltzman_temperature,
                                                                                                         boltzman_k=boltzman_k,
                                                                                                         type="positive")

    negative_phrase_scores_loader = BoltzmanPhraseScoresLoader(dataset_name=dataset_name,
                                                       phrase_scores_csv=negative_phrase_scores_csv,
                                                       minio_client=minio_client,
                                                       )

    negative_phrase_scores_loader.load_dataset()
    negative_phrase_origin_indexes, negative_cumulative_probability_arr = get_cumulative_probability_arr(minio_client=minio_client,
                                                                                                         dataset_name=dataset_name,
                                                                                                         index_phrase_score_data=negative_phrase_scores_loader.index_phrase_score_data,
                                                                                                         boltzman_temperature=boltzman_temperature,
                                                                                                         boltzman_k=boltzman_k,
                                                                                                         type="negative")

    generate_prompts(minio_client,
                     dataset_name,
                     positive_phrase_scores_loader,
                     positive_phrase_origin_indexes,
                     positive_cumulative_probability_arr,
                     negative_phrase_scores_loader,
                     negative_phrase_origin_indexes,
                     negative_cumulative_probability_arr,
                     prompt_count,
                     boltzman_temperature,
                     boltzman_k)


def parse_args():
    parser = argparse.ArgumentParser(description="Prompt Job Generator using Independent Approx V1 csv results")
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--dataset-name', required=True, help='Name of the dataset to generate prompt jobs')
    parser.add_argument('--positive-phrase-scores-csv', required=True, help='Filename of the positive phrase scores csv')
    parser.add_argument('--negative-phrase-scores-csv', required=True, help='Filename of the negative phrase scores csv')
    parser.add_argument('--prompt-count', required=True, type=int, help='Number of prompt jobs to generate')
    parser.add_argument('--boltzman-k', default=1.0, type=float, help='K for boltzman probability')
    parser.add_argument('--boltzman-temperature', default=8, type=float, help='Temperature for boltzman probability')
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
                             args.prompt_count,
                             args.boltzman_temperature,
                             args.boltzman_k)
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
                                     args.prompt_count,
                                     args.boltzman_temperature,
                                     args.boltzman_k)
            except Exception as e:
                print("Error running prompt generator for {}: {}".format(dataset, e))


if __name__ == "__main__":
    main()
