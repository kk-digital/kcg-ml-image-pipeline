import os
import sys
import io
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import random
import math
from datetime import datetime
from pytz import timezone

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.minio import cmd
from worker.prompt_generation.prompt_generator import generate_image_generation_jobs_with_temperature, generate_inpainting_job_with_temperature


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
        print(positive_prompt_total_token_size)
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


def generate_prompts_array(positive_phrase_scores_loader,
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

            data = {
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt
            }

            generated_prompts.append(data)

        return generated_prompts


def get_cumulative_probability_arr(minio_client,
                                   dataset_name,
                                   index_phrase_score_data,
                                   boltzman_temperature,
                                   boltzman_k,
                                   type="positive"
                                   ):
    energy_arr = []
    for index, data in index_phrase_score_data.items():
        score = data.energy_per_phrase
        energy_arr.append(score)

    energy_np_arr = np.array(energy_arr)
    # negate the energy
    energy_np_arr = -1.0 * energy_np_arr

    probability_arr = np.exp(-(energy_np_arr / (boltzman_k * boltzman_temperature)))

    # normalize
    normalized_probability_arr = probability_arr / np.sum(probability_arr)
    assert round(np.sum(normalized_probability_arr), 4) == 1.0, "sum={}".format(np.sum(normalized_probability_arr))

    # cumulative
    sorted_probability_arr = []
    sorted_indexes = sorted(range(len(normalized_probability_arr)), key=lambda x: normalized_probability_arr[x],
                            reverse=True)
    for i in sorted_indexes:
        sorted_probability_arr.append(normalized_probability_arr[i])
    sorted_probability_arr = np.array(sorted_probability_arr)

    cumulative_probability_arr = sorted_probability_arr.cumsum()

    print("-------------------------------------------------------------------------------------")
    print("energies=", energy_np_arr)
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


def get_cumulative_probability_arr_without_upload(
                                   index_phrase_score_data,
                                   boltzman_temperature,
                                   boltzman_k
                                   ):
    energy_arr = []
    for index, data in index_phrase_score_data.items():
        score = data.energy_per_phrase
        energy_arr.append(score)

    energy_np_arr = np.array(energy_arr)
    # negate the energy
    energy_np_arr = -1.0 * energy_np_arr

    probability_arr = np.exp(-(energy_np_arr/(boltzman_k*boltzman_temperature)))

    # normalize
    normalized_probability_arr = probability_arr/np.sum(probability_arr)
    assert round(np.sum(normalized_probability_arr), 4) == 1.0, "sum={}".format(np.sum(normalized_probability_arr))

    # cumulative
    sorted_probability_arr = []
    sorted_indexes = sorted(range(len(normalized_probability_arr)), key=lambda x: normalized_probability_arr[x],
                                   reverse=True)
    for i in sorted_indexes:
        sorted_probability_arr.append(normalized_probability_arr[i])
    sorted_probability_arr = np.array(sorted_probability_arr)

    cumulative_probability_arr = sorted_probability_arr.cumsum()
    return sorted_indexes, cumulative_probability_arr

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
    writer.writerow((["index", "phrase", "occurrences", "token length", "boltzman_temperature", "boltzman k", "energy_per_phrase", "boltzman probability", "normalized probability", "cumulative probability"]))

    for index, data in index_phrase_score_data.items():
        phrase = data.phrase
        energy_per_phrase = "{:f}".format(data.energy_per_phrase)
        occurrences = data.occurrences
        token_length = data.token_length
        boltzman_prob = probability_arr[index]
        normalized_prob = normalized_probability_arr[index]
        cumulative_prob = sorted_cumulative_prob[index]
        writer.writerow([index, phrase, occurrences, token_length, boltzman_temperature, boltzman_k, energy_per_phrase, boltzman_prob, normalized_prob, cumulative_prob])

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
