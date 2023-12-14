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


def get_gibs_probability(energy_one,
                         energy_two,
                         gibs_temperature,
                         gibs_k):
    prob = np.exp(-(energy_one-energy_two)/(gibs_k*gibs_temperature))

    return prob


def swap_phrases_based_on_gibs_sampling(initial_prompt_indices,
                                        phrase_scores_loader,
                                        gibs_temperature,
                                        gibs_k):
    phrase_data_total_size = phrase_scores_loader.get_phrase_data_total_size()
    prompt_len = len(initial_prompt_indices)
    number_of_phrases_to_swap = random.randint(round(prompt_len/2), prompt_len)
    num_swapped = 0

    swapped_index_dict = {}
    used_phrase_dict = {}
    while num_swapped < number_of_phrases_to_swap:
        rand_prompt_index = random.randint(0, prompt_len-1)
        if rand_prompt_index in swapped_index_dict:
            continue

        rand_phrase_index = random.randint(0, phrase_data_total_size-1)
        if rand_phrase_index in used_phrase_dict:
            continue

        energy_one = phrase_scores_loader.get_phrase_energy(initial_prompt_indices[rand_prompt_index])
        energy_two = phrase_scores_loader.get_phrase_energy(rand_phrase_index)
        swap_probability = get_gibs_probability(energy_one,
                                    energy_two,
                                    gibs_temperature,
                                    gibs_k)
        print("swap prob=", swap_probability)
        rand_float = random.uniform(0, 5)
        if rand_float <= swap_probability:
            # then swap
            initial_prompt_indices[rand_prompt_index] = rand_phrase_index
            swapped_index_dict[rand_prompt_index] = 1
            used_phrase_dict[rand_phrase_index] = 1
            num_swapped += 1

    return initial_prompt_indices


def generate_prompt_using_uniform_random(phrase_scores_loader):
    max_token_size = 75
    comma_token_size = 1

    prompt_total_token_size = 0
    prompt = []
    prompt_index = []
    used_phrase_dict = {}

    phrase_scores_len = phrase_scores_loader.get_phrase_data_total_size()
    # prompt
    while prompt_total_token_size < max_token_size:
        random_index = random.randint(0, phrase_scores_len-1)
        if random_index in used_phrase_dict:
            continue

        random_phrase = phrase_scores_loader.get_phrase(random_index)

        chosen_phrase_size = phrase_scores_loader.get_token_size(random_index)
        sum_token_size = prompt_total_token_size + chosen_phrase_size + comma_token_size
        if sum_token_size < max_token_size:
            # update used array
            used_phrase_dict[random_index] = 1
            prompt.append(random_phrase)
            prompt_index.append(random_index)
            prompt_total_token_size = sum_token_size
        else:
            break

    return prompt_index


def generate_prompt(positive_phrase_scores_loader,
                    negative_phrase_scores_loader,
                    gibs_temperature,
                    gibs_k,
                    ):
    initial_positive_prompt_indices = generate_prompt_using_uniform_random(positive_phrase_scores_loader)
    final_positive_prompt_indices = swap_phrases_based_on_gibs_sampling(initial_positive_prompt_indices,
                                                              positive_phrase_scores_loader,
                                                              gibs_temperature,
                                                              gibs_k)

    initial_negative_prompt_indices = generate_prompt_using_uniform_random(negative_phrase_scores_loader)
    final_negative_prompt_indices = swap_phrases_based_on_gibs_sampling(initial_negative_prompt_indices,
                                                              negative_phrase_scores_loader,
                                                              gibs_temperature,
                                                              gibs_k)

    positive_prompt_str = ', '.join([positive_phrase_scores_loader.get_phrase(prompt_index) for prompt_index in final_positive_prompt_indices])
    negative_prompt_str = ', '.join([negative_phrase_scores_loader.get_phrase(prompt_index) for prompt_index in final_negative_prompt_indices])

    prompt = (positive_prompt_str, negative_prompt_str)

    return prompt


def generate_prompts(minio_client,
                     dataset_name,
                     positive_phrase_scores_loader,
                     negative_phrase_scores_loader,
                     prompt_count,
                     gibs_temperature,
                     gibs_k):
    generated_prompts = []

    print("Generating {} prompts...".format(prompt_count))
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(prompt_count):
            futures.append(executor.submit(generate_prompt,
                                           positive_phrase_scores_loader=positive_phrase_scores_loader,
                                           negative_phrase_scores_loader=negative_phrase_scores_loader,
                                           gibs_temperature=gibs_temperature,
                                           gibs_k=gibs_k))

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
                                                                           prompt_generation_policy="independent_approx_v1_gibs",
                                                                           top_k=0.0,
                                                                           dataset_name=dataset_name,
                                                                           boltzman_temperature=gibs_temperature,
                                                                           boltzman_k=gibs_k)
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
                                                                    prompt_generation_policy="independent_approx_v1_gibs",
                                                                    top_k=0.0,
                                                                    dataset_name=dataset_name,
                                                                    boltzman_temperature=gibs_temperature,
                                                                    boltzman_k=gibs_k,
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
                                             gibs_temperature=gibs_temperature,
                                             gibs_k=gibs_k)


def upload_prompt_generation_data_to_csv(minio_client,
                                         dataset_name,
                                         prompt_generation_data,
                                         gibs_temperature,
                                         gibs_k):
    print("Saving prompt generation data to csv...")
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow((["job_uuid", "positive_prompt", "negative_prompt", "gibs temperature", "gibs k"]))

    for data in prompt_generation_data:
        job_uuid = data["job_uuid"]
        positive_prompt = data["positive_prompt"]
        negative_prompt = data["negative_prompt"]
        writer.writerow([job_uuid, positive_prompt, negative_prompt, gibs_temperature, gibs_k])

    bytes_buffer = io.BytesIO(bytes(csv_buffer.getvalue(), "utf-8"))

    date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%Y-%m-%d')
    # get final filename
    sequence = 0
    # if exist, increment sequence
    while True:
        filename = "{}-independent-approx-v1-gibs-{:02}-{}.csv".format(date_now, sequence, dataset_name)
        csv_path = os.path.join(dataset_name, "output/generated-prompts-csv", filename)

        exists = cmd.is_object_exists(minio_client, 'datasets', csv_path)
        if not exists:
            break

        sequence += 1

    # upload the csv
    cmd.upload_data(minio_client, 'datasets', csv_path, bytes_buffer)
