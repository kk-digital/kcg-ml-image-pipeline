"""
Note: You have to run image_delta_scorer.py script first for the
latest dataset so the generated prompts' scores are updated.
"""

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

base_directory = "./"
sys.path.insert(0, base_directory)

from scripts.image_scorer import ImageScorer
from utility.http.request import http_get_completed_job_by_uuid
from utility.minio import cmd
from data_loader.generated_prompts_loader import GeneratedPromptsLoader



def get_updated_generated_prompt_scores(generated_prompt,
                                        index):
    job_uuid = generated_prompt.job_uuid
    completed_job_data = http_get_completed_job_by_uuid(job_uuid)
    text_embedding_score = completed_job_data["task_attributes_dict"]["text_embedding_score"]
    text_embedding_sigma_score = completed_job_data["task_attributes_dict"]["text_embedding_sigma_score"]
    text_embedding_percentile = completed_job_data["task_attributes_dict"]["text_embedding_percentile"]
    image_clip_score = completed_job_data["task_attributes_dict"]["image_clip_score"]
    image_clip_sigma_score = completed_job_data["task_attributes_dict"]["image_clip_sigma_score"]
    image_clip_percentile = completed_job_data["task_attributes_dict"]["image_clip_percentile"]
    delta_sigma_score = completed_job_data["task_attributes_dict"]["delta_sigma_score"]
    generated_prompt.update_attributes(image_clip_score,
                                       image_clip_sigma_score,
                                       image_clip_percentile,
                                       text_embedding_score,
                                       text_embedding_sigma_score,
                                       text_embedding_percentile,
                                       delta_sigma_score)

    return  generated_prompt, index


def get_generated_prompts_scores(minio_client,
                                 dataset_name,
                                 generated_prompts_csv):

    start_time = time.time()

    generated_prompts_loader = GeneratedPromptsLoader(
        dataset_name=dataset_name,
        generated_prompts_csv=generated_prompts_csv,
        minio_client=minio_client
    )
    generated_prompts_loader.load_dataset()

    print("Getting scores data and updating generated prompt data")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(len(generated_prompts_loader.generated_prompt_data_arr)):
            generated_prompt = generated_prompts_loader.generated_prompt_data_arr[i]

            futures.append(executor.submit(get_updated_generated_prompt_scores,
                                           generated_prompt=generated_prompt,
                                           index=i,))

        for future in tqdm(as_completed(futures), total=len(futures)):
            updated_prompt_data, index = future.result()
            generated_prompts_loader.generated_prompt_data_arr[index] = updated_prompt_data


    # save again after updating
    generated_prompts_loader.save_updated_csv()

    time_elapsed = time.time() - start_time
    print("Dataset: {}: Total Time elapsed: {}s".format(dataset_name, format(time_elapsed, ".2f")))


def parse_args():
    parser = argparse.ArgumentParser(description="Get Generated Prompt Scores")
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--dataset-name', required=True, help='Name of the dataset for embeddings')
    parser.add_argument('--generated-prompts-csv', required=True, help='Filename of the generated prompt csv in minio')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)

    get_generated_prompts_scores(minio_client,
                                 args.dataset_name,
                                 args.generated_prompts_csv)


if __name__ == "__main__":
    main()
