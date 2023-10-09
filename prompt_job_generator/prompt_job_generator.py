
import requests
import argparse
import sys


base_directory = "./"
sys.path.insert(0, base_directory)

from worker.prompt_generation.prompt_generator import (generate_inpainting_generation_jobs_using_generated_prompts_and_base_prompts,
                                                       generate_image_generation_jobs_using_generated_prompts_and_base_prompts)


SERVER_ADRESS = 'http://127.0.0.1:8000'

def http_get_completed_jobs_count(dataset_name: str):
    url = SERVER_ADRESS + "/job/count-completed?dataset=" + dataset_name

    try:
        response = requests.get(url)
    except Exception as e:
        print('request exception ', e)

    if response.status_code == 200:
        job_json = response.json()
        return job_json

    return None


def http_get_pending_jobs_count(dataset_name: str):
    url = SERVER_ADRESS + "/job/count-pending?dataset=" + dataset_name

    try:
        response = requests.get(url)
    except Exception as e:
        print('request exception ', e)

    if response.status_code == 200:
        job_json = response.json()
        return job_json

    return None


def http_get_in_progress_jobs_count(dataset_name: str):
    url = SERVER_ADRESS + "/job/count-in-progress?dataset=" + dataset_name

    try:
        response = requests.get(url)
    except Exception as e:
        print('request exception ', e)

    if response.status_code == 200:
        job_json = response.json()
        return job_json

    return None



def parse_args():
    parser = argparse.ArgumentParser(description="generate prompts")

    # Required parameters
    parser.add_argument("--base_prompts_path", type=str)

    return parser.parse_args()


def generate_icon_generation_jobs():
    csv_dataset_path = 'input/civitai_phrases_database_v6.csv'
    prompt_count = 5
    base_prompts_csv_path = 'input/base-prompts/icon/base-prompts-icon-2.csv'
    dataset_name = 'icons'
    csv_phrase_limit = 0
    positive_prefix = ""
    init_img_path = "./test/test_inpainting/white_512x512.jpg"
    mask_path = "./test/test_inpainting/icon_mask.png"

    generate_inpainting_generation_jobs_using_generated_prompts_and_base_prompts(
        csv_dataset_path=csv_dataset_path,
        prompt_count=prompt_count,
        base_prompts_csv_path=base_prompts_csv_path,
        dataset_name=dataset_name,
        csv_phrase_limit=csv_phrase_limit,
        positive_prefix=positive_prefix,
        init_img_path=init_img_path,
        mask_path=mask_path
    )

def generate_propaganda_posters_image_generation_jobs():

    csv_dataset_path = 'input/civitai_phrases_database_v6.csv'
    prompt_count = 5
    base_prompts_csv_path = 'input/base-prompts/propaganda-poster/base-prompts-propaganda-poster.csv'
    dataset_name = 'propaganda-poster'
    csv_phrase_limit = 0
    positive_prefix = ""

    generate_image_generation_jobs_using_generated_prompts_and_base_prompts(
        csv_dataset_path=csv_dataset_path,
        prompt_count=prompt_count,
        base_prompts_csv_path=base_prompts_csv_path,
        dataset_name=dataset_name,
        csv_phrase_limit=csv_phrase_limit,
        positive_prefix=positive_prefix,
    )

def generate_mechs_image_generation_jobs():

    csv_dataset_path = 'input/civitai_phrases_database_v6.csv'
    prompt_count = 5
    base_prompts_csv_path = 'input/base-prompts/mech/base-prompts-mechs.csv'
    dataset_name = 'mech'
    csv_phrase_limit = 0
    positive_prefix = ""

    generate_image_generation_jobs_using_generated_prompts_and_base_prompts(
        csv_dataset_path=csv_dataset_path,
        prompt_count=prompt_count,
        base_prompts_csv_path=base_prompts_csv_path,
        dataset_name=dataset_name,
        csv_phrase_limit=csv_phrase_limit,
        positive_prefix=positive_prefix,
    )

def main():
    args = parse_args()

    print(http_get_completed_jobs_count())
    print(http_get_in_progress_jobs_count())
    print(http_get_pending_jobs_count())


if __name__ == '__main__':
    main()