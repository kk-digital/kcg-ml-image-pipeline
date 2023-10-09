
import argparse
import random
import sys
from PIL import Image

base_directory = "./"
sys.path.insert(0, base_directory)

from prompt_job_generator.http_requests.request import http_get_completed_jobs_count, http_get_in_progress_jobs_count, http_get_pending_jobs_count, http_get_dataset_list
from worker.prompt_generation.prompt_generator import (generate_inpainting_generation_jobs_using_generated_prompts_and_base_prompts,
                                                       generate_image_generation_jobs_using_generated_prompts_and_base_prompts)

def parse_args():
    parser = argparse.ArgumentParser(description="generate prompts")

    # Required parameters
    parser.add_argument("--base_prompts_path", type=str)

    return parser.parse_args()


class PromptJobGeneratorState:
    def __init__(self):
        # keep the dataset_rate in this dictionary
        # should update using orchestration api
        self.dataset_rate = {}
        # each dataset will have a list of masks
        # only relevent if its an inpainting job
        self.dataset_masks = {}
        # each dataset will have one callback to spawn the jobs
        self.dataset_callbacks = {}

    def register_callback(self, dataset, callback):
        self.dataset_callbacks[dataset] = callback

    def set_dataset_rate(self, dataset, rate):
        self.dataset_rate[dataset] = rate

    def get_dataset_rate(self, dataset):
        return self.dataset_rate[dataset]

    def add_dataset_mask(self, dataset, init_image_path, mask_path):
        if dataset not in self.dataset_masks:
            self.dataset_masks[dataset] = []

        self.dataset_masks[dataset].append({
            'init_image' : init_image_path,
            'mask' : mask_path
        })

    def get_random_dataset_mask(self, dataset):
        mask_list = self.dataset_masks[dataset]
        if mask_list is None:
            return None

        return random.choices(mask_list)

def generate_icon_generation_jobs(prompt_job_generator_state):
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

def generate_propaganda_posters_image_generation_jobs(prompt_job_generator_state):

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

def generate_mechs_image_generation_jobs(prompt_job_generator_state):

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

    prompt_job_generator_state = PromptJobGeneratorState()

    # Adding dataset masks
    prompt_job_generator_state.add_dataset_mask("icons", "./test/test_inpainting/white_512x512.jpg", "./test/test_inpainting/icon_mask.png")
    prompt_job_generator_state.add_dataset_mask("character", "./test/test_inpainting/white_512x512.jpg", "./test/test_inpainting/character_mask.png")

    # register function callbacks
    # used to spawn jobs for each job_type/dataset
    prompt_job_generator_state.register_callback("icons", generate_icon_generation_jobs)

    # get list of datasets
    list_datasets = http_get_dataset_list()

    # --- http get dataset rates
    # hard coded for now
    dataset_rates = {
        'icons': 4,
        'character' : 4,
        'mech' : 2,
        'propaganda-poster' : 1,
        'environment' : 1
    }

    for dataset in list_datasets:
        dataset_rate = dataset_rates[dataset]
        if dataset_rate is not None:
            prompt_job_generator_state.set_dataset_rate(dataset, dataset_rate)


    print(http_get_completed_jobs_count('icons'))
    print(http_get_in_progress_jobs_count('icons'))
    print(http_get_pending_jobs_count('icons'))


if __name__ == '__main__':
    main()