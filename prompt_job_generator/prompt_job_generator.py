
import argparse
import random
import sys
import time
import io

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.minio import cmd
from prompt_job_generator.http_requests.request import http_get_completed_jobs_count, http_get_in_progress_jobs_count, http_get_pending_jobs_count, http_get_dataset_list
from worker.prompt_generation.prompt_generator import (generate_inpainting_generation_jobs_using_generated_prompts_and_base_prompts,
                                                       generate_image_generation_jobs_using_generated_prompts_and_base_prompts,
                                                       initialize_prompt_list_from_csv)
from training_worker.ab_ranking.model.ab_ranking_efficient_net import ABRankingEfficientNetModel

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
        # keep the dataset_job_per_second in this dictionary
        # should update using orchestration api
        self.dataset_job_per_second = {}
        # each dataset will have a list of masks
        # only relevent if its an inpainting job
        self.dataset_masks = {}
        # each dataset will have one callback to spawn the jobs
        self.dataset_callbacks = {}
        # efficient net model we use for scoring prompts
        # each dataset will have its own  model
        # input : prompts
        # output : prompt_score
        self.prompt_efficient_net_model_dictionary = {}

        # minio connection
        self.minio_client = None

        self.phrases = None
        self.phrases_token_size = None
        self.positive_count_list = None
        self.negative_count_list = None

    def configure_minio(self, minio_access_key, minio_secret_key):
        self.minio_client = cmd.get_minio_client(minio_access_key, minio_secret_key)



    def load_efficient_net_model(self, dataset, dataset_bucket, model_path):

        efficient_net_model = ABRankingEfficientNetModel()

        model_file_data = cmd.get_file_from_minio(self.minio_client, dataset_bucket, model_path)

        if model_file_data is None:
            return

        # Create a BytesIO object and write the downloaded content into it
        byte_buffer = io.BytesIO()
        for data in model_file_data.stream(amt=8192):
            byte_buffer.write(data)
        # Reset the buffer's position to the beginning
        byte_buffer.seek(0)

        efficient_net_model.load(byte_buffer)

        self.prompt_efficient_net_model_dictionary[dataset] = efficient_net_model


    def load_prompt_list_from_csv(self, csv_dataset_path, csv_phrase_limit):
        phrases, phrases_token_size, positive_count_list, negative_count_list = initialize_prompt_list_from_csv(csv_dataset_path, csv_phrase_limit)

        self.phrases = phrases
        self.phrases_token_size = phrases_token_size
        self.positive_count_list = positive_count_list
        self.negative_count_list = negative_count_list

    def register_callback(self, dataset, callback):
        self.dataset_callbacks[dataset] = callback

    def get_callback(self, dataset):
        if dataset in self.dataset_callbacks:
            return self.dataset_callbacks[dataset]
        else:
            return None

    def set_dataset_rate(self, dataset, rate):
        self.dataset_rate[dataset] = rate

    def get_dataset_rate(self, dataset):
        if dataset in self.dataset_rate:
            return self.dataset_rate[dataset]
        else:
            return None

    def set_dataset_job_per_second(self, dataset, job_per_second):
        self.dataset_job_per_second[dataset] = job_per_second

    def get_dataset_job_per_second(self, dataset):
        if dataset in self.dataset_job_per_second:
            return self.dataset_job_per_second[dataset]
        else:
            return None

    def add_dataset_mask(self, dataset, init_image_path, mask_path):
        if dataset not in self.dataset_masks:
            self.dataset_masks[dataset] = []

        self.dataset_masks[dataset].append({
            'init_image' : init_image_path,
            'mask' : mask_path
        })

    def get_random_dataset_mask(self, dataset):
        if dataset in self.dataset_masks:
            mask_list = self.dataset_masks[dataset]
        else:
            mask_list = None

        if mask_list is None:
            return None
        random_index = random.randint(0, len(mask_list) - 1)
        return mask_list[random_index]

def generate_icon_generation_jobs(prompt_job_generator_state):
    prompt_count = 1
    base_prompts_csv_path = 'input/dataset-config/icon/base-prompts-icon-2.csv'
    dataset_name = 'icons'
    positive_prefix = ""
    init_img_path = "./test/test_inpainting/white_512x512.jpg"
    mask_path = "./test/test_inpainting/icon_mask.png"

    mask = prompt_job_generator_state.get_random_dataset_mask(dataset_name)
    print(mask)
    if mask != None:
        init_img_path = mask['init_image']
        mask_path = mask['mask']

    print(f"Adding '{dataset_name}' generation job")

    generate_inpainting_generation_jobs_using_generated_prompts_and_base_prompts(
        phrases=prompt_job_generator_state.phrases,
        phrases_token_size=prompt_job_generator_state.phrases_token_size,
        positive_count_list=prompt_job_generator_state.positive_count_list,
        negative_count_list=prompt_job_generator_state.negative_count_list,
        prompt_count=prompt_count,
        base_prompts_csv_path=base_prompts_csv_path,
        dataset_name=dataset_name,
        positive_prefix=positive_prefix,
        init_img_path=init_img_path,
        mask_path=mask_path
    )

def generate_character_generation_jobs(prompt_job_generator_state):
    prompt_count = 1
    base_prompts_csv_path = 'input/dataset-config/character/base-prompts-waifu.csv'
    dataset_name = "character"
    positive_prefix = ""
    init_img_path = "./test/test_inpainting/white_512x512.jpg"
    mask_path = "./test/test_inpainting/character_mask.png"

    mask = prompt_job_generator_state.get_random_dataset_mask(dataset_name)
    if mask != None:
        init_img_path = mask['init_image']
        mask_path = mask['mask']

    print(f"Adding '{dataset_name}' generation job")

    generate_inpainting_generation_jobs_using_generated_prompts_and_base_prompts(
        phrases=prompt_job_generator_state.phrases,
        phrases_token_size=prompt_job_generator_state.phrases_token_size,
        positive_count_list=prompt_job_generator_state.positive_count_list,
        negative_count_list=prompt_job_generator_state.negative_count_list,
        prompt_count=prompt_count,
        base_prompts_csv_path=base_prompts_csv_path,
        dataset_name=dataset_name,
        positive_prefix=positive_prefix,
        init_img_path=init_img_path,
        mask_path=mask_path
    )

def generate_propaganda_posters_image_generation_jobs(prompt_job_generator_state):

    prompt_count = 1
    base_prompts_csv_path = 'input/dataset-config/propaganda-poster/base-prompts-propaganda-poster.csv'
    dataset_name = 'propaganda-poster'
    positive_prefix = ""

    print(f"Adding '{dataset_name}' generation job")

    generate_image_generation_jobs_using_generated_prompts_and_base_prompts(
        phrases=prompt_job_generator_state.phrases,
        phrases_token_size=prompt_job_generator_state.phrases_token_size,
        positive_count_list=prompt_job_generator_state.positive_count_list,
        negative_count_list=prompt_job_generator_state.negative_count_list,
        prompt_count=prompt_count,
        base_prompts_csv_path=base_prompts_csv_path,
        dataset_name=dataset_name,
        positive_prefix=positive_prefix,
    )

def generate_mechs_image_generation_jobs(prompt_job_generator_state):
    prompt_count = 1
    base_prompts_csv_path = 'input/dataset-config/mech/base-prompts-mechs.csv'
    dataset_name = 'mech'
    positive_prefix = ""

    print(f"Adding '{dataset_name}' generation job")

    generate_image_generation_jobs_using_generated_prompts_and_base_prompts(
        phrases = prompt_job_generator_state.phrases,
        phrases_token_size=prompt_job_generator_state.phrases_token_size,
        positive_count_list=prompt_job_generator_state.positive_count_list,
        negative_count_list=prompt_job_generator_state.negative_count_list,
        prompt_count=prompt_count,
        base_prompts_csv_path=base_prompts_csv_path,
        dataset_name=dataset_name,
        positive_prefix=positive_prefix,
    )

def main():
    args = parse_args()

    minio_access_key = 'v048BpXpWrsVIHUfdAix'
    minio_secret_key = '4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu'
    csv_dataset_path = 'input/civitai_phrases_database_v6.csv'
    csv_phrase_limit = 0

    prompt_job_generator_state = PromptJobGeneratorState()

    prompt_job_generator_state.configure_minio(minio_access_key, minio_secret_key)

    # loading civitai prompt csv file
    prompt_job_generator_state.load_prompt_list_from_csv(csv_dataset_path, csv_phrase_limit)

    # Adding dataset masks
    prompt_job_generator_state.add_dataset_mask("icons", "./test/test_inpainting/white_512x512.jpg", "./test/test_inpainting/icon_mask.png")
    prompt_job_generator_state.add_dataset_mask("character", "./test/test_inpainting/white_512x512.jpg", "./test/test_inpainting/character_mask.png")

    # register function callbacks
    # used to spawn jobs for each job_type/dataset
    prompt_job_generator_state.register_callback("icons", generate_icon_generation_jobs)
    prompt_job_generator_state.register_callback("propaganda-poster", generate_propaganda_posters_image_generation_jobs)
    prompt_job_generator_state.register_callback("mech", generate_mechs_image_generation_jobs)
    prompt_job_generator_state.register_callback("character", generate_character_generation_jobs)

    prompt_job_generator_state.load_efficient_net_model('character', 'datasets',
                                          'character/models/ab_ranking_efficient_net/2023-10-10.pth')

    return
    # get list of datasets
    list_datasets = http_get_dataset_list()

    # --- http get dataset rates
    # hard coded for now
    dataset_rates_dictionary = {
        'icons': 4,
        'character' : 4,
        'mech' : 2,
        'propaganda-poster' : 1,
        'environmental' : 1
    }

    # hard coded for now
    # TODO use orchestration api to get those values
    dataset_job_per_second_dictionary = {
        'icons': 0.2,
        'character': 0.2,
        'mech': 0.2,
        'propaganda-poster': 0.5,
        'environmental': 0.5
    }

    while True:
        # Update the dataset rates
        # TODO use orchestration api instead of hard coded values
        for dataset in list_datasets:
            if dataset in dataset_rates_dictionary:
                dataset_rate = dataset_rates_dictionary[dataset]
            else:
                dataset_rate = None

            if dataset_rate is not None:
                prompt_job_generator_state.set_dataset_rate(dataset, dataset_rate)

            # Update the dataset job per second value
            # TODO use orchestration api instead of hard coded values
            for dataset in list_datasets:
                if dataset in dataset_job_per_second_dictionary:
                    dataset_job_per_second = dataset_job_per_second_dictionary[dataset]
                else:
                    dataset_job_per_second = None

                if dataset_job_per_second is not None:
                    prompt_job_generator_state.set_dataset_job_per_second(dataset, dataset_job_per_second)


        # dictionary that maps dataset => number of jobs to add
        dataset_jobs_to_add = {}

        for dataset in list_datasets:
            dataset_rate = prompt_job_generator_state.get_dataset_rate(dataset)
            dataset_job_per_second = prompt_job_generator_state.get_dataset_job_per_second(dataset)

            # if dataset_rate is not found just move on
            if dataset_rate == None:
                print("dataset rate not found for dataset ", dataset)
                continue

            # if dataset_job_per_second is not found just move on
            if dataset_job_per_second == None:
                print("dataset dataset_job_per_second not found for dataset ", dataset)
                continue

            # get the number of jobs available for the dataset
            in_progress_job_count = http_get_in_progress_jobs_count(dataset)
            pending_job_count = http_get_pending_jobs_count(dataset)

            # Target number of Jobs in Queue
            # Equals: Time Speed (Jobs/Second) times 60*5 (300); 5 minutes
            target_job_count = 60*5 * dataset_job_per_second * dataset_rate

            # get total number of jobs

            total_jobs_in_queue_count = in_progress_job_count + pending_job_count
            print(dataset, ": total_jobs_in_queue_count ", total_jobs_in_queue_count)
            number_of_jobs_to_add = 0

            if target_job_count > total_jobs_in_queue_count:
                number_of_jobs_to_add = target_job_count - total_jobs_in_queue_count

            dataset_jobs_to_add[dataset] = number_of_jobs_to_add



        # Make sure we stop lopping
        # If there are no added jobs
        added_atleast_one_job = True

        while added_atleast_one_job:
            added_atleast_one_job = False

            for dataset in list_datasets:
                if dataset in dataset_jobs_to_add:
                    number_of_jobs_to_add = dataset_jobs_to_add[dataset]
                else:
                    number_of_jobs_to_add = None

                # check if there is a missing value
                # and skip the dataset
                if number_of_jobs_to_add == None:
                    continue

                # if there are no jobs to add
                # skip the dataset
                if number_of_jobs_to_add <= 0:
                    continue

                # get dataset callback
                # used to spawn the job
                # if the callback is not found
                # just move on
                dataset_callback = prompt_job_generator_state.get_callback(dataset)

                if dataset_callback == None:
                    print("dataset callback not found for dataset ", dataset)
                    continue

                print(f'number of jobs to spawn for dataset {dataset} is {number_of_jobs_to_add}')
                # Adding a job
                dataset_callback(prompt_job_generator_state)

                dataset_jobs_to_add[dataset] = number_of_jobs_to_add - 1
                added_atleast_one_job = True

        # sleep for n number of seconds
        time_to_sleep_in_seconds = 2

        time.sleep(time_to_sleep_in_seconds)

if __name__ == '__main__':
    main()