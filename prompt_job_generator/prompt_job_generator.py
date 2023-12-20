
import argparse
import sys
import time
import threading
import os
import re
from datetime import datetime

base_directory = "./"
sys.path.insert(0, base_directory)

from prompt_job_generator_state import PromptJobGeneratorState
from utility.minio.cmd import get_list_of_objects_with_prefix
from prompt_job_generator_functions import (generate_icon_generation_jobs, generate_character_generation_jobs, generate_mechs_image_generation_jobs,
generate_propaganda_posters_image_generation_jobs, generate_environmental_image_generation_jobs, generate_waifu_image_generation_jobs)
from prompt_job_generator.http_requests.request import (http_get_in_progress_jobs_count, http_get_pending_jobs_count, http_get_dataset_list,
                                                        http_get_dataset_job_per_second, http_get_jobs_count_last_hour,
                                                        http_get_all_dataset_config, http_get_dataset_model_list, http_get_dataset_latest_ranking_model,
                                                        http_set_dataset_ranking_model, http_get_model_id)
from prompt_job_generator_constants import JOB_PER_SECOND_SAMPLE_SIZE, DEFAULT_TOP_K_VALUE, DEFAULT_DATASET_RATE

from utility.path import separate_bucket_and_file_path


def parse_args():
    parser = argparse.ArgumentParser(description="generate prompts")

    # Required parameters
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--minio_access_key", type=str, default='v048BpXpWrsVIHUfdAix')
    parser.add_argument("--minio_secret_key", type=str, default='4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu')
    parser.add_argument("--csv_dataset_path", type=str, default='input/civitai_phrases_database_v7_no_nsfw.csv')

    return parser.parse_args()


def update_dataset_prompt_queue(prompt_job_generator_state, dataset):
    prompt_queue = prompt_job_generator_state.prompt_queue

    prompt_queue.update(prompt_job_generator_state, dataset)


def update_datasets_prompt_queue(prompt_job_generator_state, list_datasets):
    # if dataset list is null return
    if list_datasets is None:
        return

    thread_list = []

    for dataset in list_datasets:
        #thread = threading.Thread(target=update_dataset_prompt_queue,
        #                          args=(prompt_job_generator_state, dataset, ))
        #thread.start()
        #thread_list.append(thread)
        update_dataset_prompt_queue(prompt_job_generator_state, dataset)

    for thread in thread_list:
        thread.join()


def update_database_model_list(prompt_job_generator_state, list_datasets):

    # if dataset list is null return
    if list_datasets is None:
        return

    # loop through all datasets and
    # for each dataset update the model_list
    # from orchestration api
    for dataset in list_datasets:

        dataset_model_list = http_get_dataset_model_list(dataset)

        if dataset_model_list is None:
            continue

        dataset_model_dictionary = {}

        for item in dataset_model_list:
            model_name = item['model_name']
            dataset_model_dictionary[model_name] = item

        prompt_job_generator_state.set_dataset_model_list(dataset, dataset_model_dictionary)


def update_dataset_latest_ranking_model(prompt_job_generator_state, list_datasets):
    # if dataset list is null return
    if list_datasets is None:
        return

    # loop through all datasets and
    # for each dataset update the
    # latest ranking model
    for dataset in list_datasets:
        latest_ranking_model = http_get_dataset_latest_ranking_model(dataset)

        if latest_ranking_model is None:
            continue

        model_file_hash = latest_ranking_model['model_file_hash']
        model_id = http_get_model_id(model_file_hash)
        model_name = latest_ranking_model['model_name']

        if model_id is None:
            continue

        model_id = int(model_id)
        latest_ranking_model['model_id'] = model_id
        if model_name == 'latest':
            dataset_latest_model = http_get_dataset_latest_ranking_model(dataset)
            if dataset_latest_model is not None:
                model_name = dataset_latest_model['model_name']

        if model_name == 'latest':
            print('Could not find latest model for some reason !')


        http_set_dataset_ranking_model(dataset, model_name)


def extract_dates_from_strings(string_list):
    date_pattern = r'\b\d{4}-\d{2}-\d{2}\b'  # Regular expression for the date pattern

    date_list = []
    for text in string_list:
        matches = re.findall(date_pattern, text)
        date_list.extend(matches)

    return date_list


# should only be used if we are using independent approx v1 to generate prompts
def load_dataset_prompt_gen_approx_v1(prompt_job_generator_state, dataset):
    minio_client = prompt_job_generator_state.minio_client

    # get the list of all csv files
    # in the minio directory
    # the goal is to filter by date
    # we are interested in the latest csv files
    csv_list = get_list_of_objects_with_prefix(minio_client, 'datasets',
                                               f'{dataset}/output/phrases-score-csv/')

    date_string_list = extract_dates_from_strings(csv_list)

    date_objects = []

    # go through date strings and
    # convert to date class to compare them easily
    for date_string in date_string_list:
        date = datetime.strptime(date_string, "%Y-%m-%d")
        date_objects.append(date)

    # Find the latest date using the max function
    latest_date = max(date_objects)
    # Format the latest date as a string
    latest_date_str = latest_date.strftime("%Y-%m-%d")

    positive_phrase_scores_csv = f'{latest_date_str}-positive-phrases-score.csv'
    negative_phrase_scores_csv = f'{latest_date_str}-negative-phrases-score.csv'
    prompt_job_generator_state.prompt_queue.load_prompt_approx_v1(dataset,
                                                                  minio_client,
                                                                  positive_phrase_scores_csv,
                                                                  negative_phrase_scores_csv)


def update_dataset_config_data(prompt_job_generator_state, list_datasets):

    # if dataset list is null return
    if list_datasets is None:
        return

    dataset_config_json = http_get_all_dataset_config()
    dataset_config_dictionary = {}

    for dataset_config in dataset_config_json:
        dataset = dataset_config['dataset_name']

        dataset_config_dictionary[dataset] = dataset_config

    # loop through all datasets and
    # for each dataset update the dataset_rate
    # from orchestration api rates
    total_rate = 0
    for dataset in list_datasets:

        if dataset not in dataset_config_dictionary:
            continue

        dataset_data = dataset_config_dictionary[dataset]

        # the number type fields we get from the orchestration api
        # have to be converted from string to number
        # convert the string to float
        if 'dataset_rate' in dataset_data:
            dataset_rate = float(dataset_data['dataset_rate'])
            dataset_data['dataset_rate'] = dataset_rate
        else:
            dataset_rate = DEFAULT_DATASET_RATE

        if 'top_k' in dataset_data:
            dataset_top_k = float(dataset_data['top_k'])
            dataset_data['dataset_top_k'] = dataset_top_k
        else:
            dataset_top_k = DEFAULT_TOP_K_VALUE

        total_rate += dataset_rate

        if 'ranking_model' in dataset_data:
            model_name = dataset_data['ranking_model']
            if model_name is not None and model_name == 'latest':
                model_info = http_get_dataset_latest_ranking_model(dataset)

                if model_info is not None:
                    new_model_name = model_info['model_name']

                    dataset_data['ranking_model'] = new_model_name

        if 'generation_policy' in dataset_data:
            if dataset_data['generation_policy'] == 'independent-approx-v1-top-k':
                load_dataset_prompt_gen_approx_v1(prompt_job_generator_state, dataset)

        prompt_job_generator_state.set_dataset_data(dataset, dataset_data)

    prompt_job_generator_state.set_total_rate(total_rate)


def update_dataset_job_queue_size(prompt_job_generator_state, list_datasets):

    # if dataset list is null return
    if list_datasets is None:
        return

    # loop through all datasets and
    # for each dataset update the job_queue_size & job_queue_target
    # from orchestration api rates
    for dataset in list_datasets:

        # get the number of jobs available for the dataset
        in_progress_job_count = http_get_in_progress_jobs_count(dataset)
        pending_job_count = http_get_pending_jobs_count(dataset)
        job_per_second = http_get_dataset_job_per_second(dataset, JOB_PER_SECOND_SAMPLE_SIZE)
        jobs_count_last_hour = http_get_jobs_count_last_hour(dataset)

        if job_per_second is None:
            job_per_second = 0.2

        if job_per_second == 0:
            job_per_second = 0.2

        # TODO remove this bullshit
        if job_per_second > 1:
            job_per_second = 0.2

        if in_progress_job_count is None or pending_job_count is None:
            continue

        # get the hourly job limit
        jobs_hourly_limit = prompt_job_generator_state.get_dataset_hourly_limit(dataset)

        # the number of jobs we are allowed to add
        maximum_jobs_to_add = jobs_hourly_limit - jobs_count_last_hour

        # make sure the maximum jobs to add is positive
        if maximum_jobs_to_add < 0:
            maximum_jobs_to_add = 0

        job_queue_size = in_progress_job_count + pending_job_count
        # Target number of Jobs in Queue
        # Equals: Time Speed (Jobs/Second) times 60*5 (300); 5 minutes
        job_queue_target = int(60 * 5 * job_per_second)

        # make sure the queue target size is allways smaller than the maximum queue size
        if job_queue_target > maximum_jobs_to_add:
            job_queue_target = maximum_jobs_to_add

        prompt_job_generator_state.set_dataset_job_queue_size(dataset, job_queue_size)
        prompt_job_generator_state.set_dataset_job_queue_target(dataset, job_queue_target)


def load_dataset_models(prompt_job_generator_state, dataset_list):

    if dataset_list is None:
        return

    for dataset in dataset_list:
        model_name = prompt_job_generator_state.get_dataset_ranking_model(dataset)

        if model_name is None:
            continue

        model_info = prompt_job_generator_state.get_dataset_model_info(dataset, model_name)

        if model_info is None:
            continue

        if not isinstance(model_info, dict):
            continue

        model_type = model_info['model_type']

        model_path = model_info['model_path']


        bucket_name, file_path = separate_bucket_and_file_path(model_path)

        if model_type == 'image-pair-ranking-efficient-net':
            prompt_job_generator_state.load_efficient_net_model(bucket_name, 'datasets', model_path)
        elif model_type == 'ab_ranking_efficient_net':
            prompt_job_generator_state.load_efficient_net_model(bucket_name, 'datasets', model_path)
        elif model_type == 'ab_ranking_linear':
            prompt_job_generator_state.load_linear_model(bucket_name, 'datasets', model_path)
        elif model_type == 'image-pair-ranking-linear':
            prompt_job_generator_state.load_linear_model(bucket_name, 'datasets', model_path)
        elif model_type == 'ab_ranking_elm_v1':
            prompt_job_generator_state.load_elm_v1_model(bucket_name, 'datasets', model_path)
        elif model_type == 'image-pair-ranking-elm-v1':
            prompt_job_generator_state.load_elm_v1_model(bucket_name, 'datasets', model_path)

        scoring_model = prompt_job_generator_state.get_dataset_scoring_model(dataset)

def update_dataset_prompt_queue_background_thread(prompt_job_generator_state):

    while True:
        # get list of datasets
        list_datasets = http_get_dataset_list()

        update_datasets_prompt_queue(prompt_job_generator_state, list_datasets)

        sleep_time_in_seconds = 1.0
        time.sleep(sleep_time_in_seconds)


def update_dataset_values_background_thread(prompt_job_generator_state):

    while True:
        # get list of datasets
        list_datasets = http_get_dataset_list()

        update_database_model_list(prompt_job_generator_state, list_datasets)

        # not sure if we want to keep this one
        # this will force the prompt generator to use
        # the latest ranking model
        update_dataset_latest_ranking_model(prompt_job_generator_state, list_datasets)

        update_dataset_config_data(prompt_job_generator_state, list_datasets)
        update_dataset_job_queue_size(prompt_job_generator_state, list_datasets)

        load_dataset_models(prompt_job_generator_state, list_datasets)

        sleep_time_in_seconds = 2.0
        time.sleep(sleep_time_in_seconds)


def find_png_files(folder_path):
    jpg_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.png'):
                jpg_file_path = os.path.join(root, file)
                jpg_files.append(jpg_file_path)

    return jpg_files


def main():
    args = parse_args()

    device = args.device
    minio_access_key = args.minio_access_key
    minio_secret_key = args.minio_secret_key
    csv_dataset_path = args.csv_dataset_path
    csv_phrase_limit = 0

    prompt_job_generator_state = PromptJobGeneratorState(device=device)

    prompt_job_generator_state.configure_minio(minio_access_key, minio_secret_key)
    prompt_job_generator_state.load_clip_model()

    # loading civitai prompt csv file
    prompt_job_generator_state.load_prompt_list_from_csv(csv_dataset_path, csv_phrase_limit)

    # Adding dataset masks
    prompt_job_generator_state.add_dataset_mask("icons", "./test/test_inpainting/white_512x512.jpg",
                                                "./test/test_inpainting/icon_mask.png")
    prompt_job_generator_state.add_dataset_mask("character", "./test/test_inpainting/white_512x512.jpg",
                                                "./test/test_inpainting/character_mask.png")

    mech_masks = find_png_files('./input/mask/mech')
    for mask in mech_masks:
        prompt_job_generator_state.add_dataset_mask("mech", "./test/test_inpainting/white_512x512.jpg",
                                                    mask)

    # register function callbacks
    # used to spawn jobs for each job_type/dataset
    # when we want to spawn a job for a specific dataset
    # we call this function
    prompt_job_generator_state.register_callback("icons", generate_icon_generation_jobs)
    prompt_job_generator_state.register_callback("propaganda-poster", generate_propaganda_posters_image_generation_jobs)
    prompt_job_generator_state.register_callback("mech", generate_mechs_image_generation_jobs)
    prompt_job_generator_state.register_callback("character", generate_character_generation_jobs)
    prompt_job_generator_state.register_callback("environmental", generate_environmental_image_generation_jobs)
    prompt_job_generator_state.register_callback("waifu", generate_waifu_image_generation_jobs)

    # setting the base prompt csv for each dataset
    prompt_job_generator_state.prompt_queue.set_dataset_base_prompt('icons',
                                                                    'input/dataset-config/icon/base-prompts-dsp.csv')
    prompt_job_generator_state.prompt_queue.set_dataset_base_prompt('propaganda-poster',
                                                                    'input/dataset-config/propaganda-poster/base-prompts-propaganda-poster.csv')
    prompt_job_generator_state.prompt_queue.set_dataset_base_prompt('mech',
                                                                    'input/dataset-config/mech/base-prompts-dsp.csv')
    prompt_job_generator_state.prompt_queue.set_dataset_base_prompt('character',
                                                                    'input/dataset-config/character/base-prompts-waifu.csv')
    prompt_job_generator_state.prompt_queue.set_dataset_base_prompt('waifu',
                                                                    'input/dataset-config/character/base-prompts-waifu.csv')
    prompt_job_generator_state.prompt_queue.set_dataset_base_prompt('environmental',
                                                                    'input/dataset-config/environmental/base-prompts-environmental.csv')

    # get list of datasets
    list_datasets = http_get_dataset_list()

    update_database_model_list(prompt_job_generator_state, list_datasets)
    # not sure if we want to keep this one
    # this will force the prompt generator to use
    # the latest ranking model
    update_dataset_latest_ranking_model(prompt_job_generator_state, list_datasets)
    update_dataset_config_data(prompt_job_generator_state, list_datasets)
    update_dataset_job_queue_size(prompt_job_generator_state, list_datasets)

    # load the models at the start for each dataset
    load_dataset_models(prompt_job_generator_state, list_datasets)

    print("generating starting prompts")

    # generate prompts in the prompt queue
    update_datasets_prompt_queue(prompt_job_generator_state, list_datasets)

    print("starting threads")

    thread = threading.Thread(target=update_dataset_values_background_thread, args=(prompt_job_generator_state,))
    thread.start()

    thread = threading.Thread(target=update_dataset_prompt_queue_background_thread, args=(prompt_job_generator_state,))
    thread.start()

    print('starting prompt job generator')
    while True:
        # dictionary that maps dataset => number of jobs to add
        dataset_number_jobs_to_add = {}

        for dataset in list_datasets:
            if dataset == 'test-generations':
                continue

            dataset_number_jobs_to_add[dataset] = 0

            dataset_rate = prompt_job_generator_state.get_dataset_rate(dataset)
            dataset_job_queue_size = prompt_job_generator_state.get_dataset_job_queue_size(dataset)
            dataset_job_queue_target = prompt_job_generator_state.get_dataset_job_queue_target(dataset)

            # if dataset_rate is not found just move on
            if dataset_rate == None:
                print("dataset rate not found for dataset ", dataset)
                continue

            if dataset_job_queue_size is None:
                print("dataset job queue size is not found for dataset : ", dataset)
                continue

            if dataset_job_queue_target is None:
                print("dataset job queue target is not found for dataset : ", dataset)
                continue

            number_of_jobs_to_add = 0

            if dataset_job_queue_target > dataset_job_queue_size:
                number_of_jobs_to_add = dataset_job_queue_target - dataset_job_queue_size

            dataset_number_jobs_to_add[dataset] = number_of_jobs_to_add


        # If JobQueueSize < JobQueueTarget
        #- then keep "updating"/ adding
        #- for each Dataset, TodoJob[i] += DatasetRate[i] / TotalRate
        #- then at end of loop, if >1.0, then emit job for that dataset
        dataset_todo_jobs = {}
        for dataset in list_datasets:
            dataset_todo_jobs[dataset] = 0

        # Make sure we stop lopping
        # If there are no added jobs
        added_atleast_one_job = True

        while added_atleast_one_job:
            added_atleast_one_job = False

            for dataset in list_datasets:
                # get dataset rate
                # dataset rates should update in background using
                # orchestration api
                dataset_rate = prompt_job_generator_state.get_dataset_rate(dataset)
                total_rate = prompt_job_generator_state.total_rate

                # if dataset_rate does not exist skip this dataset
                if dataset_rate is None:
                    continue

                if not prompt_job_generator_state.prompt_queue.database_prompt_available(dataset):
                    # print('no prompt is available for dataset ', dataset)
                    continue

                # get dataset callback
                # used to spawn the job
                # if the callback is not found
                # just move on
                dataset_callback = prompt_job_generator_state.get_callback(dataset)

                if dataset_callback == None:
                    print("dataset callback not found for dataset ", dataset)
                    continue

                number_of_jobs_to_add = dataset_number_jobs_to_add[dataset]

                if number_of_jobs_to_add >= 1 and dataset_rate > 0:
                    dataset_todo_jobs[dataset] += (dataset_rate / total_rate)
                    added_atleast_one_job = True

                if dataset_todo_jobs[dataset] >= 1.0:
                    # spawn job
                    dataset_todo_jobs[dataset] -= 1.0
                    prompt_job_generator_state.append_dataset_job_queue_size(dataset, 1)
                    dataset_number_jobs_to_add[dataset] = number_of_jobs_to_add - 1

                    print(f'number of jobs to spawn for dataset {dataset} is {number_of_jobs_to_add}')
                    # Adding a job
                    dataset_callback(prompt_job_generator_state)

        # sleep for n number of seconds
        time_to_sleep_in_seconds = 2
        time.sleep(time_to_sleep_in_seconds)

if __name__ == '__main__':
    main()