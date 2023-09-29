import argparse
import os
import sys
import time
import random
import tiktoken
import sys
import os
import shutil
import json
import math
import csv
import torch
import uuid
from tqdm import tqdm
import numpy as np

base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from worker.http import request
from worker.generation_task.generation_task import GenerationTask


class GeneratedPrompt:
    def __init__(self, positive_prompt_str: str, negative_prompt_str: str, num_topics: int, num_modifiers: int,
                 num_styles: int, num_constraints: int, prompt_vector: []):
        self.positive_prompt_str = positive_prompt_str
        self.negative_prompt_str = negative_prompt_str
        self.num_topics = num_topics
        self.num_modifiers = num_modifiers
        self.num_styles = num_styles
        self.num_constraints = num_constraints

        # prompt_vector is a vector of -1, 0, or 1
        # 1 - used phrase for positive prompt
        # 0 - unused phrase
        # -1 - used for negative prompt
        self.prompt_vector = prompt_vector

    def get_positive_prompt_str(self):
        return self.positive_prompt_str

    def get_negative_prompt_str(self):
        return self.negative_prompt_str

    def to_json(self):
        return {'positive-prompt-str': self.positive_prompt_str,
                'negative-prompt-str': self.negative_prompt_str,
                'prompt-vector': self.prompt_vector,
                'num-topics': self.num_topics,
                'num-modifiers': self.num_modifiers,
                'num-styles': self.num_styles,
                'num-constraints': self.num_constraints,
                }


class PromptData:
    def __init__(self, index: int, phrase: str):
        self.Index = index

        # type is a list since
        # a phrase can have multiple
        # types. Like "chibi" can be
        # a topic and also a style.
        #
        # types can be:
        # topic - ex. "waifu"
        # modifier - ex. "beautiful"
        # style - ex. "water color"
        # constraint - ex. "white background"
        self.Types = []
        self.Phrase = phrase


class PromptList():
    def __init__(self):
        self.Prompts = []

    def is_phrase_exist(self, phrase: str):
        for prompt in self.Prompts:
            if prompt.Phrase == phrase:
                return True

        return False

    def add_phrase(self, phrase: str):
        if not self.is_phrase_exist(phrase):
            index = len(self.Prompts)
            new_prompt = PromptData(index, phrase)
            self.Prompts.append(new_prompt)
        else:
            print("Phrase: {} already exists".format(phrase))

    def add_phrases(self, phrases: []):
        for phrase in phrases:
            self.add_phrase(phrase)

    def add_type_to_phrase(self, phrase: str, prompt_type: str):
        # check if phrase exist
        prompt_data = [prompt for prompt in self.Prompts if (prompt.Phrase == phrase)]

        # if exist add type to phrase
        if len(prompt_data) != 0:
            prompt_data = prompt_data[0]
            # check first if type is already in list
            is_prompt_type_exists = len(
                [prompt_type for prompt_type in prompt_data.Types if prompt_type == prompt_type]) > 0
            if not is_prompt_type_exists:
                prompt_data.Types.append(prompt_type)
            else:
                raise Exception("Trying to add existing type:{0} to phrase:{1}".format(prompt_type, phrase))
        # if not, make phrase and add type
        else:
            self.add_phrase(phrase)
            self.add_type_to_phrase(phrase, prompt_type)

    def add_types_to_phrase(self, phrase: str, types: []):
        for prompt_type in types:
            self.add_type_to_phrase(phrase, prompt_type)

    def add_topic_phrases(self, phrases: []):
        for phrase in phrases:
            self.add_phrase(phrase)
            self.add_type_to_phrase(phrase, prompt_type="topic")

    def add_style_phrases(self, phrases: []):
        for phrase in phrases:
            self.add_phrase(phrase)
            self.add_type_to_phrase(phrase, prompt_type="style")

    def add_modifier_phrases(self, phrases: []):
        for phrase in phrases:
            self.add_phrase(phrase)
            self.add_type_to_phrase(phrase, prompt_type="modifier")

    def add_constraint_phrases(self, phrases: []):
        for phrase in phrases:
            self.add_phrase(phrase)
            self.add_type_to_phrase(phrase, prompt_type="constraint")


def initialize_prompt_list_from_csv(csv_dataset_path, csv_phrase_limit=0):
    prompt_list = PromptList()
    phrase_token_size_list = []
    positive_count_list = []
    negative_count_list = []
    with open(csv_dataset_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                # index,total count,positive count,negative count,token size,phrase str
                phrase = row[5]

                index = len(prompt_list.Prompts)
                new_prompt = PromptData(index, phrase)
                new_prompt.Types.append("topic")
                prompt_list.Prompts.append(new_prompt)

                # add token count
                phrase_token_size = int(row[4])
                phrase_token_size_list.append(phrase_token_size)

                # add positive count
                positive_count = int(row[2])
                positive_count_list.append(positive_count)

                # add negative count
                negative_count = int(row[3])
                negative_count_list.append(negative_count)

                line_count += 1

            if csv_phrase_limit != 0 and line_count > csv_phrase_limit:
                break

    return prompt_list.Prompts, phrase_token_size_list, positive_count_list, negative_count_list


def get_sorted_list_with_cumulative(phrases, phrases_token_size, count_list):
    # sort by count
    sorted_phrases = []
    sorted_token_size = []
    sorted_count = []
    sorted_cumulative_sum = []
    sorted_indexes = sorted(range(len(count_list)), key=lambda x: count_list[x], reverse=True)

    prev_sum = 0
    for i in sorted_indexes:
        sorted_phrases.append(phrases[i])
        sorted_token_size.append(phrases_token_size[i])
        sorted_count.append(count_list[i])

        # add cumulative sum
        cumulative_sum = prev_sum + count_list[i]
        sorted_cumulative_sum.append(cumulative_sum)
        prev_sum = cumulative_sum

    return sorted_phrases, sorted_token_size, sorted_count, sorted_cumulative_sum


def count_number_of_digits(num):
    count = 0
    while (num > 0):
        count = count + 1
        num = num // 10
    return count


# find the first element, whose cumulative total is more than the random number
def find_first_element_binary_search(cumulative_total_arr, random_num):
    low = 0
    high = len(cumulative_total_arr) - 1
    mid = 0

    while low <= high:
        mid = (high + low) / 2
        mid = math.floor(mid)

        # If random_num is greater, ignore left half
        if cumulative_total_arr[mid] < random_num:
            low = mid + 1
        # If random_num is smaller, ignore right half
        elif cumulative_total_arr[mid] > random_num:
            high = mid - 1
        # means random_num is present at mid
        else:
            return mid

        # use this index since sometimes the exact
        # random num is not in the list
        if low == high:
            return low

    # If we reach here, then the element was not present
    return -1


def generate_prompts_from_csv_proportional_selection(csv_dataset_path,
                                                     prompt_count,
                                                     csv_phrase_limit=0,
                                                     positive_prefix=""):
    generated_prompts = []
    max_token_size = 75
    comma_token_size = 1

    phrases, \
        phrases_token_size,\
        positive_count_list,\
        negative_count_list = initialize_prompt_list_from_csv(csv_dataset_path, csv_phrase_limit)

    total_len_phrases = len(phrases)

    positive_phrases, \
        positive_token_size, \
        positive_count, \
        positive_cumulative_sum = get_sorted_list_with_cumulative(phrases, phrases_token_size, positive_count_list)

    positive_total_cumulative = positive_cumulative_sum[-1]

    negative_phrases, \
        negative_token_size, \
        negative_count, \
        negative_cumulative_sum = get_sorted_list_with_cumulative(phrases, phrases_token_size, negative_count_list)

    negative_total_cumulative = negative_cumulative_sum[-1]

    # del unused var at this point
    del phrases
    del phrases_token_size
    del positive_count_list
    del negative_count_list

    positive_prefix_token_size = 0
    if positive_prefix != "":
        # get token size for prefix
        enc = tiktoken.get_encoding("cl100k_base")
        positive_prefix_prompt_tokens = enc.encode(positive_prefix)
        positive_prefix_token_size = len(positive_prefix_prompt_tokens)

    print("Generating {} prompts...".format(prompt_count))
    for i in tqdm(range(0, prompt_count)):
        positive_prompt_total_token_size = positive_prefix_token_size
        negative_prompt_total_token_size = 0
        positive_prompt = []
        negative_prompt = []
        prompt_vector = [0] * total_len_phrases

        # positive prompt
        while positive_prompt_total_token_size < max_token_size:
            random_int = random.randint(0, positive_total_cumulative)
            random_index = find_first_element_binary_search(positive_cumulative_sum, random_int)
            if prompt_vector[random_index] != 0:
                continue

            prompt_index = random_index
            random_prompt = positive_phrases[prompt_index]

            chosen_phrase_size = positive_token_size[prompt_index]
            sum_token_size = positive_prompt_total_token_size + chosen_phrase_size + comma_token_size
            if sum_token_size < max_token_size:
                # update used array
                prompt_vector[prompt_index] = 1
                positive_prompt.append(random_prompt)
                positive_prompt_total_token_size = sum_token_size
            else:
                break

        # negative prompt
        while negative_prompt_total_token_size < max_token_size:
            random_int = random.randint(0, negative_total_cumulative)
            random_index = find_first_element_binary_search(negative_cumulative_sum, random_int)

            if prompt_vector[random_index] != 0:
                continue

            prompt_index = random_index
            random_prompt = negative_phrases[prompt_index]

            chosen_phrase_size = negative_token_size[prompt_index]
            sum_token_size = negative_prompt_total_token_size + chosen_phrase_size + comma_token_size
            if sum_token_size < max_token_size:
                # update used array
                prompt_vector[prompt_index] = -1
                negative_prompt.append(random_prompt)
                negative_prompt_total_token_size = sum_token_size
            else:
                break

        positive_prompt_str = ', '.join([prompt.Phrase for prompt in positive_prompt])
        if positive_prefix != "":
            positive_prompt_str = "{}, {}".format(positive_prefix, positive_prompt_str)
        negative_prompt_str = ', '.join([prompt.Phrase for prompt in negative_prompt])

        num_topics = len([prompt.Phrase for prompt in positive_prompt if "topic" in prompt.Types])
        num_modifiers = len([prompt.Phrase for prompt in positive_prompt if "modifier" in prompt.Types])
        num_styles = len([prompt.Phrase for prompt in positive_prompt if "style" in prompt.Types])
        num_constraints = len([prompt.Phrase for prompt in positive_prompt if "constraint" in prompt.Types])

        prompt = GeneratedPrompt(positive_prompt_str, negative_prompt_str, num_topics, num_modifiers,
                            num_styles, num_constraints, prompt_vector)

        # save prompt json
        generated_prompts.append(prompt)

    return generated_prompts


def generate_image_generation_jobs_using_generated_prompts(csv_dataset_path,
                                                           prompt_count,
                                                           dataset_name,
                                                           csv_phrase_limit=0,
                                                           positive_prefix=""):
    prompts = generate_prompts_from_csv_proportional_selection(csv_dataset_path,
                                                               prompt_count,
                                                               csv_phrase_limit,
                                                               positive_prefix)

    # get sequential ids
    sequential_ids = request.http_get_sequential_id(dataset_name, prompt_count)

    count = 0
    # generate jobs
    for prompt in prompts:
        # generate UUID
        task_uuid = str(uuid.uuid4())
        task_type = "image_generation_task"
        model_name = "v1-5-pruned-emaonly"
        model_file_name = "v1-5-pruned-emaonly"
        model_file_path = "input/model/sd/v1-5-pruned-emaonly/v1-5-pruned-emaonly.safetensors"
        task_input_dict = {
            "positive_prompt": prompt.positive_prompt_str,
            "negative_prompt": prompt.negative_prompt_str,
            "cfg_strength": 12,
            "seed": "",
            "dataset": dataset_name,
            "file_path": sequential_ids[count]+".jpg",
            "num_images": 1,
            "image_width": 512,
            "image_height": 512,
            "sampler": "ddim",
            "sampler_steps": 20
        }

        generation_task = GenerationTask(uuid=task_uuid,
                                         task_type=task_type,
                                         model_name=model_name,
                                         model_file_name=model_file_name,
                                         model_file_path=model_file_path,
                                         task_input_dict=task_input_dict)
        generation_task_json = generation_task.to_dict()

        # add job
        request.http_add_job(generation_task_json)

        count += 1


def generate_inpainting_generation_jobs_using_generated_prompts(csv_dataset_path,
                                                                prompt_count,
                                                                dataset_name,
                                                                csv_phrase_limit=0,
                                                                positive_prefix="",
                                                                init_img_path="./test/test_inpainting/white_512x512.jpg",
                                                                mask_path="./test/test_inpainting/icon_mask.png"):
    prompts = generate_prompts_from_csv_proportional_selection(csv_dataset_path,
                                                               prompt_count,
                                                               csv_phrase_limit,
                                                               positive_prefix)

    # get sequential ids
    sequential_ids = request.http_get_sequential_id(dataset_name, prompt_count)

    count = 0
    # generate jobs
    for prompt in prompts:
        # generate UUID
        task_uuid = str(uuid.uuid4())
        task_type = "inpainting_generation_task"
        model_name = "v1-5-pruned-emaonly"
        model_file_name = "v1-5-pruned-emaonly"
        model_file_path = "input/model/sd/v1-5-pruned-emaonly/v1-5-pruned-emaonly.safetensors"
        task_input_dict = {
            "positive_prompt": prompt.positive_prompt_str,
            "negative_prompt": prompt.negative_prompt_str,
            "cfg_strength": 12,
            "seed": "",
            "dataset": dataset_name,
            "file_path": sequential_ids[count]+".jpg",
            "image_width": 512,
            "image_height": 512,
            "sampler": "ddim",
            "sampler_steps": 20,
            "init_img": init_img_path,
            "init_mask": mask_path,
            "mask_blur": 0,
            "inpainting_fill_mode": 1,
            "styles": [],
            "resize_mode": 0,
            "denoising_strength": 0.75,
            "image_cfg_scale": 1.5,
            "inpaint_full_res_padding": 32,
            "inpainting_mask_invert": 0
        }

        generation_task = GenerationTask(uuid=task_uuid,
                                         task_type=task_type,
                                         model_name=model_name,
                                         model_file_name=model_file_name,
                                         model_file_path=model_file_path,
                                         task_input_dict=task_input_dict)
        generation_task_json = generation_task.to_dict()

        # add job
        request.http_add_job(generation_task_json)

        count += 1



