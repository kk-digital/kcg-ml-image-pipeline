import os
import sys
import json
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pytz import timezone
import io
import csv
import tiktoken

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.minio import cmd
from data_loader.utils import get_object, get_phrases_from_prompt, get_datasets
from data_loader.generated_image_data import GeneratedImageData

class PromptPhraseInformation:
    def __init__(self,
                 phrase: str,
                 occurrences: int,
                 token_length: int):
        self.phrase = phrase
        self.occurrences = occurrences
        self.token_length = token_length

class PhraseVectorLoader:
    def __init__(self,
                 dataset_name,
                 minio_ip_addr=None,
                 minio_access_key=None,
                 minio_secret_key=None,):
        self.dataset_name = dataset_name

        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.minio_client = cmd.get_minio_client(minio_access_key=self.minio_access_key,
                                                 minio_secret_key=self.minio_secret_key,
                                                 minio_ip_addr=minio_ip_addr)

        self.index_positive_prompt_phrase_info = {}
        self.positive_phrases_index_dict = {}
        self.index_positive_phrases_dict = {}

        self.index_negative_prompt_phrase_info = {}
        self.negative_phrases_index_dict = {}
        self.index_negative_phrases_dict = {}

    def get_data_paths(self):
        print("Getting paths for dataset: {}...".format(self.dataset_name))
        all_objects = cmd.get_list_of_objects_with_prefix(self.minio_client, 'datasets', self.dataset_name)

        # Filter the objects to get only those that end with the chosen suffix
        file_suffix = "_data.msgpack"
        type_paths = [obj for obj in all_objects if obj.endswith(file_suffix)]

        print("Total paths found=", len(type_paths))
        return type_paths

    def get_phrases(self, path):
        # get object from minio
        data = get_object(self.minio_client, path)
        generated_image_data = GeneratedImageData.from_msgpack_string(data)

        positive_prompt = generated_image_data.positive_prompt
        negative_prompt = generated_image_data.negative_prompt

        positive_prompt_phrases = get_phrases_from_prompt(positive_prompt)
        negative_prompt_phrases = get_phrases_from_prompt(negative_prompt)

        return positive_prompt_phrases, negative_prompt_phrases

    def load_dataset_phrases(self):
        start_time = time.time()
        print("Loading dataset references...")

        dataset_list = get_datasets(self.minio_client)
        if self.dataset_name not in dataset_list:
            raise Exception("Dataset is not in minio server")

        data_paths = self.get_data_paths()
        positive_index = 0
        negative_index = 0

        # initialize tokenizer
        tokenizer = tiktoken.get_encoding("cl100k_base")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for path in data_paths:
                futures.append(executor.submit(self.get_phrases,
                                               path=path))

            for future in tqdm(as_completed(futures), total=len(futures)):
                positive_phrases, negative_phrases = future.result()

                for phrase in positive_phrases:
                    if len(phrase) > 80:
                        phrase = "more than 80 characters"

                    if phrase not in self.positive_phrases_index_dict:
                        index_to_use = positive_index
                        if phrase == "more than 80 characters":
                            index_to_use = -1

                        self.positive_phrases_index_dict[phrase] = index_to_use
                        self.index_positive_phrases_dict[index_to_use] = phrase

                        # token length
                        tokens = tokenizer.encode(phrase)
                        token_length = len(tokens)
                        prompt_phrase_info = PromptPhraseInformation(phrase,
                                                                     occurrences=1,
                                                                     token_length=token_length)
                        self.index_positive_prompt_phrase_info[index_to_use] = prompt_phrase_info

                        if index_to_use != -1:
                            positive_index += 1
                    else:
                        phrase_index = self.positive_phrases_index_dict[phrase]
                        prompt_phrase_info = self.index_positive_prompt_phrase_info[phrase_index]
                        count = prompt_phrase_info.occurrences
                        count += 1
                        prompt_phrase_info.occurrences = count
                        self.index_positive_prompt_phrase_info[phrase_index] = prompt_phrase_info

                for phrase in negative_phrases:
                    if len(phrase) > 80:
                        phrase = "more than 80 characters"

                    if phrase not in self.negative_phrases_index_dict:
                        index_to_use = negative_index
                        if phrase == "more than 80 characters":
                            index_to_use = -1

                        self.negative_phrases_index_dict[phrase] = index_to_use
                        self.index_negative_phrases_dict[index_to_use] = phrase

                        # token length
                        tokens = tokenizer.encode(phrase)
                        token_length = len(tokens)

                        prompt_phrase_info = PromptPhraseInformation(phrase,
                                                                     occurrences=1,
                                                                     token_length=token_length)
                        self.index_negative_prompt_phrase_info[index_to_use] = prompt_phrase_info
                        if index_to_use != -1:
                            negative_index += 1
                    else:
                        phrase_index = self.negative_phrases_index_dict[phrase]
                        prompt_phrase_info = self.index_negative_prompt_phrase_info[phrase_index]
                        count = prompt_phrase_info.occurrences
                        count += 1
                        prompt_phrase_info.occurrences = count
                        self.index_negative_prompt_phrase_info[phrase_index] = prompt_phrase_info

        print("Dataset loaded...")
        print("Time elapsed: {0}s".format(format(time.time() - start_time, ".2f")))

    def load_dataset_phrases_from_csv(self, csv_filename):
        start_time = time.time()
        print("Loading dataset references from csv...")

        full_path = os.path.join(self.dataset_name, "output/phrases-csv", csv_filename)

        # check if exists
        if not cmd.is_object_exists(self.minio_client, "datasets", full_path):
            raise Exception("{} doesnt exist in minio server...".format(full_path))

        csv_data = get_object(self.minio_client,full_path)
        csv_data = csv_data.decode(encoding='utf-8')
        lines = csv_data.splitlines()
        csv_reader = csv.reader(lines, delimiter=',')

        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
            else:
                positive_index = int(row[0])
                phrase = row[1]
                num_occurrences = int(row[2])
                token_length = int(row[3])
                self.positive_phrases_index_dict[phrase] = positive_index
                self.index_positive_phrases_dict[positive_index] = phrase
                prompt_info = PromptPhraseInformation(phrase,
                                                      num_occurrences,
                                                      token_length)
                self.index_positive_prompt_phrase_info[positive_index] = prompt_info

            line_count += 1


        print("Dataset loaded...")
        print("Time elapsed: {0}s".format(format(time.time() - start_time, ".2f")))

    def upload_csv(self):
        print("Saving phrases csv...")
        csv_header = ["index", "phrase", "occurrences", "token length"]
        # positive
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(csv_header)

        for phrase, index in self.positive_phrases_index_dict.items():
            prompt_phrase_info = self.index_positive_prompt_phrase_info[index]
            num_occurrences = prompt_phrase_info.occurrences
            token_length = prompt_phrase_info.token_length
            writer.writerow([index, phrase, num_occurrences, token_length])

        bytes_buffer = io.BytesIO(bytes(csv_buffer.getvalue(), "utf-8"))
        # upload the csv
        date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%Y-%m-%d')
        filename = "{}-positive-phrases.csv".format(date_now)
        csv_path = os.path.join(self.dataset_name, "output/phrases-csv", filename)
        cmd.upload_data(self.minio_client, 'datasets', csv_path, bytes_buffer)

        # negative
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(csv_header)

        for phrase, index in self.negative_phrases_index_dict.items():
            prompt_phrase_info = self.index_negative_prompt_phrase_info[index]
            num_occurrences = prompt_phrase_info.occurrences
            token_length = prompt_phrase_info.token_length
            writer.writerow([index, phrase, num_occurrences, token_length])

        bytes_buffer = io.BytesIO(bytes(csv_buffer.getvalue(), "utf-8"))
        # upload the csv
        date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%Y-%m-%d')
        filename = "{}-negative-phrases.csv".format(date_now)
        csv_path = os.path.join(self.dataset_name, "output/phrases-csv", filename)
        cmd.upload_data(self.minio_client, 'datasets', csv_path, bytes_buffer)

    def get_phrase_vector(self, prompt, input_type="positive"):
        if input_type == "positive":
            len_vector = len(self.positive_phrases_index_dict)
            if self.index_positive_phrases_dict.get(-1) is not None:
                len_vector -= 1
        else:
            len_vector = len(self.negative_phrases_index_dict)
            if self.index_negative_phrases_dict.get(-1) is not None:
                len_vector -= 1

        phrase_vector = [False] * len_vector
        phrases = get_phrases_from_prompt(prompt)
        for phrase in phrases:
            if len(phrase) > 80:
                phrase = "more than 80 characters"

            if input_type == "positive":
                index = self.positive_phrases_index_dict[phrase]
            else:
                index = self.negative_phrases_index_dict[phrase]

            if index == -1:
                continue
            phrase_vector[index] = True

        return phrase_vector

    def get_token_length_vector(self, input_type="positive"):
        token_length_vector = []

        if input_type == "positive":
            for index, data in self.index_positive_prompt_phrase_info.items():
                if index == -1:
                    continue
                token_length = data.token_length
                token_length_vector.append(token_length)
        else:
            for index, data in self.index_negative_prompt_phrase_info.items():
                if index == -1:
                    continue
                token_length = data.token_length
                token_length_vector.append(token_length)

        return token_length_vector

    def get_positive_phrases_arr(self):
        positive_phrases_arr = []
        for _, phrase in self.index_positive_phrases_dict.items():
            positive_phrases_arr.append(phrase)

        return positive_phrases_arr

    def get_negative_phrases_arr(self):
        negative_phrases_arr = []
        for _, phrase in self.index_negative_phrases_dict.items():
            negative_phrases_arr.append(phrase)

        return negative_phrases_arr



