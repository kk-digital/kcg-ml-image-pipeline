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

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.minio import cmd
from data_loader.utils import get_object, get_phrases_from_prompt, get_datasets
from data_loader.generated_image_data import GeneratedImageData

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

        self.positive_phrases_index_dict = {}
        self.index_positive_phrases_dict = {}
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

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for path in data_paths:
                futures.append(executor.submit(self.get_phrases,
                                               path=path))

            for future in tqdm(as_completed(futures), total=len(futures)):
                positive_phrases, negative_phrases = future.result()

                for phrase in positive_phrases:
                    if phrase not in self.positive_phrases_index_dict:
                        self.positive_phrases_index_dict[phrase] = positive_index
                        self.index_positive_phrases_dict[positive_index] = phrase
                        positive_index += 1

                for phrase in negative_phrases:
                    if phrase not in self.negative_phrases_index_dict:
                        self.negative_phrases_index_dict[phrase] = negative_index
                        self.index_negative_phrases_dict[negative_index] = phrase
                        negative_index += 1

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
        csv_data = csv_data.decode().replace("'", '"')
        lines = csv_data.splitlines()
        csv_reader = csv.reader(lines, delimiter=',')

        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
            else:
                positive_index = int(row[0])
                phrase = row[1]
                self.positive_phrases_index_dict[phrase] = positive_index
                self.index_positive_phrases_dict[positive_index] = phrase

            line_count += 1


        print("Dataset loaded...")
        print("Time elapsed: {0}s".format(format(time.time() - start_time, ".2f")))

    def upload_csv(self):
        print("Saving phrases csv...")
        # positive
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow((["index", "phrase"]))

        for phrase, index in self.positive_phrases_index_dict.items():
            writer.writerow([index, phrase])

        bytes_buffer = io.BytesIO(bytes(csv_buffer.getvalue(), "utf-8"))
        # upload the csv
        date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%Y-%m-%d')
        filename = "{}-positive-phrases.csv".format(date_now)
        csv_path = os.path.join(self.dataset_name, "output/phrases-csv", filename)
        cmd.upload_data(self.minio_client, 'datasets', csv_path, bytes_buffer)

        # negative
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow((["index", "phrase"]))

        for phrase, index in self.negative_phrases_index_dict.items():
            writer.writerow([index, phrase])

        bytes_buffer = io.BytesIO(bytes(csv_buffer.getvalue(), "utf-8"))
        # upload the csv
        date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%Y-%m-%d')
        filename = "{}-negative-phrases.csv".format(date_now)
        csv_path = os.path.join(self.dataset_name, "output/phrases-csv", filename)
        cmd.upload_data(self.minio_client, 'datasets', csv_path, bytes_buffer)

    def get_positive_phrase_vector(self, prompt):
        phrase_vector = [False] * len(self.positive_phrases_index_dict)
        phrases = get_phrases_from_prompt(prompt)
        for phrase in phrases:
            index = self.positive_phrases_index_dict[phrase]
            phrase_vector[index] = True

        return phrase_vector


