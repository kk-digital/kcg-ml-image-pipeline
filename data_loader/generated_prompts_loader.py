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


class GeneratedPromptData:
    job_uuid = None
    positive_prompt = None
    negative_prompt = None
    boltzman_temperature = None
    boltzman_k = None

    # attributes
    image_clip_score = None
    image_clip_sigma_score = None
    image_clip_percentile = None
    text_embedding_score = None
    text_embedding_sigma_score = None
    text_embedding_percentile = None
    delta_sigma_score = None

    def __init__(self,
                 job_uuid: str,
                 positive_prompt: str,
                 negative_prompt: str,
                 boltzman_temperature: float,
                 boltzman_k: float):
        self.job_uuid = job_uuid
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt
        self.boltzman_temperature = boltzman_temperature
        self.boltzman_k = boltzman_k

    def update_attributes(self,
                          image_clip_score,
                          image_clip_sigma_score,
                          image_clip_percentile,
                          text_embedding_score,
                          text_embedding_sigma_score,
                          text_embedding_percentile,
                          delta_sigma_score):
        self.image_clip_score = image_clip_score
        self.image_clip_sigma_score = image_clip_sigma_score
        self.image_clip_percentile = image_clip_percentile
        self.text_embedding_score = text_embedding_score
        self.text_embedding_sigma_score = text_embedding_sigma_score
        self.text_embedding_percentile = text_embedding_percentile
        self.delta_sigma_score = delta_sigma_score


class GeneratedPromptsLoader:
    def __init__(self,
                 dataset_name,
                 generated_prompts_csv,
                 minio_client):
        self.dataset_name = dataset_name
        self.generated_prompts_csv = generated_prompts_csv
        self.minio_client = minio_client

        self.generated_prompt_data_arr = []

    def load_dataset(self):
        start_time = time.time()
        print("-------------------------------------------------------------------------------------")
        print("Loading data from csv...")

        full_path = os.path.join(self.dataset_name, "output/generated-prompts-csv", self.generated_prompts_csv)

        # check if exists
        if not cmd.is_object_exists(self.minio_client, "datasets", full_path):
            raise Exception("{} doesnt exist in minio server...".format(full_path))

        csv_data = get_object(self.minio_client, full_path)
        csv_data = csv_data.decode(encoding='utf-8')
        lines = csv_data.splitlines()
        csv_reader = csv.reader(lines, delimiter=',')

        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
            else:
                # job_uuid	positive_prompt	negative_prompt	boltzman temperature	boltzman k
                job_uuid = row[0]
                positive_prompt = row[1]
                negative_prompt = row[2]
                boltzman_temperature = float(row[3])
                boltzman_k = float(row[4])


                prompt_data = GeneratedPromptData(job_uuid,
                                                  positive_prompt,
                                                  negative_prompt,
                                                  boltzman_temperature,
                                                  boltzman_k)
                self.generated_prompt_data_arr.append(prompt_data)

            line_count += 1

        print("Generated Prompts Data Len=", len(self.generated_prompt_data_arr))
        print("Dataset loaded...")
        print("Time elapsed: {0}s".format(format(time.time() - start_time, ".2f")))

    def save_updated_csv(self):
        print("Saving updated generated prompts csv...")
        csv_header = ["job_uuid", "positive_prompt", "negative_prompt", "boltzman temperature", "boltzman k", "text_embedding_score", "text_embedding_sigma_score", "text_embedding_percentile", "image_clip_score", "image_clip_sigma_score", "image_clip_percentile", "delta_sigma_score"]
        # positive
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(csv_header)

        for prompt_data in self.generated_prompt_data_arr:
            job_uuid = prompt_data.job_uuid
            positive_prompt = prompt_data.positive_prompt
            negative_prompt = prompt_data.negative_prompt
            boltzman_temperature = prompt_data.boltzman_temperature
            boltzman_k = prompt_data.boltzman_k
            text_embedding_score = prompt_data.text_embedding_score
            text_embedding_sigma_score = prompt_data.text_embedding_sigma_score
            text_embedding_percentile = prompt_data.text_embedding_percentile
            image_clip_score = prompt_data.image_clip_score
            image_clip_sigma_score = prompt_data.image_clip_sigma_score
            image_clip_percentile = prompt_data.image_clip_percentile
            delta_sigma_score = prompt_data.delta_sigma_score

            writer.writerow([job_uuid, positive_prompt, negative_prompt, boltzman_temperature, boltzman_k, text_embedding_score, text_embedding_sigma_score, text_embedding_percentile, image_clip_score, image_clip_sigma_score, image_clip_percentile, delta_sigma_score])

        bytes_buffer = io.BytesIO(bytes(csv_buffer.getvalue(), "utf-8"))

        # overwrite the csv
        csv_path = os.path.join(self.dataset_name, "output/generated-prompts-csv", self.generated_prompts_csv)
        cmd.upload_data(self.minio_client, 'datasets', csv_path, bytes_buffer)


