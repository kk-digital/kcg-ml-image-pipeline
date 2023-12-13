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


class PhraseScoreData:
    def __init__(self,
                 phrase: str,
                 occurrences: int,
                 token_length: int,
                 prompt_phrase_average_weight: float,
                 energy_per_token: float,
                 sigma_energy_per_token: float,
                 energy_per_phrase: float):
        self.phrase = phrase
        self.occurrences = occurrences
        self.token_length = token_length
        self.prompt_phrase_average_weight = prompt_phrase_average_weight
        self.energy_per_token = energy_per_token
        self.sigma_energy_per_token = sigma_energy_per_token
        self.energy_per_phrase = energy_per_phrase
        

class BoltzmanPhraseScoresLoader:
    def __init__(self,
                 dataset_name,
                 phrase_scores_csv,
                 minio_client):
        self.dataset_name = dataset_name
        self.phrase_scores_csv = phrase_scores_csv
        self.minio_client = minio_client

        self.index_phrase_score_data = {}

    def load_dataset(self):
        start_time = time.time()
        print("-------------------------------------------------------------------------------------")
        print("Loading dataset references from csv...")

        full_path = os.path.join(self.dataset_name, "output/phrases-score-csv", self.phrase_scores_csv)

        # check if exists
        if not cmd.is_object_exists(self.minio_client, "datasets", full_path):
            raise Exception("{} doesnt exist in minio server...".format(full_path))

        csv_data = get_object(self.minio_client, full_path)
        csv_data = csv_data.decode(encoding='utf-8')
        lines = csv_data.splitlines()
        csv_reader = csv.reader(lines, delimiter=',')

        line_count = 0
        index_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
            else:
                # index
                index = int(row[0])

                # skip -1
                if index == -1:
                    continue

                # phrase
                phrase = row[1]

                # occurrences
                occurrences = int(row[2])

                # token length
                token_length = int(row[3])

                # prompt_phrase_average_weight
                prompt_phrase_average_weight = float(row[4])

                # energy_per_token
                energy_per_token = float(row[5])

                # sigma energy per token
                sigma_energy_per_token = float(row[6])

                # energy_per_phrase
                energy_per_phrase = float(row[7])

                phrase_data = PhraseScoreData(phrase=phrase,
                                              occurrences=occurrences,
                                              token_length=token_length,
                                              prompt_phrase_average_weight=prompt_phrase_average_weight,
                                              energy_per_token=energy_per_token,
                                              sigma_energy_per_token=sigma_energy_per_token,
                                              energy_per_phrase=energy_per_phrase,
                                              )
                self.index_phrase_score_data[index_count] = phrase_data
                index_count += 1

            line_count += 1

        print("index phrase score data len=", len(self.index_phrase_score_data))
        print("Dataset loaded...")
        print("Time elapsed: {0}s".format(format(time.time() - start_time, ".2f")))

    def get_phrase(self, index):
        return self.index_phrase_score_data[index].phrase

    def get_token_size(self, index):
        return self.index_phrase_score_data[index].token_length


