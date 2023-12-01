import sys
import threading
import io
import random

base_directory = "./"
sys.path.insert(0, base_directory)

from scripts.prompt_job_generator.prompt_generator_using_independent_approx_v1_csv import (PhraseScoresLoader,
                                                                                           get_cumulative_probability_arr)

# Helper class that generates prompts using
# Indenpendent approx v1
class IndependentApproxV1:
    def __init__(self, dataset_name, boltzman_temperature, boltzman_k):
        self.dataset_name = dataset_name
        self.boltzman_temperature = boltzman_temperature
        self.boltzman_k = boltzman_k

        self.minio_client = None
        self.positive_phrase_scores_loader = None
        self.positive_phrase_origin_indexes = None
        self.positive_cumulative_probability_arr = None
        self.negative_phrase_scores_loader = None
        self.negative_phrase_origin_indexes = None
        self.negative_cumulative_probability_arr = None
        self.positive_phrase_scores_csv = None
        self.negative_phrase_scores_csv = None

    def load_csv(self, minio_client, positive_phrase_scores_csv, negative_phrase_scores_csv):
        dataset_name = self.dataset_name
        boltzman_temperature = self.boltzman_temperature
        boltzman_k = self.boltzman_k

        self.minio_client = minio_client
        self.positive_phrase_scores_csv = positive_phrase_scores_csv
        self.negative_phrase_scores_csv = negative_phrase_scores_csv

        positive_phrase_scores_loader = PhraseScoresLoader(dataset_name=dataset_name,
                                                           phrase_scores_csv=positive_phrase_scores_csv,
                                                           minio_client=minio_client,
                                                           )
        positive_phrase_scores_loader.load_dataset()
        positive_phrase_origin_indexes, positive_cumulative_probability_arr = get_cumulative_probability_arr(
            minio_client=minio_client,
            dataset_name=dataset_name,
            index_phrase_score_data=positive_phrase_scores_loader.index_phrase_score_data,
            boltzman_temperature=boltzman_temperature,
            boltzman_k=boltzman_k,
            type="positive")

        negative_phrase_scores_loader = PhraseScoresLoader(dataset_name=dataset_name,
                                                           phrase_scores_csv=negative_phrase_scores_csv,
                                                           minio_client=minio_client,
                                                           )

        negative_phrase_scores_loader.load_dataset()
        negative_phrase_origin_indexes, negative_cumulative_probability_arr = get_cumulative_probability_arr(
            minio_client=minio_client,
            dataset_name=dataset_name,
            index_phrase_score_data=negative_phrase_scores_loader.index_phrase_score_data,
            boltzman_temperature=boltzman_temperature,
            boltzman_k=boltzman_k,
            type="negative")

        self.positive_phrase_scores_loader = positive_phrase_scores_loader
        self.positive_phrase_origin_indexes = positive_phrase_origin_indexes
        self.positive_cumulative_probability_arr = positive_cumulative_probability_arr
        self.negative_phrase_scores_loader = negative_phrase_scores_loader
        self.negative_phrase_origin_indexes = negative_phrase_origin_indexes
        self.negative_cumulative_probability_arr = negative_cumulative_probability_arr