import sys
import threading
import io
import random

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.boltzman.boltzman import (get_cumulative_probability_arr_without_upload)

# Helper class that generates prompts using
# Indenpendent approx v1
class BoltzmanProbabilitiesCache:
    def __init__(self):
        self.probabilities_dictionary = {}

    def get_boltzman_probabilities(self, boltzman_temperature, boltzman_k):

        key = (boltzman_temperature, boltzman_k)
        if key not in self.probabilities_dictionary:
            positive_phrase_origin_indexes, positive_cumulative_probability_arr = get_cumulative_probability_arr_without_upload(
                index_phrase_score_data=self.positive_phrase_scores_loader.index_phrase_score_data,
                boltzman_temperature=boltzman_temperature,
                boltzman_k=boltzman_k)

            negative_phrase_origin_indexes, negative_cumulative_probability_arr = get_cumulative_probability_arr_without_upload(
                index_phrase_score_data=self.negative_phrase_scores_loader.index_phrase_score_data,
                boltzman_temperature=boltzman_temperature,
                boltzman_k=boltzman_k)

            boltzman_probabilities = BoltzmanProbabilities(positive_phrase_origin_indexes=positive_phrase_origin_indexes,
                                     positive_cumulative_probability_arr=positive_cumulative_probability_arr,
                                     negative_phrase_origin_indexes=negative_phrase_origin_indexes,
                                     negative_cumulative_probability_arr=negative_cumulative_probability_arr)

            self.probabilities_dictionary[key] = boltzman_probabilities

        boltzman_probabilities = self.probabilities_dictionary[key]

        return boltzman_probabilities

class BoltzmanProbabilities:
    def __init__(self, positive_phrase_origin_indexes,
                 positive_cumulative_probability_arr,
                 negative_phrase_origin_indexes,
                 negative_cumulative_probability_arr):

        self.positive_phrase_origin_indexes = positive_phrase_origin_indexes
        self.positive_cumulative_probability_arr = positive_cumulative_probability_arr
        self.negative_phrase_origin_indexes = negative_phrase_origin_indexes
        self.negative_cumulative_probability_arr = negative_cumulative_probability_arr
