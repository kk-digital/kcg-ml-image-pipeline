import os
import sys
import argparse
import io
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

base_directory = "./"
sys.path.insert(0, base_directory)

from scripts.image_scorer import ImageScorer
from training_worker.http import request
from utility.minio import cmd
from data_loader.phrase_scores_loader import PhraseScoresLoader


def get_cumulative_probability_arr(index_phrase_score_data):
    scores_arr = [0] * len(index_phrase_score_data)
    for index, data in index_phrase_score_data.items():
        score = data.score
        scores_arr[index] = score

    scores_np_arr = np.array(scores_arr)

    k = 1.0
    temperature = 0.8
    probability_arr = np.exp(-(scores_np_arr/(k*temperature)))
    max_val = probability_arr.max
    normalized_probability_arr = probability_arr / max_val
    cumulative_probability_arr = normalized_probability_arr.cumsum()
    print("scores=", scores_np_arr)
    print("prob=", probability_arr)
    print("normalized=", normalized_probability_arr)
    print("cumulative=", cumulative_probability_arr)

    return cumulative_probability_arr


def run_prompt_generator(minio_client,
                         dataset_name,
                         positive_phrase_scores_csv,
                         negative_phrase_scores_csv):
    positive_phrase_scores_loader = PhraseScoresLoader(dataset_name=dataset_name,
                                                       phrase_scores_csv=positive_phrase_scores_csv,
                                                       minio_client=minio_client,
                                                       )

    positive_cumulative_probability_arr = get_cumulative_probability_arr(positive_phrase_scores_loader.index_phrase_score_data)

    # negative_phrase_scores_loader = PhraseScoresLoader(dataset_name=dataset_name,
    #                                                    phrase_scores_csv=negative_phrase_scores_csv,
    #                                                    minio_client=minio_client,
    #                                                    )
    #
    # negative_probability_arr = get_probability_arr(negative_phrase_scores_loader.index_phrase_score_data)



def parse_args():
    parser = argparse.ArgumentParser(description="Prompt Job Generator using Independent Approx V1 csv results")
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--dataset-name', required=True, help='Name of the dataset to generate prompt jobs')
    parser.add_argument('--positive-phrase-scores-csv', required=True, help='Filename of the positive phrase scores csv')
    parser.add_argument('--negative-phrase-scores-csv', required=True, help='Filename of the negative phrase scores csv')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    dataset_name = args.dataset_name
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)
    if dataset_name != "all":
        run_prompt_generator(minio_client,
                             args.dataset_name,
                             args.clip_model_filename,
                             args.embedding_model_filename)
    else:
        # if all, do for all existing datasets
        # get dataset name list
        dataset_names = request.http_get_dataset_names()
        print("dataset names=", dataset_names)
        for dataset in dataset_names:
            try:
                run_prompt_generator(minio_client,
                                     dataset,
                                     args.clip_model_filename,
                                     args.embedding_model_filename)
            except Exception as e:
                print("Error running prompt generator for {}: {}".format(dataset, e))


if __name__ == "__main__":
    main()
