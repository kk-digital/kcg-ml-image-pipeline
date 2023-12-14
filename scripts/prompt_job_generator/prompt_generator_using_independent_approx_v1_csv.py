import os
import sys
import argparse
base_directory = "./"
sys.path.insert(0, base_directory)

from utility.http import request
from utility.minio import cmd
from utility.boltzman.boltzman_phrase_scores_loader import BoltzmanPhraseScoresLoader
from utility.boltzman.boltzman import get_cumulative_probability_arr, generate_prompts


def run_prompt_generator(minio_client,
                         dataset_name,
                         positive_phrase_scores_csv,
                         negative_phrase_scores_csv,
                         prompt_count,
                         boltzman_temperature,
                         boltzman_k,
                         ):
    positive_phrase_scores_loader = BoltzmanPhraseScoresLoader(dataset_name=dataset_name,
                                                       phrase_scores_csv=positive_phrase_scores_csv,
                                                       minio_client=minio_client,
                                                       )
    positive_phrase_scores_loader.load_dataset()
    positive_phrase_origin_indexes, positive_cumulative_probability_arr = get_cumulative_probability_arr(minio_client=minio_client,
                                                                                                         dataset_name=dataset_name,
                                                                                                         index_phrase_score_data=positive_phrase_scores_loader.index_phrase_score_data,
                                                                                                         boltzman_temperature=boltzman_temperature,
                                                                                                         boltzman_k=boltzman_k,
                                                                                                         type="positive")

    negative_phrase_scores_loader = BoltzmanPhraseScoresLoader(dataset_name=dataset_name,
                                                       phrase_scores_csv=negative_phrase_scores_csv,
                                                       minio_client=minio_client,
                                                       )

    negative_phrase_scores_loader.load_dataset()
    negative_phrase_origin_indexes, negative_cumulative_probability_arr = get_cumulative_probability_arr(minio_client=minio_client,
                                                                                                         dataset_name=dataset_name,
                                                                                                         index_phrase_score_data=negative_phrase_scores_loader.index_phrase_score_data,
                                                                                                         boltzman_temperature=boltzman_temperature,
                                                                                                         boltzman_k=boltzman_k+10,
                                                                                                         type="negative")

    generate_prompts(minio_client,
                     dataset_name,
                     positive_phrase_scores_loader,
                     positive_phrase_origin_indexes,
                     positive_cumulative_probability_arr,
                     negative_phrase_scores_loader,
                     negative_phrase_origin_indexes,
                     negative_cumulative_probability_arr,
                     prompt_count,
                     boltzman_temperature,
                     boltzman_k)


def parse_args():
    parser = argparse.ArgumentParser(description="Prompt Job Generator using Independent Approx V1 csv results")
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--dataset-name', required=True, help='Name of the dataset to generate prompt jobs')
    parser.add_argument('--positive-phrase-scores-csv', required=True, help='Filename of the positive phrase scores csv')
    parser.add_argument('--negative-phrase-scores-csv', required=True, help='Filename of the negative phrase scores csv')
    parser.add_argument('--prompt-count', required=True, type=int, help='Number of prompt jobs to generate')
    parser.add_argument('--boltzman-k', default=1.0, type=float, help='K for boltzman probability')
    parser.add_argument('--boltzman-temperature', default=8, type=float, help='Temperature for boltzman probability')
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
                             args.positive_phrase_scores_csv,
                             args.negative_phrase_scores_csv,
                             args.prompt_count,
                             args.boltzman_temperature,
                             args.boltzman_k)
    else:
        # if all, do for all existing datasets
        # get dataset name list
        dataset_names = request.http_get_dataset_names()
        print("dataset names=", dataset_names)
        for dataset in dataset_names:
            try:
                run_prompt_generator(minio_client,
                                     args.dataset_name,
                                     args.positive_phrase_scores_csv,
                                     args.negative_phrase_scores_csv,
                                     args.prompt_count,
                                     args.boltzman_temperature,
                                     args.boltzman_k)
            except Exception as e:
                print("Error running prompt generator for {}: {}".format(dataset, e))


if __name__ == "__main__":
    main()
