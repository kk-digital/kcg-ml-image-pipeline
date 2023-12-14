import os
import sys
import argparse
base_directory = "./"
sys.path.insert(0, base_directory)

from utility.http import request
from utility.minio import cmd
from utility.boltzman.boltzman_phrase_scores_loader import BoltzmanPhraseScoresLoader
from utility.gibs_sampling.gibs_sampling import generate_prompts


def run_prompt_generator(minio_client,
                         dataset_name,
                         positive_phrase_scores_csv,
                         negative_phrase_scores_csv,
                         prompt_count,
                         gibs_temperature,
                         gibs_k,
                         ):
    positive_phrase_scores_loader = BoltzmanPhraseScoresLoader(dataset_name=dataset_name,
                                                       phrase_scores_csv=positive_phrase_scores_csv,
                                                       minio_client=minio_client,
                                                       )
    positive_phrase_scores_loader.load_dataset()


    negative_phrase_scores_loader = BoltzmanPhraseScoresLoader(dataset_name=dataset_name,
                                                       phrase_scores_csv=negative_phrase_scores_csv,
                                                       minio_client=minio_client,
                                                       )

    negative_phrase_scores_loader.load_dataset()


    generate_prompts(minio_client,
                     dataset_name,
                     positive_phrase_scores_loader,
                     negative_phrase_scores_loader,
                     prompt_count,
                     gibs_temperature,
                     gibs_k)


def parse_args():
    parser = argparse.ArgumentParser(description="Prompt Job Generator using Independent Approx V1 csv results")
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--dataset-name', required=True, help='Name of the dataset to generate prompt jobs')
    parser.add_argument('--positive-phrase-scores-csv', required=True, help='Filename of the positive phrase scores csv')
    parser.add_argument('--negative-phrase-scores-csv', required=True, help='Filename of the negative phrase scores csv')
    parser.add_argument('--prompt-count', required=True, type=int, help='Number of prompt jobs to generate')
    parser.add_argument('--gibs-k', default=1.0, type=float, help='K for gibs probability')
    parser.add_argument('--gibs-temperature', default=8, type=float, help='Temperature for gibs probability')
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
                             args.gibs_temperature,
                             args.gibs_k)
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
                                     args.gibs_temperature,
                                     args.gibs_k)
            except Exception as e:
                print("Error running prompt generator for {}: {}".format(dataset, e))


if __name__ == "__main__":
    main()
