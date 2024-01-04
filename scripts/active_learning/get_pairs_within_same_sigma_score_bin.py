import sys
import os
root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_directory)

from tqdm.auto import tqdm

import pandas as pd

import json
import argparse

from utility.active_learning.pairs import get_candidate_pairs_by_score


def parse_args():
    parser = argparse.ArgumentParser(description="Script for getting candidate pairs within same sigma score bin. \
    We will attempt to select (pairs / bins) pairs within each category.")

    # Required parameters
    parser.add_argument("--csv-path", type=str,
                        help="The path to csv file")
    parser.add_argument("--output-path", type=str,
                        help="The path to save the pairs json file")
    parser.add_argument("--pairs", type=int, default=1000,
                        help="The number of pairs")
    parser.add_argument("--bins", type=int, default=10,
                        help="The number of bins")
    parser.add_argument("--bin-type", type=str, default='quantile',
                        help="The binning method: fixed-range or quantile")

    return parser.parse_args()

def main():
    
    args = parse_args()

    # read

    df = pd.read_csv(args.csv_path)

    assert 'job_uuid' in df.columns
    assert 'image_clip_sigma_score' in df.columns

    # pairing

    pairs = get_candidate_pairs_by_score(
        scores = df['image_clip_sigma_score'].values, 
        max_pairs = args.pairs, 
        n_bins = args.bins, 
        use_quantiles = (args.bin_type == 'quantile')
    )

    pairs = [{
      "image1_job_uuid": df.loc[i, 'job_uuid'],
      "image2_job_uuid": df.loc[j, 'job_uuid'],
      "metadata": {
          "image1_clip_sigma_score": df.loc[i, 'image_clip_sigma_score'],
          "image2_clip_sigma_score": df.loc[j, 'image_clip_sigma_score'],
          "pairs": args.pairs,
          "bins": args.bins,
      },
      "generator_string": "within_same_sigma_score_bin",  
    } for i, j in pairs]

    # save

    json.dump(pairs, open(args.output_path, 'wt'))
    

if __name__ == '__main__':
    main()
