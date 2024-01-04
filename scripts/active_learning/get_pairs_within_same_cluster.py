import sys
import os
root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_directory)

from tqdm.auto import tqdm

import pandas as pd

import json
import argparse

from utility.active_learning.pairs import get_candidate_pairs_within_category


def parse_args():
    parser = argparse.ArgumentParser(description="Script for getting candidate pairs within same cluster.")

    # Required parameters
    parser.add_argument("--csv-path", type=str,
                        help="The path to csv file")
    parser.add_argument("--output-path", type=str,
                        help="The path to save the pairs json file")
    parser.add_argument("--pairs", type=int, default=1000,
                        help="The number of pairs")
    parser.add_argument("--cluster-type", type=str, 
                        help="The column name of cluster_id")

    return parser.parse_args()

def main():
    
    args = parse_args()

    # read

    df = pd.read_csv(args.csv_path)

    assert 'job_uuid' in df.columns
    assert args.cluster_type in df.columns

    # pairing

    pairs = get_candidate_pairs_within_category(
        categories = df[args.cluster_type].values, 
        max_pairs = args.pairs
    )

    pairs = [{
      "image1_job_uuid": df.loc[i, 'job_uuid'],
      "image2_job_uuid": df.loc[j, 'job_uuid'],
      "metadata": {
          "cluster_id": float(df.loc[i, args.cluster_type]),
          "pairs": args.pairs,
          "cluster_type": args.cluster_type,
      },
      "generator_string": "within_same_cluster",  
    } for i, j in pairs]

    # save

    json.dump(pairs, open(args.output_path, 'wt'))
    

if __name__ == '__main__':
    main()
