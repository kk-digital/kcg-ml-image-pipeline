import sys
import os

import json
import argparse
import pandas as pd
        

def parse_args():
    parser = argparse.ArgumentParser(description="merge pairs")

    # Required parameters
    parser.add_argument('json_paths', nargs='+', help='List of json file paths')
    parser.add_argument("--output-path", type=str,
                        help="The path to save the merged pairs json file")

    return parser.parse_args()

def main():
    
    args = parse_args()

    # read

    pairs = list()
    existed = set()
    
    for json_path in args.json_paths:

        for pair in json.load(open(json_path, 'rt')):

            image1_job_uuid = pair['image1_job_uuid']
            image2_job_uuid = pair['image2_job_uuid']

            if (image1_job_uuid, image2_job_uuid) in existed:
                continue
            if (image2_job_uuid, image1_job_uuid) in existed:
                continue

            pairs.append(pair)
            existed.add((image1_job_uuid, image2_job_uuid))

    # save
    
    json.dump(pairs, open(args.output_path, 'wt'))
    

if __name__ == '__main__':
    main()
