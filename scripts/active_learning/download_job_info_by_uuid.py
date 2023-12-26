import sys
import os

import requests
import json
import argparse
from tqdm.auto import tqdm
import pandas as pd


class JobInfoDownloader:

    def __init__(self, api_addr: str):

        self.url = f'http://{api_addr}/job/get-job'

    def get_info(self, job_uuid: str):
        
        response = requests.get(f'{self.url}/{job_uuid}')
        
        info = json.loads(response.content)

        return info
        

def parse_args():
    parser = argparse.ArgumentParser(description="download job info by job_uuid")

    # Required parameters
    parser.add_argument("--csv-path", type=str,
                        help="The path to csv file")
    parser.add_argument("--api-addr", type=str, default=None,
                        help="The api server ip address")

    return parser.parse_args()

def main():
    
    args = parse_args()

    downloader = JobInfoDownloader(
        api_addr=args.api_addr
    )

    # read

    df = pd.read_csv(args.csv_path)

    assert 'job_uuid' in df.columns

    if 'file_path' not in df.columns:
        df['file_path'] = None
    if 'file_hash' not in df.columns:
        df['file_hash'] = None
        
    if 'image_clip_percentile' not in df.columns:
        df['image_clip_percentile'] = None
    if 'image_clip_score' not in df.columns:
        df['image_clip_score'] = None
    if 'image_clip_sigma_score' not in df.columns:
        df['image_clip_sigma_score'] = None
        
    if 'text_embedding_percentile' not in df.columns:
        df['text_embedding_percentile'] = None
    if 'text_embedding_score' not in df.columns:
        df['text_embedding_score'] = None
    if 'text_embedding_sigma_score' not in df.columns:
        df['text_embedding_sigma_score'] = None

    # load

    for i, job_uuid in enumerate(tqdm(df['job_uuid'])):

        try:
            
            info = downloader.get_info(job_uuid)
    
            if 'task_output_file_dict' in info:
                output_info = info['task_output_file_dict']
                if 'output_file_path' in output_info:
                    df.loc[i, 'file_path'] = output_info['output_file_path']
                if 'output_file_hash' in output_info:
                    df.loc[i, 'file_hash'] = output_info['output_file_hash']
    
            if 'task_attributes_dict' in info:
                attribute_info = info['task_attributes_dict']
                if 'image_clip_percentile' in attribute_info:
                    df.loc[i, 'image_clip_percentile'] = attribute_info['image_clip_percentile']
                if 'image_clip_score' in attribute_info:
                    df.loc[i, 'image_clip_score'] = attribute_info['image_clip_score']
                if 'image_clip_sigma_score' in attribute_info:
                    df.loc[i, 'image_clip_sigma_score'] = attribute_info['image_clip_sigma_score']
                if 'text_embedding_percentile' in attribute_info:
                    df.loc[i, 'text_embedding_percentile'] = attribute_info['text_embedding_percentile']
                if 'text_embedding_score' in attribute_info:
                    df.loc[i, 'text_embedding_score'] = attribute_info['text_embedding_score']
                if 'text_embedding_sigma_score' in attribute_info:
                    df.loc[i, 'text_embedding_sigma_score'] = attribute_info['text_embedding_sigma_score']

        except:
            continue

    # save

    df.to_csv(args.csv_path, index=False)
    

if __name__ == '__main__':
    main()
