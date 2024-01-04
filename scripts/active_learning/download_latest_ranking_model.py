import sys
import os
root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_directory)

from utility.minio.cmd import connect_to_minio_client

import requests
import json
import argparse


class LatsetRankingModelDownloader:

    def __init__(self, api_addr: str, minio_addr: str, minio_access_key: str, minio_secret_key: str):

        self.url = f'http://{api_addr}/models/rank-embedding/latest-model'
        
        self.client = connect_to_minio_client(
            minio_addr, 
            minio_access_key, 
            minio_secret_key
        )

        self.bucket_name = 'datasets'

    def get_model(self, dataset: str, input_type: str, output_type: str):
        
        response = requests.get(self.url, params={
            'dataset': dataset,
            'input_type': input_type,
            'output_type': output_type
        })
        
        info = json.loads(response.content)

        return info

    def download_model(self, model_path: str, output: str):

        file_name = os.path.split(model_path)[-1]

        self.client.fget_object(
            bucket_name=self.bucket_name, 
            object_name=model_path,
            file_path=os.path.join(output, file_name)
        )
        

def parse_args():
    parser = argparse.ArgumentParser(description="download latest ranking model")

    # Required parameters
    parser.add_argument("--dataset", type=str,
                        help="The dataset of ranking model")
    parser.add_argument("--input-type", type=str,
                        help="The input_type of ranking model")
    parser.add_argument("--output", type=str,
                        help="The folder to save the safetensor")
    parser.add_argument("--api-addr", type=str, default=None,
                        help="The api server ip address")
    parser.add_argument("--minio-addr", type=str, default=None,
                        help="The minio server ip address")
    parser.add_argument("--minio-access-key", type=str,
                        help="The minio access key to use so worker can upload files to minio server")
    parser.add_argument("--minio-secret-key", type=str,
                        help="The minio secret key to use so worker can upload files to minio server")

    return parser.parse_args()

def main():
    
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    downloader = LatsetRankingModelDownloader(
        api_addr=args.api_addr,
        minio_addr=args.minio_addr,
        minio_access_key=args.minio_access_key,
        minio_secret_key=args.minio_secret_key
    )

    info = downloader.get_model(
        dataset=args.dataset,
        input_type=args.input_type,
        output_type='score'
    )

    print(info)

    downloader.download_model(
        model_path=info['model_path'],
        output=args.output,
    )
    

if __name__ == '__main__':
    main()