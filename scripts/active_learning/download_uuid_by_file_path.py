import sys
import os
root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_directory)

from utility.minio.cmd import connect_to_minio_client

import msgpack
import json
import argparse
from tqdm.auto import tqdm
import pandas as pd


class JobUuidDownloader:

    def __init__(self, minio_addr: str, minio_access_key: str, minio_secret_key: str):

        self.client = connect_to_minio_client(
            minio_addr, 
            minio_access_key, 
            minio_secret_key
        )

        self.bucket_name = 'datasets'

    def get_info(self, file_path: str):

        object_name = file_path.replace(f'{self.bucket_name}/', '')
        object_name = os.path.splitext(object_name.split('_')[0])[0]
        object_name = f'{object_name}_data.msgpack'
        
        data = self.client.get_object(self.bucket_name, object_name).data
        info = msgpack.unpackb(data)

        return info
        

def parse_args():
    parser = argparse.ArgumentParser(description="download job uuid info by file_path")

    # Required parameters
    parser.add_argument("--csv-path", type=str,
                        help="The path to csv file")
    parser.add_argument("--minio-addr", type=str, default=None,
                        help="The minio server ip address")
    parser.add_argument("--minio-access-key", type=str,
                        help="The minio access key to use so worker can upload files to minio server")
    parser.add_argument("--minio-secret-key", type=str,
                        help="The minio secret key to use so worker can upload files to minio server")

    return parser.parse_args()

def main():
    
    args = parse_args()

    downloader = JobUuidDownloader(
        minio_addr=args.minio_addr,
        minio_access_key=args.minio_access_key,
        minio_secret_key=args.minio_secret_key
    )

    # read

    df = pd.read_csv(args.csv_path)

    assert 'file_path' in df.columns

    if 'job_uuid' not in df.columns:
        df['job_uuid'] = None
    if 'file_hash' not in df.columns:
        df['file_hash'] = None
        
    # load

    for i, file_path in enumerate(tqdm(df['file_path'])):

        try:
            
            info = downloader.get_info(file_path)

            if 'job_uuid' in info:
                df.loc[i, 'job_uuid'] = info['job_uuid']

            if 'file_hash' in info:
                df.loc[i, 'file_hash'] = info['file_hash']

        except:
            continue
                
    # save

    df.to_csv(args.csv_path, index=False)
    

if __name__ == '__main__':
    main()
