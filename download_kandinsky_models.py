import argparse
import math
import os
import time

import requests
from tqdm import tqdm

from configs.model_config import ModelPathConfig
from utility.utils_logger import logger
from utility.labml.monit import section
from utility.minio.cmd import is_minio_server_accessible, download_folder_from_minio, connect_to_minio_client

config = ModelPathConfig()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minio-ip-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')

    return parser.parse_args()

def create_directory_tree_folders(config):
    config.create_paths()


def download_file(url, file_path, description, update_interval=500, chunk_size=4096):
    if not os.path.isfile(file_path):
        def memory2str(mem):
            sizes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
            power = int(math.log(mem, 1024))
            size = sizes[power]
            for _ in range(power):
                mem /= 1024
            if power > 0:
                return f'{mem:.2f}{size}'
            else:
                return f'{mem}{size}'

        with open(file_path, 'wb') as f:
            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')
            if total_length is None:
                f.write(response.content)
            else:
                print(f'Downloading {file_path}.', flush=True)
                downloaded, total_length = 0, int(total_length)
                total_size = memory2str(total_length)
                bar_format = '{percentage:3.0f}%|{bar:20}| {desc} [{elapsed}<{remaining}' \
                             '{postfix}]'
                if update_interval * chunk_size * 100 >= total_length:
                    update_interval = 1
                with tqdm(total=total_length, bar_format=bar_format) as bar:
                    counter = 0
                    now_time, now_size = time.time(), downloaded
                    for data in response.iter_content(chunk_size=chunk_size):
                        f.write(data)
                        downloaded += len(data)
                        counter += 1
                        bar.update(len(data))
                        if counter % update_interval == 0:
                            elapsed = time.time() - now_time
                            runtime_downloaded = downloaded - now_size
                            now_time, now_size = time.time(), downloaded

                            cur_size = memory2str(downloaded)
                            speed_size = memory2str(runtime_downloaded / elapsed)
                            bar.set_description(f'{cur_size}/{total_size}')
                            bar.set_postfix_str(f'{speed_size}/s')

                            counter = 0
    else:
        logger.debug(f"{description} already exists.")


if __name__ == "__main__":
    args = parse_args()

    with section("Creating directory tree folders."):
        create_directory_tree_folders(config)

    logger.info("Downloading models. This may take a while.")

    minio_ip_addr = '192.168.3.5:9000'
    access_key = 'v048BpXpWrsVIHUfdAix'
    secret_key = '4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu'

    minio_client = connect_to_minio_client(minio_ip_addr=minio_ip_addr,
                                               access_key=access_key,
                                               secret_key=secret_key)

    # download kandinsky prior model
    prior_path = config.get_model('kandinsky/kandinsky-2-2-prior', check_existence=False)

    bucket_name = "models"
    folder_name = "kandinsky/kandinsky-2-2-cache/kandinsky-2-2-prior"
    download_folder_from_minio(minio_client, bucket_name, folder_name, prior_path)

    with section("Downloading kandinsky decoder models"):
        # download Dreamshaper inpainting model
        decoder_path = config.get_model('kandinsky/kandinsky-2-2-decoder', check_existence=False)

        bucket_name = "models"
        folder_name = "kandinsky/kandinsky-2-2-cache/kandinsky-2-2-decoder"
        download_folder_from_minio(minio_client, bucket_name, folder_name, decoder_path)
    
    with section("Downloading kandinsky inpainting decoder models"):
        # download kandinsky inpainting decoder models
        inpainting_decoder_path = config.get_model('kandinsky/kandinsky-2-2-decoder-inpaint', check_existence=False)

        bucket_name = "models"
        folder_name = "kandinsky/kandinsky-2-2-cache/kandinsky-2-2-decoder-inpaint"
        download_folder_from_minio(minio_client, bucket_name, folder_name, inpainting_decoder_path)
