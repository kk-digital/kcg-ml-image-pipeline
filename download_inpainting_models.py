import math
import os
import time

import requests
from tqdm import tqdm

from configs.model_config import ModelPathConfig
from utility.utils_logger import logger
from utility.labml.monit import section
from utility.minio.cmd import is_minio_server_accessible, connect_to_minio_client, download_from_minio

config = ModelPathConfig()


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
    with section("Creating directory tree folders."):
        create_directory_tree_folders(config)

    logger.info("Downloading models. This may take a while.")

    # check if minio server is available
    is_minio_accessible = is_minio_server_accessible()
    if is_minio_accessible:
        minio_client = connect_to_minio_client()
    else:
        print("Need to be connected to minio to proceed.")

    with section("Downloading Never-ending-dream model"):
        # download Never Ending dream inpainting model
        ned_path = config.get_model('inpainting/ned-v1-22', check_existence=False)

        bucket_name = "models"
        object_name = "models/never-ending-dream/ned-v1-22-inpainting-A62B4AC3A2.safetensors"
        download_from_minio(minio_client, bucket_name, object_name, ned_path)
    
    with section("Downloading Dreamshaper model"):
        # download Dreamshaper inpainting model
        dreamshaper_path = config.get_model('dreamshaper-8', check_existence=False)

        bucket_name = "models"
        object_name = "models/dreamshaper/dreamshaper-8-inpainting-AC84E05D50.safetensors"
        download_from_minio(minio_client, bucket_name, object_name, dreamshaper_path)