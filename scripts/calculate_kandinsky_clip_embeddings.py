import argparse
from io import BytesIO
import json
import os
import sys
import numpy as np
import requests
import msgpack
from PIL import Image

base_directory = "./"
sys.path.insert(0, base_directory)
from utility.path import separate_bucket_and_file_path
from kandinsky.models.clip_image_encoder.clip_image_encoder import KandinskyCLIPImageEncoder
from utility.clip import clip
from utility.minio import cmd

API_URL = "http://192.168.3.1:8111"

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--minio-access-key", type=str,
                        help="The minio access key to use so worker can upload files to minio server")
    parser.add_argument("--minio-secret-key", type=str,
                        help="The minio secret key to use so worker can upload files to minio server")

    return parser.parse_args()

def get_job_list():
    response = requests.get(f'{API_URL}/image/list-image-metadata-by-dataset')
        
    jobs = json.loads(response.content)

    return jobs

def calculate_image_feature_vector(clip, minio_client, bucket_name, file_path):
    # get image from minio server
    try:
        response = minio_client.get_object(bucket_name, file_path)
        image_data = BytesIO(response.data)
        img = Image.open(image_data)
        img = img.convert("RGB")
    except Exception as e:
        raise e
    finally:
        response.close()
        response.release_conn()

    # get feature
    clip_feature_vector = clip.get_image_features(img)

    # put to cpu
    clip_feature_vector = clip_feature_vector.cpu().detach()

    # convert to np array
    clip_feature_vector_np_arr = np.array(clip_feature_vector, dtype=np.float32)

    # convert to normal list
    clip_feature_vector_arr = clip_feature_vector_np_arr.tolist()

    return clip_feature_vector_arr

def main():
    args = parse_args()

    minio_client= cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                       minio_secret_key=args.minio_secret_key)

    sd_clip_model = clip.ClipModel(device="cuda")
    kandinsky_clip_model = KandinskyCLIPImageEncoder(device="cuda")

    jobs_list= get_job_list()

    for job in jobs_list:
        image_path= job['image_path']
        bucket_name, input_file_path = separate_bucket_and_file_path(image_path)
        file_path = os.path.splitext(input_file_path)[0]
        
        if "kandinsky" in job['task_type']:
            output_path = file_path + "_clip.msgpack"
            clip_feature_vector= calculate_image_feature_vector(clip= sd_clip_model,
                                                                minio_client=minio_client,
                                                                bucket_name=bucket_name,
                                                                file_path= input_file_path)

            clip_feature_dict = {"clip-feature-vector": clip_feature_vector}
            clip_feature_msgpack = msgpack.packb(clip_feature_dict)

            data = BytesIO()
            data.write(clip_feature_msgpack)
            data.seek(0)

            cmd.upload_data(minio_client, bucket_name, output_path, data)

        output_path = file_path + "_clip_kandinsky.msgpack"
        clip_feature_vector= calculate_image_feature_vector(clip= kandinsky_clip_model,
                                                            minio_client=minio_client,
                                                            bucket_name=bucket_name,
                                                            file_path= input_file_path)

        clip_feature_dict = {"clip-feature-vector": clip_feature_vector}
        clip_feature_msgpack = msgpack.packb(clip_feature_dict)

        data = BytesIO()
        data.write(clip_feature_msgpack)
        data.seek(0)

        cmd.upload_data(minio_client, bucket_name, output_path, data)


if __name__ == '__main__':
    main()
