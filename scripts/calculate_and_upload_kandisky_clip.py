import argparse
from io import BytesIO
import os
import sys
import numpy as np
import msgpack
from PIL import Image

base_directory = "./"
sys.path.insert(0, base_directory)
from kandinsky.models.clip_image_encoder.clip_image_encoder import KandinskyCLIPImageEncoder
from utility.minio import cmd
from utility.path import separate_bucket_and_file_path

# Hardcoded MinIO credentials
MINIO_IP = "192.168.3.5:9000"
MINIO_ACCESS_KEY = 'v048BpXpWrsVIHUfdAix'
MINIO_SECRET_KEY = '4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu'

def calculate_image_feature_vector(clip_model, minio_client, bucket_name, file_path):
    # Get image from MinIO server
    try:
        response = minio_client.get_object(bucket_name, file_path)
        image_data = BytesIO(response.read())
        img = Image.open(image_data)
        img = img.convert("RGB")
    except Exception as e:
        raise e
    finally:
        response.close()
        response.release_conn()

    # Get feature vector
    clip_feature_vector = clip_model.get_image_features(img)

    # Convert to numpy array and then to list
    clip_feature_vector_arr = clip_feature_vector.cpu().detach().numpy().tolist()

    return clip_feature_vector_arr

def main():
    minio_client = cmd.connect_to_minio_client(MINIO_IP,MINIO_ACCESS_KEY, MINIO_SECRET_KEY)

    # Load Kandinsky CLIP model
    kandinsky_clip_model = KandinskyCLIPImageEncoder(device="cuda")
    kandinsky_clip_model.load_submodels()

    # Define the dataset bucket and path
    dataset_bucket = "external"
    dataset_prefix = ""

    # List objects in the dataset
    objects = minio_client.list_objects(bucket_name=dataset_bucket, prefix=dataset_prefix, recursive=True)

    for obj in objects:
        if obj.object_name.endswith('.jpg'):
            input_file_path = obj.object_name
            output_file_path = input_file_path.replace('.jpg', '_clip_kandinsky.msgpack')

            print(f"Processing {input_file_path}...")

            # Calculate Kandinsky CLIP feature vector
            clip_feature_vector = calculate_image_feature_vector(clip_model=kandinsky_clip_model,
                                                                 minio_client=minio_client,
                                                                 bucket_name=dataset_bucket,
                                                                 file_path=input_file_path)

            # Prepare the data for msgpack
            clip_feature_dict = {"clip-feature-vector": clip_feature_vector}
            clip_feature_msgpack = msgpack.packb(clip_feature_dict)

            # Upload to MinIO
            data = BytesIO()
            data.write(clip_feature_msgpack)
            data.seek(0)

            cmd.upload_data(minio_client, dataset_bucket, output_file_path, data)

            print(f"Uploaded {output_file_path}.")

if __name__ == '__main__':
    main()
