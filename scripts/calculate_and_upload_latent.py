#!/usr/bin/env python3
import os
import sys
import json
import torch
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from diffusers import AutoPipelineForText2Image
import msgpack
from minio import Minio


# Check for GPU availability
if not torch.cuda.is_available():
    print("GPU is not available. Exiting.")
    sys.exit(1)

device = 'cuda'

# Stable Diffusion model setup
model_path = '/input/models/runwayml-stable-diffusion-inpainting/'
stable_diffusion = AutoPipelineForText2Image.from_pretrained(
    model_path, local_files_only=True, torch_dtype=torch.float16, variant="fp16"
)
stable_diffusion.vae.eval().cuda()

MINIO_ADDRESS = '192.168.3.5:9000'
ACCESS_KEY = 'v048BpXpWrsVIHUfdAix'
SECRET_KEY = '4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu'
BUCKET_NAME = 'datasets'
LOCAL_SAVE_DIR = './data'
BATCH_SIZE = 4

def connect_to_minio_client():
    print("Connecting to MinIO client...")
    client = Minio(MINIO_ADDRESS, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False)
    print("Successfully connected to MinIO client...")
    return client

def worker(dataset_name, minio_client):
    to_tensor_transform = ToTensor()
    os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)

    objects = minio_client.list_objects(BUCKET_NAME, prefix=f'{dataset_name}/', recursive=True)

    for obj in tqdm(objects, leave=False):
        if obj.object_name.endswith('.jpg'):
            image_data = minio_client.get_object(BUCKET_NAME, obj.object_name)
            image = Image.open(image_data).convert('RGB')
            image_tensor = to_tensor_transform(image).half().cuda()
            image_tensor = (image_tensor - 0.5) * 2.0

            with torch.no_grad():
                latent = stable_diffusion.vae.encode(image_tensor.unsqueeze(0)).latent_dist.mean.detach().cpu().numpy()

            data_msgpack_name = obj.object_name.replace('.jpg', '_data.msgpack')
            data_msgpack_data = minio_client.get_object(BUCKET_NAME, data_msgpack_name)
            data = msgpack.unpackb(data_msgpack_data.read(), raw=False)

            new_data = {
                'job_uuid': data['job_uuid'],
                'file_hash': data['file_hash'],
                'latent': latent.tolist()
            }

            local_dir_path = os.path.join(LOCAL_SAVE_DIR, os.path.dirname(obj.object_name))
            os.makedirs(local_dir_path, exist_ok=True)

            latent_msgpack_path = os.path.join(local_dir_path, os.path.basename(obj.object_name).replace('.jpg', '_latent.msgpack'))
            with open(latent_msgpack_path, 'wb') as f:
                msgpack.pack(new_data, f)

if __name__ == "__main__":
    minio_client = connect_to_minio_client()
    worker('test-generations', minio_client)
