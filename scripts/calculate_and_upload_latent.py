import io
import os
import sys
base_directory = "./"
sys.path.insert(0, base_directory)
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from tqdm.auto import tqdm
from diffusers import AutoPipelineForText2Image
import msgpack
from minio import Minio
from utility.minio import cmd


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

minio_ip_addr = '192.168.3.5:9000'
access_key = 'v048BpXpWrsVIHUfdAix'
secret_key = '4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu'
BUCKET_NAME = 'datasets'

def list_datasets(minio_client, bucket_name):
    """
    List all directories in the bucket, assuming each directory is a dataset.
    """
    objects = minio_client.list_objects(bucket_name, recursive=False)
    for obj in objects:
        if obj.is_dir:
            yield obj.object_name.rstrip('/')

def check_if_file_exists(minio_client, bucket_name, file_path):
    try:
        return minio_client.stat_object(bucket_name, file_path) is not None
    except:
        return False

def worker(dataset_name, minio_client):
    to_tensor_transform = ToTensor()

    objects = minio_client.list_objects(BUCKET_NAME, prefix=f'{dataset_name}/', recursive=True)

    for obj in tqdm(objects, leave=False):
        if obj.object_name.endswith('.jpg'):
            latent_file_path = obj.object_name.replace('.jpg', '_latent.msgpack')

            # Check if _latent.msgpack already exists
            if check_if_file_exists(minio_client, BUCKET_NAME, latent_file_path):
                print(f"Skipping {latent_file_path} as it already exists.")
                continue

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

            buffer = io.BytesIO()
            msgpack.pack(new_data, buffer)
            buffer.seek(0)

            # Upload _latent.msgpack to MinIO
            cmd.upload_data(minio_client, BUCKET_NAME, latent_file_path, buffer)

if __name__ == "__main__":
    minio_client = cmd.connect_to_minio_client(minio_ip_addr, access_key, secret_key)
    for dataset in list_datasets(minio_client, BUCKET_NAME):
        print(f"Processing dataset: {dataset}")
        worker(dataset, minio_client)
