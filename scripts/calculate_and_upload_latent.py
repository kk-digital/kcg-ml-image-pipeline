import io
import sys
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from tqdm.auto import tqdm
from diffusers import AutoPipelineForText2Image
import msgpack
from minio import Minio
from utility.minio import cmd

# Settings
BATCH_SIZE = 8  # Adjust this based on your GPU memory and model requirements
base_directory = "./"
sys.path.insert(0, base_directory)

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
stable_diffusion.vae.eval().to(device)

minio_ip_addr = '192.168.3.5:9000'
access_key = 'v048BpXpWrsVIHUfdAix'
secret_key = '4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu'
BUCKET_NAME = 'datasets'

def list_datasets(minio_client, bucket_name):
    objects = minio_client.list_objects(bucket_name, recursive=False)
    for obj in objects:
        if obj.is_dir:
            yield obj.object_name.rstrip('/')

def check_if_file_exists(minio_client, bucket_name, file_path):
    try:
        return minio_client.stat_object(bucket_name, file_path) is not None
    except:
        return False


def process_batch(batch_images, batch_names, minio_client, to_tensor_transform, stable_diffusion, BUCKET_NAME):
    # Convert list of image tensors into a single tensor for batch processing
    batch_tensor = torch.stack(batch_images).to(device)
    batch_tensor = (batch_tensor - 0.5) * 2.0  # Normalize batch

    with torch.no_grad():
        # Get latent representations for the whole batch
        batch_latents = stable_diffusion.vae.encode(batch_tensor).latent_dist.mean.detach().cpu().numpy()

    # Iterate over the batch
    for i, latent in enumerate(batch_latents):
        latent_file_path = batch_names[i].replace('.jpg', '_latent.msgpack')
        data_msgpack_name = batch_names[i].replace('.jpg', '_data.msgpack')
        
        # Retrieve the corresponding data for each image
        data_msgpack_data = minio_client.get_object(BUCKET_NAME, data_msgpack_name)
        data = msgpack.unpackb(data_msgpack_data.read(), raw=False)

        new_data = {
            'job_uuid': data['job_uuid'],
            'file_hash': data['file_hash'],
            'latent': latent.tolist()
        }

        # Pack the new data with msgpack
        buffer = io.BytesIO()
        msgpack.pack(new_data, buffer)
        buffer.seek(0)

        # Upload _latent.msgpack to MinIO
        minio_client.put_object(BUCKET_NAME, latent_file_path, buffer, length=buffer.getbuffer().nbytes)

    # Clear the lists for the next batch
    batch_images.clear()
    batch_names.clear()


def worker(dataset_name, minio_client, to_tensor_transform):
    objects = minio_client.list_objects(BUCKET_NAME, prefix=f'{dataset_name}/', recursive=True)
    batch_images = []
    batch_names = []

    for obj in tqdm(objects, leave=False):
        if obj.object_name.endswith('.jpg'):
            latent_file_path = obj.object_name.replace('.jpg', '_latent.msgpack')

            if check_if_file_exists(minio_client, BUCKET_NAME, latent_file_path):
                print(f"Skipping {latent_file_path} as it already exists.")
                continue

            image_data = minio_client.get_object(BUCKET_NAME, obj.object_name)
            image = Image.open(image_data).convert('RGB')
            image_tensor = to_tensor_transform(image).half().to(device)

            batch_images.append(image_tensor)
            batch_names.append(latent_file_path)

            if len(batch_images) == BATCH_SIZE:
                process_batch(batch_images, batch_names, minio_client, to_tensor_transform)
                batch_images = []
                batch_names = []

    if batch_images:  # Process the remaining images if they don't make up a full batch
        process_batch(batch_images, batch_names, minio_client, to_tensor_transform)

if __name__ == "__main__":
    minio_client = cmd.connect_to_minio_client(minio_ip_addr, access_key, secret_key)
    to_tensor_transform = ToTensor()
    
    for dataset in list_datasets(minio_client, BUCKET_NAME):
        print(f"Processing dataset: {dataset}")
        worker(dataset, minio_client, to_tensor_transform)
