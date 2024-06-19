import io
import os
import sys
base_directory = "./"
sys.path.insert(0, base_directory)
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import PIL
from tqdm.auto import tqdm
from diffusers import AutoPipelineForText2Image
import msgpack
from minio import Minio
from utility.minio import cmd
from utility.utils_logger import logger
from kandinsky.model_paths import PRIOR_MODEL_PATH
from stable_diffusion.utils_backend import get_device
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import numpy as np

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

class KandinskyCLIPImageEncoder(nn.Module):

    def __init__(self, device=None, image_processor=None, vision_model=None):  # , input_mode = PIL.Image.Image):
        super().__init__()
        self.device = get_device(device)
        self.vision_model = vision_model
        self.image_processor = image_processor
        self.to(self.device)

    def load_submodels(self, encoder_path=PRIOR_MODEL_PATH):
        try:
            self.vision_model = (CLIPVisionModelWithProjection.from_pretrained(encoder_path,
                                                                               subfolder="image_encoder",
                                                                               torch_dtype=torch.float16,
                                                                               local_files_only=True).eval().to(self.device))
            
            logger.info(f"CLIP VisionModelWithProjection successfully loaded from : {encoder_path}/image_encoder \n")
            self.image_processor = CLIPImageProcessor.from_pretrained(encoder_path, subfolder="image_processor", local_files_only=True)
            logger.info(f"CLIP ImageProcessor successfully loaded from : {encoder_path}/image_processor \n")
            return self
        except Exception as e:
            logger.error('Error loading submodels: ', e)

    def unload_submodels(self):
        # Unload the model from GPU memory
        if self.vision_model is not None:
            self.vision_model.to('cpu')
            del self.vision_model
            torch.cuda.empty_cache()
            self.vision_model = None
        if self.image_processor is not None:
            del self.image_processor
            torch.cuda.empty_cache()
            self.image_processor = None

    def convert_image_to_tensor(self, image: PIL.Image.Image):
        return torch.from_numpy(np.array(image)) \
            .permute(2, 0, 1) \
            .unsqueeze(0) \
            .to(self.device) * (2 / 255.) - 1.0

    def forward(self, image):
        # Preprocess image
        # Compute CLIP features
        if isinstance(image, PIL.Image.Image):
            image = self.image_processor(image, return_tensors="pt")['pixel_values']
        
        if isinstance(image, torch.Tensor):
            with torch.no_grad():
                features = self.vision_model(pixel_values=image.to(self.device).half()).image_embeds
        else:
            raise ValueError(
                f"`image` can only contain elements to be of type `PIL.Image.Image` or `torch.Tensor`  but is {type(image)}"
            )
        
        return features
    
    def get_image_features(self, image):
        # Preprocess image
        if isinstance(image, PIL.Image.Image):
            image = self.image_processor(image, return_tensors="pt")['pixel_values']
        
        # Compute CLIP features
        if isinstance(image, torch.Tensor):
            with torch.no_grad():
                features = self.vision_model(pixel_values=image.to(self.device).half()).image_embeds
        else:
            raise ValueError(
                f"`image` can only contain elements to be of type `PIL.Image.Image` or `torch.Tensor`  but is {type(image)}"
            )
        
        return features.to(torch.float16)

    @staticmethod
    def compute_sha256(image_data):
        # Compute SHA256
        return hashlib.sha256(image_data).hexdigest()

    @staticmethod
    def convert_image_to_rgb(image):
        return image.convert("RGB")

    @staticmethod
    def get_input_type(image):
        if isinstance(image, PIL.Image.Image):
            return PIL.Image.Image
        elif isinstance(image, torch.Tensor):
            return torch.Tensor
        else:
            raise ValueError("Image must be PIL Image or Tensor")

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

def worker(dataset_name, minio_client, encoder):
    to_tensor_transform = ToTensor()

    objects = minio_client.list_objects(BUCKET_NAME, prefix=f'{dataset_name}/', recursive=True)

    for obj in tqdm(objects, leave=False):
        if obj.object_name.endswith('.jpg'):
            latent_file_path = obj.object_name.replace('.jpg', '_latent.msgpack')
            clip_file_path = obj.object_name.replace('.jpg', '_clip.msgpack')

            # Check if _latent.msgpack or _clip.msgpack already exists
            if check_if_file_exists(minio_client, BUCKET_NAME, latent_file_path):
                print(f"Skipping {latent_file_path} as it already exists.")
                continue

            image_data = minio_client.get_object(BUCKET_NAME, obj.object_name)
            image = Image.open(image_data).convert('RGB')
            image_tensor = to_tensor_transform(image).half().cuda()
            image_tensor = (image_tensor - 0.5) * 2.0

            with torch.no_grad():
                latent = stable_diffusion.vae.encode(image_tensor.unsqueeze(0)).latent_dist.mean.detach().cpu().numpy()
                clip_features = encoder.get_image_features(image).cpu().numpy()

            data_msgpack_name = obj.object_name.replace('.jpg', '_data.msgpack')
            data_msgpack_data = minio_client.get_object(BUCKET_NAME, data_msgpack_name)
            data = msgpack.unpackb(data_msgpack_data.read(), raw=False)

            new_data = {
                'job_uuid': data['job_uuid'],
                'file_hash': data['file_hash'],
                'latent_vector': latent.tolist()
            }

            new_clip_data = {
                'job_uuid': data['job_uuid'],
                'file_hash': data['file_hash'],
                'clip_features': clip_features.tolist()
            }

            buffer_latent = io.BytesIO()
            msgpack.pack(new_data, buffer_latent)
            buffer_latent.seek(0)

            buffer_clip = io.BytesIO()
            msgpack.pack(new_clip_data, buffer_clip)
            buffer_clip.seek(0)

            # Upload _latent.msgpack to MinIO
            cmd.upload_data(minio_client, BUCKET_NAME, latent_file_path, buffer_latent)

            # Upload _clip.msgpack to MinIO
            cmd.upload_data(minio_client, BUCKET_NAME, clip_file_path, buffer_clip)

if __name__ == "__main__":
    minio_client = cmd.connect_to_minio_client(minio_ip_addr, access_key, secret_key)
    encoder = KandinskyCLIPImageEncoder().load_submodels()
    for dataset in list_datasets(minio_client, BUCKET_NAME):
        print(f"Processing dataset: {dataset}")
        worker(dataset, minio_client, encoder)