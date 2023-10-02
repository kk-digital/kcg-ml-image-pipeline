
import queue
import sys

base_directory = "./"
sys.path.insert(0, base_directory)

from worker.minio.minio import get_minio_client
from stable_diffusion import StableDiffusion, CLIPTextEmbedder
from configs.model_config import ModelPathConfig
from stable_diffusion.model_paths import (SDconfigs, CLIPconfigs)
from worker.image_generation.scripts.stable_diffusion_base_script import StableDiffusionBaseScript
from utility.clip import clip
class WorkerState:
    def __init__(self, device, minio_access_key, minio_secret_key, queue_size):
        self.device = device
        self.config = ModelPathConfig()
        self.stable_diffusion = None
        self.clip_text_embedder = None
        self.txt2img = None
        self.minio_client = get_minio_client(minio_access_key, minio_secret_key)
        self.queue_size = queue_size
        self.job_queue = queue.Queue()
        self.clip = clip.ClipModel()

    def load_models(self, model_path='input/model/sd/v1-5-pruned-emaonly/v1-5-pruned-emaonly.safetensors'):
        # NOTE: Initializing stable diffusion
        self.stable_diffusion = StableDiffusion(device=self.device)

        self.stable_diffusion.quick_initialize().load_autoencoder(self.config.get_model(SDconfigs.VAE)).load_decoder(
            self.config.get_model(SDconfigs.VAE_DECODER))
        self.stable_diffusion.model.load_unet(self.config.get_model(SDconfigs.UNET))
        self.stable_diffusion.initialize_latent_diffusion(path=model_path, force_submodels_init=True)

        self.clip_text_embedder = CLIPTextEmbedder(device=self.device)

        self.clip_text_embedder.load_submodels(
            tokenizer_path=self.config.get_model_folder_path(CLIPconfigs.TXT_EMB_TOKENIZER),
            transformer_path=self.config.get_model_folder_path(CLIPconfigs.TXT_EMB_TEXT_MODEL)
        )

        # Starts the text2img
        self.txt2img = StableDiffusionBaseScript(
            sampler_name="ddim",
            n_steps=20,
            force_cpu=False,
            cuda_device=self.device,
        )
        self.txt2img.initialize_latent_diffusion(autoencoder=None, clip_text_embedder=None, unet_model=None,
                                                 path=model_path, force_submodels_init=True)

        # load clip model
        self.clip.load_clip()
