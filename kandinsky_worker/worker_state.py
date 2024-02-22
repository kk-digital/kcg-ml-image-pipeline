import queue
import sys
import torch

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.minio.cmd import get_minio_client
from configs.model_config import ModelPathConfig
from kandinsky.model_paths import PRIOR_MODEL_PATH, INPAINT_DECODER_MODEL_PATH, DECODER_MODEL_PATH
from kandinsky.models.clip_image_encoder.clip_image_encoder import KandinskyCLIPImageEncoder
from kandinsky.pipelines.kandinsky_text2img import KandinskyV22Pipeline
from kandinsky.pipelines.kandinsky_img2img import KandinskyV22Img2ImgPipeline
from kandinsky.pipelines.kandinsky_prior import KandinskyV22PriorPipeline
from kandinsky.pipelines.kandinsky_inpainting import KandinskyV22InpaintPipeline
from diffusers.models import UNet2DConditionModel
from diffusers.models import UNet2DConditionModel
from kandinsky.models.clip_text_encoder.clip_text_encoder import KandinskyCLIPTextEmbedder

class WorkerState:
    def __init__(self, device, minio_access_key, minio_secret_key, queue_size):
        self.device = device
        self.config = ModelPathConfig()
        self.unet= None
        self.prior_model= None
        self.decoder_model= None
        self.inpainting_decoder_model= None
        self.clip_text_embedder= None
        self.minio_client = get_minio_client(minio_access_key, minio_secret_key)
        self.queue_size = queue_size
        self.job_queue = queue.Queue()

    def load_models(self, prior_path=PRIOR_MODEL_PATH, decoder_path= DECODER_MODEL_PATH, 
                    inpaint_decoder_path= INPAINT_DECODER_MODEL_PATH):
        
            self.clip = KandinskyCLIPImageEncoder(device= self.device)
            self.clip.load_submodels()

            self.clip_text_embedder= KandinskyCLIPTextEmbedder(device= self.device)
            self.clip_text_embedder.load_submodels()
            
            self.unet = UNet2DConditionModel.from_pretrained(decoder_path, local_files_only=True, subfolder='unet').to(torch.float16).to(self.device)
            
            self.prior_model = KandinskyV22PriorPipeline.from_pretrained(prior_path, local_files_only=True, 
                                                                    image_encoder=self.clip.vision_model, torch_dtype=torch.float16).to(self.device)
            self.decoder_model = KandinskyV22Pipeline.from_pretrained(decoder_path, local_files_only=True, use_safetensors=True, 
                                                                unet=self.unet, torch_dtype=torch.float16).to(self.device)
            self.inpainting_decoder_model = KandinskyV22InpaintPipeline.from_pretrained(inpaint_decoder_path, local_files_only=True,
                                                                                        unet=self.unet, torch_dtype=torch.float16).to(self.device)

            self.img2img_decoder = KandinskyV22Img2ImgPipeline.from_pretrained(decoder_path, local_files_only=True,
                                                                    unet=self.unet, torch_dtype=torch.float16).to(self.device)