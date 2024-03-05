import io
import queue
import sys
import torch
from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from utility.minio import cmd

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
from training_worker.ab_ranking.model.ab_ranking_fc import ABRankingFCNetwork
from utility.http import request

class WorkerState:
    def __init__(self, device, minio_access_key, minio_secret_key, queue_size):
        self.device = device
        self.config = ModelPathConfig()
        self.unet= None
        self.prior_model= None
        self.decoder_model= None
        self.inpainting_decoder_model= None
        self.clip_text_embedder= None
        self.scoring_models= {}
        self.minio_client = get_minio_client(minio_access_key, minio_secret_key)
        self.queue_size = queue_size
        self.job_queue = queue.Queue()
    
    def load_scoring_models(self):
        dataset_list= request.http_get_dataset_names()

        for dataset in dataset_list:
            scoring_model= self.load_scoring_model(dataset)

            if scoring_model is not None:
                 self.scoring_models[dataset]= scoring_model

    # load elm scoring models
    def load_scoring_model(self, dataset):
        input_path=f"{dataset}/models/ranking/"
        scoring_model = ABRankingELMModel(1280)
        file_name=f"score-elm-v1-clip-h.safetensors"

        model_files=cmd.get_list_of_objects_with_prefix(self.minio_client, 'datasets', input_path)
        most_recent_model = None

        for model_file in model_files:
            if model_file.endswith(file_name):
                most_recent_model = model_file

        if most_recent_model:
            model_file_data =cmd.get_file_from_minio(self.minio_client, 'datasets', most_recent_model)
        else:
            print("No .safetensors files found in the list.")
            return None

        print(most_recent_model)

        # Create a BytesIO object and write the downloaded content into it
        byte_buffer = io.BytesIO()
        for data in model_file_data.stream(amt=8192):
            byte_buffer.write(data)
        # Reset the buffer's position to the beginning
        byte_buffer.seek(0)

        scoring_model.load_safetensors(byte_buffer)
        scoring_model.model=scoring_model.model.to(torch.device(self.device))

        return scoring_model

    def load_models(self, prior_path=PRIOR_MODEL_PATH, decoder_path= DECODER_MODEL_PATH, 
                    inpaint_decoder_path= INPAINT_DECODER_MODEL_PATH):
        
            self.clip = KandinskyCLIPImageEncoder(device= self.device)
            self.clip.load_submodels()

            self.clip_text_embedder= KandinskyCLIPTextEmbedder(device= self.device)
            self.clip_text_embedder.load_submodels()
            
            self.unet = UNet2DConditionModel.from_pretrained(decoder_path, local_files_only=True, subfolder='unet').to(torch.float16).to(self.device)

            self.inpainting_unet = UNet2DConditionModel.from_pretrained(inpaint_decoder_path, local_files_only=True, subfolder='unet').to(torch.float16).to(self.device)
            
            self.prior_model = KandinskyV22PriorPipeline.from_pretrained(prior_path, local_files_only=True, 
                                                                    image_encoder=self.clip.vision_model, torch_dtype=torch.float16).to(self.device)
            self.decoder_model = KandinskyV22Pipeline.from_pretrained(decoder_path, local_files_only=True, use_safetensors=True, 
                                                                unet=self.unet, torch_dtype=torch.float16).to(self.device)
            self.inpainting_decoder_model = KandinskyV22InpaintPipeline.from_pretrained(inpaint_decoder_path, local_files_only=True,
                                                                                        unet=self.unet, torch_dtype=torch.float16).to(self.device)

            self.img2img_decoder = KandinskyV22Img2ImgPipeline.from_pretrained(decoder_path, local_files_only=True,
                                                                    unet=self.unet, torch_dtype=torch.float16).to(self.device)
        
            self.load_scoring_models()