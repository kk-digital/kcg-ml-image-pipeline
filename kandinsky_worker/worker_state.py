import io
import queue
import sys
import torch
import msgpack
from torch.nn.functional import cosine_similarity

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
from utility.http import request
from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from utility.minio import cmd
from kandinsky_worker.dataloaders.image_embedding import ImageEmbedding 
from worker.generation_task.generation_task import GenerationTask
from data_loader.utils import get_object
from utility.path import separate_bucket_and_file_path

class WorkerState:
    def __init__(self, device, minio_access_key, minio_secret_key, queue_size):
        self.device = device
        self.config = ModelPathConfig()
        self.prior_model= None
        self.decoder_model= None
        self.inpainting_decoder_model= None
        self.clip_text_embedder= None
        self.scoring_models= {}
        self.minio_client = get_minio_client(minio_access_key, minio_secret_key)
        self.queue_size = queue_size
        self.job_queue = queue.Queue()
        self.self_training_data={}
        self.dataset_list=[]
    
    def load_scoring_models(self):
        dataset_list= request.http_get_dataset_names()

        for dataset in dataset_list:
            scoring_model= self.load_scoring_model(dataset)

            if scoring_model is not None:
                 self.scoring_models[dataset]= scoring_model
                 self.self_training_data[dataset]=[]
                 self.dataset_list.append(dataset)

    # load elm scoring models
    def load_scoring_model(self, dataset):
        input_path=f"{dataset}/models/ranking/"
        scoring_model = ABRankingELMModel(1280, device=self.device)
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
            
            unet = UNet2DConditionModel.from_pretrained(decoder_path, local_files_only=True, subfolder='unet').to(torch.float16).to(self.device)
            
            self.decoder_model = KandinskyV22Pipeline.from_pretrained(decoder_path, local_files_only=True, use_safetensors=True, 
                                                                unet=unet, torch_dtype=torch.float16).to(self.device)
            
            self.img2img_decoder = KandinskyV22Img2ImgPipeline.from_pretrained(decoder_path, local_files_only=True,
                                                                    unet=unet, torch_dtype=torch.float16).to(self.device)
            del unet

            inpainting_unet = UNet2DConditionModel.from_pretrained(inpaint_decoder_path, local_files_only=True, subfolder='unet').to(torch.float16).to(self.device)

            self.inpainting_decoder_model = KandinskyV22InpaintPipeline.from_pretrained(inpaint_decoder_path, local_files_only=True,
                                                                                        unet=inpainting_unet, torch_dtype=torch.float16).to(self.device)
            del inpainting_unet

            self.prior_model = KandinskyV22PriorPipeline.from_pretrained(prior_path, local_files_only=True, 
                                                                    image_encoder=self.clip.vision_model, torch_dtype=torch.float16).to(self.device) 
            # get scoring models
            self.load_scoring_models()
    
    def get_input_clip_vector(self, output_file_path):
        image_embeddings_path = output_file_path.replace("_clip_kandinsky.msgpack", "_embedding.msgpack")    
        embedding_data = get_object(self.minio_client, image_embeddings_path)
        embedding_dict = ImageEmbedding.from_msgpack_bytes(embedding_data)
        features_vector= embedding_dict.image_embedding

        return features_vector

    def get_output_clip_vector(self, output_file_path):
        features_data = get_object(self.minio_client, output_file_path)
        features_vector = msgpack.unpackb(features_data)["clip-feature-vector"]
        features_vector= torch.tensor(features_vector)

        return features_vector

    def calculate_self_training_data(self, job: dict):
        output_file_path= job['task_output_file_dict']['output_file_path']
        bucket_name, file_path = separate_bucket_and_file_path(output_file_path)
        dataset= file_path.split('/')[0]
        scoring_model= self.scoring_models[dataset]

        if scoring_model is None:
            raise Exception("No scoring model has been loaded for this dataset.")

        score_mean= float(scoring_model.mean)
        score_std= float(scoring_model.standard_deviation)

        input_clip_vector= self.get_input_clip_vector(file_path)
        output_clip_vector= self.get_output_clip_vector(file_path)

        input_clip_score = scoring_model.predict_clip(input_clip_vector.to(device=scoring_model._device)).item()
        input_clip_score = (input_clip_score - score_mean) / score_std 
        output_clip_score = scoring_model.predict_clip(output_clip_vector.to(device=scoring_model._device)).item()
        output_clip_score = (output_clip_score - score_mean) / score_std 

        cosine_sim = cosine_similarity(input_clip_vector, output_clip_vector).item()

        data = {
            'input_clip': input_clip_vector.detach().cpu().numpy().tolist(),
            'output_clip': output_clip_vector.detach().cpu().numpy().tolist(),
            'input_clip_score': input_clip_score,
            'output_clip_score': output_clip_score,
            'cosine_sim': cosine_sim
        }

        # save the data to self training
        self.self_training_data[dataset].append(data)


    # function for storing self training data in a msgpack file
    def store_self_training_data(self):
        
        for dataset in self.dataset_list:
            if len(self.self_training_data[dataset])<1000:
                continue

            index = request.http_get_self_training_sequential_id(dataset)

            batch= self.self_training_data[dataset].copy()
            self.self_training_data[dataset]=[]
        
            file_path=f"{str(index).zfill(4)}.msgpack"
            packed_data = msgpack.packb(batch, use_single_float=True)

            # Create a BytesIO buffer with the packed data
            buffer = io.BytesIO(packed_data)
            buffer.seek(0)  # Make sure to seek to the start of the BytesIO buffer

            minio_path = f"{dataset}/data/latent-generator/self_training/{file_path}"
            cmd.upload_data(self.minio_client, 'datasets', minio_path, buffer)