import hashlib
import io
import os
import sys
from diffusers import StableDiffusionInpaintPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from stable_diffusion.model_paths import CLIP_TOKENIZER_DIR_PATH, CLIP_TEXT_MODEL_DIR_PATH, NED_INPAINTING_PATH, DREAMSHAPER_INPAINTING_PATH, INPAINTING_CONFIG_FILE
from PIL import Image
import torch

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from utility.labml.monit import section
from utility.minio import cmd
from utility.path import separate_bucket_and_file_path
from utility.utils_logger import logger

class StableDiffusionInpaintingPipeline:
    def __init__(self,
                 model_type="ned",
                 denoising_strength=0.75,
                 guidance_scale=7.5,
                 steps=40,
                 width=512,
                 height=512
                 ):
        """
        model_type (str): is the type of inpainting model to use, the options are "ned" and "dreamshaper"
        denoising_strength (float): The strength of denoising applied during inpainting. A higher value enhances denoising effects.
        guidance_scale (float): The scale of guidance for inpainting. It controls the impact of guidance on the inpainting process.
        steps (int): The number of optimization steps to perform during inpainting. More steps may lead to finer inpainting results.
        width (int): The width of the inpainting canvas. Specifies the horizontal dimension of the image to be inpainted.
        height (int): The height of the inpainting canvas. Specifies the vertical dimension of the image to be inpainted.
        """

        # set parameters
        self.denoising_strength=denoising_strength
        self.guidance_scale=guidance_scale
        self.steps=steps
        self.width=width
        self.height=height

        self.tokenizer=None
        self.text_encoder=None
        self.inpainting_model=None

        # get model path
        if model_type=="ned":
            self.inpainting_model_path= NED_INPAINTING_PATH
        elif model_type=="dreamshaper":
            self.inpainting_model_path= DREAMSHAPER_INPAINTING_PATH

        # get device
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
    
    def load_models(self, tokenizer_path=CLIP_TOKENIZER_DIR_PATH, text_encoder_path=CLIP_TEXT_MODEL_DIR_PATH):

        with section("Loading Tokenizer and Text encoder"):
            self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, local_files_only=True, return_tensors="pt", padding=True, truncation=True)
            logger.debug(f"Tokenizer successfully loaded from : {tokenizer_path}")
            self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, local_files_only=True,
                                                             use_safetensors=True).eval().to(self.device)
            
            self.text_encoder = self.text_encoder.to(device=self.device)
            print(self.device)
            logger.debug(f"Text encoder model successfully loaded from : {text_encoder_path}")

        with section("Loading Inpainting model"):
            self.inpainting_model = StableDiffusionInpaintPipeline.from_single_file(
                pretrained_model_link_or_path=self.inpainting_model_path,
                config_files={'v1': INPAINTING_CONFIG_FILE},
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                local_files_only=True, use_safetensors=True, load_safety_checker=False
            ).to(self.device)
            logger.debug(f"Inpainting model successfully loaded from : {text_encoder_path}")
    
    def unload_models(self):
        if self.tokenizer is not None:
            self.tokenizer
            del self.tokenizer
            self.tokenizer = None
        if self.text_encoder is not None:
            self.text_encoder.to("cpu")
            del self.text_encoder
            self.text_encoder = None
        if self.inpainting_model is not None:
            self.inpainting_model.to("cpu")
            del self.inpainting_model
            self.inpainting_model = None

        torch.cuda.empty_cache()
    
    def inpaint(self, prompt:str, initial_image: Image, image_mask: Image):
        """
        prompt (str): The textual prompt that guides the inpainting process. Describes the desired content or style of the inpainted image.
        image (Image): The initial image to be inpainted. This serves as the starting point for the inpainting process.
        image_mask (Image): The mask indicating the areas of the image that should be inpainted. Specifies regions to be filled or modified.
        
        Returns:
        Image (Image): The inpainted image resulting from the application of the specified prompt and mask. Represents the final result of the inpainting process.
        """

        # resizing and converting initial image and mask
        init_image = initial_image.convert("RGB").resize((self.width, self.height))
        mask = image_mask.convert("RGB").resize((self.width, self.height))

        with torch.no_grad():
            output = self.inpainting_model(
                prompt=prompt, 
                image=init_image, 
                mask_image=mask, 
                num_inference_steps=self.steps, 
                strength=self.denoising_strength, 
                guidance_scale=self.guidance_scale
            )

        return output.images[0]

    def convert_image_to_png(self, image):
        # convert image to bytes arr
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='jpeg')
        img_byte_arr.seek(0)

        # get hash from byte array
        output_file_hash = (hashlib.sha256(img_byte_arr.getbuffer())).hexdigest()

        return output_file_hash, img_byte_arr

    def save_image_to_disk(self, image_data, output_file_path, minio_client):
        # get bucket name and file path from output path
        bucket_name, file_path = separate_bucket_and_file_path(output_file_path)
        
        # upload image data to minIO
        cmd.upload_data(minio_client, bucket_name, file_path, image_data)


# Example Usage--------------------------------------------
        
# pipeline = StableDiffusionInpaintingPipeline(model_type="ned",
#                                             denoising_strength=0.75,
#                                             guidance_scale=7.5,
#                                             steps=40,
#                                             width=512,
#                                             height=512)
# pipeline.load_models()

# result_image= pipeline.inpaint(prompt=prompt, initial_image=initial_image, image_mask= mask)