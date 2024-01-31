import hashlib
import io
import os
import sys
import torch
from kandinsky2 import get_kandinsky2
from PIL import Image
import numpy as np

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from utility.labml.monit import section
from utility.path import separate_bucket_and_file_path
from utility.utils_logger import logger
from utility.minio import cmd

class KandinskyInpaintingPipeline:
    def __init__(self,
                 denoising_strength=0.75,
                 guidance_scale=7.5,
                 steps=150,
                 width=512,
                 height=512
                 ):
        """
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

        self.inpainting_model=None


        # get device
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def load_models(self):
        with section("Loading Inpainting model"):
            self.inpainting_model= get_kandinsky2(self.device, task_type='inpainting', model_version='2.2', use_flash_attention=False)
            logger.debug(f"Kandinsky Inpainting model successfully loaded")
   
    def unload_models(self):
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

        images = self.inpainting_model.generate_inpainting(
                prompt,
                initial_image, 
                image_mask, 
                num_steps=self.steps,
                batch_size=1, 
                guidance_scale=self.guidance_scale,
                h=self.height, w=self.width,
                sampler='p_sampler', 
                prior_cf_scale=4,
                prior_steps="5"
                )

        return images[0]

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
