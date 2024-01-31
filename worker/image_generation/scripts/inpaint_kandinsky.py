import hashlib
import io
import os
import sys
import torch
from diffusers import KandinskyV22PriorPipeline, KandinskyV22InpaintPipeline
from transformers import CLIPVisionModelWithProjection
from diffusers.models import UNet2DConditionModel
from transformers import CLIPVisionModelWithProjection
from diffusers.models import UNet2DConditionModel

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

    def load_models(self, image_encoder_path, unet_path, prior_path, decoder_path):
        with section("Loading Inpainting model"):
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path, 
                                                                                local_files_only=True,
                                                                                use_safetensors=True).to(torch.float16).to(self.device)
            
            self.unet = UNet2DConditionModel.from_pretrained(unet_path, local_files_only=True,
                                                                        use_safetensors=True).to(torch.float16).to(self.device)
            
            self.prior = KandinskyV22PriorPipeline.from_pretrained(prior_path, 
                                                                   local_files_only=True,
                                                                   use_safetensors=True, 
                                                                   image_encoder=self.image_encoder, 
                                                                   torch_dtype=torch.float16).to(self.device)
            
            self.decoder = KandinskyV22InpaintPipeline.from_pretrained(decoder_path, 
                                                                   local_files_only=True,
                                                                   use_safetensors=True, 
                                                                   unet=self.unet, 
                                                                   torch_dtype=torch.float16).to(self.device)
            logger.debug(f"Kandinsky Inpainting model successfully loaded")
   
    def unload_models(self):
        pass

    def generate_inpainting(
        self,
        prompt,
        initial_img,
        img_mask,
        batch_size=1,
        decoder_steps=50,
        prior_steps=25,
        decoder_guidance_scale=4,
        prior_guidance_scale=4,
        h=512,
        w=512,
        negative_prior_prompt="",
        negative_decoder_prompt="",
    ):
        
        img_emb = self.prior(prompt=prompt, num_inference_steps=prior_steps,
                        num_images_per_prompt=batch_size, guidance_scale=prior_guidance_scale,
                        negative_prompt=negative_prior_prompt)
        negative_emb = self.prior(prompt=negative_prior_prompt, num_inference_steps=prior_steps,
                             num_images_per_prompt=batch_size, guidance_scale=prior_guidance_scale)
        if negative_decoder_prompt == "":
            negative_emb = negative_emb.negative_image_embeds
        else:
            negative_emb = negative_emb.image_embeds
        images = self.decoder(image_embeds=img_emb.image_embeds, negative_image_embeds=negative_emb,
                         num_inference_steps=decoder_steps, height=h,
                         width=w, guidance_scale=decoder_guidance_scale,
                         image=initial_img, mask_image=img_mask).images
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
