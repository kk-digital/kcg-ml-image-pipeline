import hashlib
import io
import os
import sys
import torch
from kandinsky.pipelines.kandinsky_text2img import KandinskyV22Pipeline
from kandinsky.pipelines.kandinsky_img2img import KandinskyV22Img2ImgPipeline
from kandinsky.pipelines.kandinsky_prior import KandinskyV22PriorPipeline
from kandinsky.pipelines.kandinsky_inpainting import KandinskyV22InpaintPipeline
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
from kandinsky.model_paths import PRIOR_MODEL_PATH, DECODER_MODEL_PATH, INPAINT_DECODER_MODEL_PATH


class KandinskyPipeline:
    def __init__(self,
                 width=512,
                 height=512,
                 batch_size=1,
                 decoder_steps=50,
                 prior_steps=25,
                 strength=0.4,
                 decoder_guidance_scale=4,
                 prior_guidance_scale=4
                ):
        """
        steps (int): The number of optimization steps to perform during inpainting. More steps may lead to finer inpainting results.
        width (int): The width of the inpainting canvas. Specifies the horizontal dimension of the image to be inpainted.
        height (int): The height of the inpainting canvas. Specifies the vertical dimension of the image to be inpainted.
        """

        # set parameters
        self.batch_size= batch_size
        self.prior_guidance_scale=prior_guidance_scale
        self.decoder_guidance_scale=decoder_guidance_scale
        self.strength=strength
        self.prior_steps= prior_steps
        self.decoder_steps=decoder_steps
        self.width=width
        self.height=height

        self.inpainting_model=None

        # get device
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def load_models(self, prior_path=PRIOR_MODEL_PATH, decoder_path= DECODER_MODEL_PATH, 
                    inpaint_decoder_path= INPAINT_DECODER_MODEL_PATH, task_type="inpainting"):
        
        with section("Loading Inpainting model"):
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(prior_path, local_files_only=True, subfolder='image_encoder').to(torch.float16).to(self.device)
            if task_type == "text2img":
                self.unet = UNet2DConditionModel.from_pretrained(decoder_path, local_files_only=True, subfolder='unet').to(torch.float16).to(self.device)
                
                self.prior = KandinskyV22PriorPipeline.from_pretrained(prior_path, local_files_only=True, 
                                                                       image_encoder=self.image_encoder, torch_dtype=torch.float16).to(self.device)
                self.decoder = KandinskyV22Pipeline.from_pretrained(decoder_path, local_files_only=True, use_safetensors=True, 
                                                                    unet=self.unet, torch_dtype=torch.float16).to(self.device)
            elif task_type == "inpainting":
                self.unet = UNet2DConditionModel.from_pretrained(inpaint_decoder_path, local_files_only=True, subfolder='unet').to(torch.float16).to(self.device)
                self.prior = KandinskyV22PriorPipeline.from_pretrained(prior_path, local_files_only=True, 
                                                                       image_encoder=self.image_encoder, torch_dtype=torch.float16).to(self.device)
                self.decoder = KandinskyV22InpaintPipeline.from_pretrained(inpaint_decoder_path, local_files_only=True, 
                                                                    unet=self.unet, torch_dtype=torch.float16).to(self.device)
            elif task_type == "img2img":
                self.unet = UNet2DConditionModel.from_pretrained(decoder_path, local_files_only=True, subfolder='unet').to(torch.float16).to(self.device)
                self.prior = KandinskyV22PriorPipeline.from_pretrained(prior_path, local_files_only=True,
                                                                       image_encoder=self.image_encoder, torch_dtype=torch.float16).to(self.device)
                self.decoder = KandinskyV22Img2ImgPipeline.from_pretrained(decoder_path, local_files_only=True,
                                                                    unet=self.unet, torch_dtype=torch.float16).to(self.device)
            else:
                raise ValueError("Only text2img, img2img, inpainting is available")
            
            logger.debug(f"Kandinsky Inpainting model successfully loaded")
   
    def set_models(self, image_encoder, unet, prior_model, decoder_model):
        # setting the kandinsky submodels directly
        self.image_encoder=image_encoder
        self.unet = unet
        self.prior = prior_model
        self.decoder = decoder_model
        
    def unload_models(self):
        if self.image_encoder is not None:
            self.image_encoder.to("cpu")
            del self.image_encoder
            self.image_encoder = None
        
        if self.unet is not None:
            self.unet.to("cpu")
            del self.unet
            self.unet = None
        
        if self.prior is not None:
            self.prior.to("cpu")
            del self.prior
            self.prior = None
        
        if self.decoder is not None:
            self.decoder.to("cpu")
            del self.decoder
            self.decoder = None

    def get_new_h_w(self, h, w):
        new_h = h // 64
        if h % 64 != 0:
            new_h += 1
        new_w = w // 64
        if w % 64 != 0:
            new_w += 1
        return new_h * 64, new_w * 64

    def generate_inpainting(
        self,
        prompt,
        initial_img,
        img_mask,
        negative_prior_prompt="",
        negative_decoder_prompt=""
    ):
        
        img_emb = self.prior(prompt=prompt, num_inference_steps=self.prior_steps,
                        num_images_per_prompt=self.batch_size, guidance_scale=self.prior_guidance_scale,
                        negative_prompt=negative_prior_prompt)
        negative_emb = self.prior(prompt=negative_prior_prompt, num_inference_steps=self.prior_steps,
                             num_images_per_prompt=self.batch_size, guidance_scale=self.prior_guidance_scale)
        if negative_decoder_prompt == "":
            negative_emb = negative_emb.negative_image_embeds
        else:
            negative_emb = negative_emb.image_embeds
        images, latents = self.decoder(image_embeds=img_emb.image_embeds, negative_image_embeds=negative_emb,
                         num_inference_steps=self.decoder_steps, height=self.height,
                         width=self.width, guidance_scale=self.decoder_guidance_scale,
                         image=initial_img, mask_image=img_mask)
        return images[0], latents
    
    def generate_text2img(
        self,
        prompt,
        negative_prior_prompt="",
        negative_decoder_prompt=""
    ):
        height, width = self.get_new_h_w(self.height, self.width)
        img_emb = self.prior(prompt=prompt, num_inference_steps=self.prior_steps,
                        num_images_per_prompt=self.batch_size, guidance_scale=self.prior_guidance_scale,
                        negative_prompt=negative_prior_prompt)
        negative_emb = self.prior(prompt=negative_decoder_prompt, num_inference_steps=self.prior_steps,
                             num_images_per_prompt=self.batch_size, guidance_scale=self.prior_guidance_scale)
        if negative_decoder_prompt == "":
            negative_emb = negative_emb.negative_image_embeds
        else:
            negative_emb = negative_emb.image_embeds
        images, latents = self.decoder(image_embeds=img_emb.image_embeds, negative_image_embeds=negative_emb,
                         num_inference_steps=self.decoder_steps, height=height,
                         width=width, guidance_scale=self.decoder_guidance_scale)
        return images[0], latents

    def generate_img2img(
        self,
        image,
        prompt="",
        negative_prior_prompt="",
        negative_decoder_prompt=""
    ):
        height, width = self.get_new_h_w(self.height, self.width)
        img_emb = self.prior(prompt=prompt, num_inference_steps=self.prior_steps,
                        num_images_per_prompt=self.batch_size, guidance_scale=self.prior_guidance_scale,
                        negative_prompt=negative_prior_prompt)
        negative_emb = self.prior(prompt=negative_prior_prompt, num_inference_steps=self.prior_steps,
                             num_images_per_prompt=self.batch_size, guidance_scale=self.prior_guidance_scale)
        if negative_decoder_prompt == "":
            negative_emb = negative_emb.negative_image_embeds
        else:
            negative_emb = negative_emb.image_embeds
        images = self.decoder(image_embeds=img_emb.image_embeds, negative_image_embeds=negative_emb,
                         num_inference_steps=self.decoder_steps, height=height,
                         width=width, guidance_scale=self.decoder_guidance_scale,
                             strength=self.strength, image=image).images
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