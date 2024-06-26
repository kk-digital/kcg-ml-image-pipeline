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
                 device,
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

        self.device = device


    def load_models(self, prior_path=PRIOR_MODEL_PATH, decoder_path= DECODER_MODEL_PATH, 
                    inpaint_decoder_path= INPAINT_DECODER_MODEL_PATH, task_type="inpainting"):
        
        with section("Loading kandinsky models"):
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(prior_path, local_files_only=True, subfolder='image_encoder').to(torch.float16).to(self.device)
            if task_type == "text2img":
                unet = UNet2DConditionModel.from_pretrained(decoder_path, local_files_only=True, subfolder='unet').to(torch.float16).to(self.device)
                
                self.prior = KandinskyV22PriorPipeline.from_pretrained(prior_path, local_files_only=True, 
                                                                       image_encoder=self.image_encoder, torch_dtype=torch.float16).to(self.device)
                self.decoder = KandinskyV22Pipeline.from_pretrained(decoder_path, local_files_only=True, use_safetensors=True, 
                                                                    unet=unet, torch_dtype=torch.float16).to(self.device)
            elif task_type == "inpainting":
                unet = UNet2DConditionModel.from_pretrained(inpaint_decoder_path, local_files_only=True, subfolder='unet').to(torch.float16).to(self.device)
                self.prior = KandinskyV22PriorPipeline.from_pretrained(prior_path, local_files_only=True, 
                                                                       image_encoder=self.image_encoder, torch_dtype=torch.float16).to(self.device)
                self.decoder = KandinskyV22InpaintPipeline.from_pretrained(inpaint_decoder_path, local_files_only=True, 
                                                                    unet=unet, torch_dtype=torch.float16).to(self.device)
            elif task_type == "img2img":
                unet = UNet2DConditionModel.from_pretrained(decoder_path, local_files_only=True, subfolder='unet').to(torch.float16).to(self.device)
                self.decoder = KandinskyV22Img2ImgPipeline.from_pretrained(decoder_path, local_files_only=True,
                                                                    unet=unet, torch_dtype=torch.float16).to(self.device)
                
            else:
                raise ValueError("Only text2img, img2img, inpainting is available")
            
            del unet
            logger.debug(f"Kandinsky Inpainting model successfully loaded")
   
    def set_models(self, image_encoder, prior_model, decoder_model):
        # setting the kandinsky submodels directly
        self.image_encoder=image_encoder
        self.prior = prior_model
        self.decoder = decoder_model
        
    def unload_models(self):
        if self.image_encoder is not None:
            self.image_encoder.to("cpu")
            del self.image_encoder
            self.image_encoder = None
        
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
        negative_decoder_prompt="",
        seed=None
    ):
        
        if seed is not None:
            generator=torch.Generator(device=self.device).manual_seed(seed)

        with torch.no_grad():
            if seed is not None:
                img_emb = self.prior(prompt=prompt, num_inference_steps=self.prior_steps,
                                num_images_per_prompt=self.batch_size, guidance_scale=self.prior_guidance_scale,
                                negative_prompt=negative_prior_prompt, generator= generator)
                negative_emb = self.prior(prompt=negative_prior_prompt, num_inference_steps=self.prior_steps,
                                    num_images_per_prompt=self.batch_size, guidance_scale=self.prior_guidance_scale,
                                    generator= generator)
            else:
                img_emb = self.prior(prompt=prompt, num_inference_steps=self.prior_steps,
                                num_images_per_prompt=self.batch_size, guidance_scale=self.prior_guidance_scale,
                                negative_prompt=negative_prior_prompt)
                negative_emb = self.prior(prompt=negative_prior_prompt, num_inference_steps=self.prior_steps,
                                    num_images_per_prompt=self.batch_size, guidance_scale=self.prior_guidance_scale)
                
        if negative_decoder_prompt == "":
            negative_emb = negative_emb.negative_image_embeds
        else:
            negative_emb = negative_emb.image_embeds
        
        with torch.no_grad():
            if seed is not None:
                images, latents = self.decoder(image_embeds=img_emb.image_embeds, negative_image_embeds=negative_emb,
                                num_inference_steps=self.decoder_steps, height=self.height,
                                width=self.width, guidance_scale=self.decoder_guidance_scale,
                                image=initial_img, mask_image=img_mask, generator= generator)
            
            else:
                images, latents = self.decoder(image_embeds=img_emb.image_embeds, negative_image_embeds=negative_emb,
                                num_inference_steps=self.decoder_steps, height=self.height,
                                width=self.width, guidance_scale=self.decoder_guidance_scale,
                                image=initial_img, mask_image=img_mask)
                
        return images[0], latents
    
    def generate_text2img(
        self,
        prompt,
        negative_prior_prompt="",
        negative_decoder_prompt="",
        seed=None
    ):
        height, width = self.get_new_h_w(self.height, self.width)

        if seed is not None:
            generator=torch.Generator(device=self.device).manual_seed(seed)

        with torch.no_grad():
            if seed is not None:
                img_emb = self.prior(prompt=prompt, num_inference_steps=self.prior_steps,
                                num_images_per_prompt=self.batch_size, guidance_scale=self.prior_guidance_scale,
                                negative_prompt=negative_prior_prompt, generator= generator)
                negative_emb = self.prior(prompt=negative_decoder_prompt, num_inference_steps=self.prior_steps,
                                    num_images_per_prompt=self.batch_size, guidance_scale=self.prior_guidance_scale, 
                                    generator= generator)
            else:
                img_emb = self.prior(prompt=prompt, num_inference_steps=self.prior_steps,
                                num_images_per_prompt=self.batch_size, guidance_scale=self.prior_guidance_scale,
                                negative_prompt=negative_prior_prompt)
                negative_emb = self.prior(prompt=negative_decoder_prompt, num_inference_steps=self.prior_steps,
                                    num_images_per_prompt=self.batch_size, guidance_scale=self.prior_guidance_scale)
            
        if negative_decoder_prompt == "":
            negative_emb = negative_emb.negative_image_embeds
        else:
            negative_emb = negative_emb.image_embeds
        
        with torch.no_grad():
            if seed is not None:
                images, latents = self.decoder(image_embeds=img_emb.image_embeds, negative_image_embeds=negative_emb,
                                num_inference_steps=self.decoder_steps, height=height,
                                width=width, guidance_scale=self.decoder_guidance_scale, generator=generator)
            else:
                images, latents = self.decoder(image_embeds=img_emb.image_embeds, negative_image_embeds=negative_emb,
                                num_inference_steps=self.decoder_steps, height=height,
                                width=width, guidance_scale=self.decoder_guidance_scale)
                
        return images[0], latents

    def get_zero_embed(self, batch_size=1):
        zero_img = torch.zeros(1, 3, self.image_encoder.config.image_size, self.image_encoder.config.image_size).to(
            device=self.device, dtype=self.image_encoder.dtype
        )
        zero_image_emb = self.image_encoder(zero_img)["image_embeds"]
        zero_image_emb = zero_image_emb.repeat(batch_size, 1)
        return zero_image_emb
 
    def generate_img2img(
        self,
        init_img,
        image_embeds,
        init_vae=None,
        negative_image_embeds=None,
        seed=None
    ):
        height, width = self.get_new_h_w(self.height, self.width)
        if negative_image_embeds==None:
            negative_image_embeds= self.get_zero_embed()
        
        image_embeds = image_embeds.type(self.decoder.dtype)
        negative_image_embeds = negative_image_embeds.type(self.decoder.dtype)

        with torch.no_grad():
            if seed:
                generator=torch.Generator(device=self.device).manual_seed(seed)
                images, latents = self.decoder(
                    image=init_img,
                    image_embeds=image_embeds,
                    initial_latents= init_vae,
                    negative_image_embeds= negative_image_embeds, 
                    guidance_scale=self.decoder_guidance_scale,
                    num_inference_steps=self.decoder_steps,
                    height=height,
                    width=width,
                    strength= self.strength,
                    generator= generator
                )
            else:
                images, latents = self.decoder(
                    image=init_img,
                    image_embeds=image_embeds,
                    initial_latents= init_vae,
                    negative_image_embeds= negative_image_embeds, 
                    guidance_scale=self.decoder_guidance_scale,
                    num_inference_steps=self.decoder_steps,
                    height=height,
                    width=width,
                    strength= self.strength
                )
        
        return images[0], latents
    
    def generate_img2img_in_batches(
        self,
        init_imgs,
        image_embeds,
        batch_size,
        negative_image_embeds=None,
        seed=None
    ):
        height, width = self.get_new_h_w(self.height, self.width)
        if negative_image_embeds==None:
            negative_image_embeds= self.get_zero_embed(batch_size=batch_size)
        
        with torch.no_grad():
            if seed:
                generator=torch.Generator(device=self.device).manual_seed(seed)
                images, latents = self.decoder(
                    image=init_imgs,
                    image_embeds=image_embeds,
                    negative_image_embeds= negative_image_embeds, 
                    guidance_scale=self.decoder_guidance_scale,
                    num_inference_steps=self.decoder_steps,
                    height=height,
                    width=width,
                    strength= self.strength,
                    generator= generator
                )
            else:
                images, latents = self.decoder(
                    image=init_imgs,
                    image_embeds=image_embeds,
                    negative_image_embeds= negative_image_embeds, 
                    guidance_scale=self.decoder_guidance_scale,
                    num_inference_steps=self.decoder_steps,
                    height=height,
                    width=width,
                    strength= self.strength
                )
        
        return images, latents
  
    def generate_img2img_inpainting(
        self,
        init_img,
        img_mask,
        image_embeds,
        negative_image_embeds=None,
        seed=None
    ):
        if negative_image_embeds==None:
            negative_image_embeds= self.get_zero_embed()
        
        with torch.no_grad():
            if seed:
                generator=torch.Generator(device=self.device).manual_seed(seed)
                images, latents = self.decoder(
                    image_embeds=image_embeds, 
                    negative_image_embeds=negative_image_embeds,
                    num_inference_steps=self.decoder_steps, 
                    height=self.height,
                    width=self.width, 
                    guidance_scale=self.decoder_guidance_scale,
                    image=init_img, 
                    mask_image=img_mask,
                    generator= generator
                )
            else:
                images, latents = self.decoder(
                    image_embeds=image_embeds, 
                    negative_image_embeds=negative_image_embeds,
                    num_inference_steps=self.decoder_steps, 
                    height=self.height,
                    width=self.width, 
                    guidance_scale=self.decoder_guidance_scale,
                    image=init_img, 
                    mask_image=img_mask
                )
        
        return images[0], latents
    
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