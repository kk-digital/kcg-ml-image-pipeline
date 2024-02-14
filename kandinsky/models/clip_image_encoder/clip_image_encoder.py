import hashlib
import os
import sys

import PIL
import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.getcwd())
from utility.utils_logger import logger
from kandinsky.model_paths import PRIOR_MODEL_PATH
from stable_diffusion.utils_backend import get_device
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


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
                features = self.vision_model(pixel_values= image.to(self.device).half()).image_embeds
        else:
            raise ValueError(
                f"`image` can only contains elements to be of type `PIL.Image.Image` or `torch.Tensor`  but is {type(image)}"
            )
        
        return features
    
    def get_image_features(self, image):
        # Preprocess image
        if isinstance(image, PIL.Image.Image):
            image = self.image_processor(image, return_tensors="pt")['pixel_values']
        
         # Compute CLIP features
        if isinstance(image, torch.Tensor):
            with torch.no_grad():
                features = self.vision_model(pixel_values= image.to(self.device).half()).image_embeds
        else:
            raise ValueError(
                f"`image` can only contains elements to be of type `PIL.Image.Image` or `torch.Tensor`  but is {type(image)}"
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