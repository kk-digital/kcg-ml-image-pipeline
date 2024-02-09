import gc
import subprocess
from pathlib import Path

import torch
from diffusers import AutoPipelineForInpainting

TARGET_DIR = Path("input/model/kandinsky")
TARGET_DIR.mkdir(exist_ok=True)


# Function to save the components of the prior model
def save_prior_components(pipe, prior_folder):
    pipe.prior_scheduler.save_pretrained(prior_folder / "scheduler")
    pipe.prior_prior.save_pretrained(prior_folder / "prior")
    pipe.prior_image_processor.save_pretrained(prior_folder / "image_processor")
    pipe.prior_image_encoder.save_pretrained(prior_folder / "image_encoder")
    pipe.prior_tokenizer.save_pretrained(prior_folder / "tokenizer")
    pipe.prior_text_encoder.save_pretrained(prior_folder / "text_encoder")


# Function to save the components of the decoder model
def save_decoder_components(pipe, decoder_folder):
    pipe.scheduler.save_pretrained(decoder_folder / "scheduler")
    pipe.unet.save_pretrained(decoder_folder / "unet")
    pipe.movq.save_pretrained(decoder_folder / "movq")


# Download and save the kandinsky-2-2-decoder
pipe = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder",
    torch_dtype=torch.float16,
    resume_download=True,
)
prior_folder = TARGET_DIR / "kandinsky-2-2-prior"
decoder_folder = TARGET_DIR / "kandinsky-2-2-decoder"
save_prior_components(pipe, prior_folder)
save_decoder_components(pipe, decoder_folder)

del pipe
torch.cuda.empty_cache()
gc.collect()  # Collect garbage in CPU memory
print("Memory cleared")

# Download and save the kandinsky-2-2-decoder-inpaint
pipe = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint",
    torch_dtype=torch.float16,
    resume_download=True,
)
decoder_inpaint_folder = TARGET_DIR / "kandinsky-2-2-decoder-inpaint"
save_decoder_components(pipe, decoder_inpaint_folder)

del pipe
torch.cuda.empty_cache()
gc.collect()  # Collect garbage in CPU memory
print("Memory cleared")


# Step 3: load from folders

from diffusers import (
    DDPMScheduler,
    KandinskyV22CombinedPipeline,
    KandinskyV22InpaintCombinedPipeline,
    PriorTransformer,
    UnCLIPScheduler,
    UNet2DConditionModel,
    VQModel,
)
from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

prior_scheduler = UnCLIPScheduler.from_pretrained(prior_folder / "scheduler")
prior_prior = PriorTransformer.from_pretrained(prior_folder / "prior")
prior_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    prior_folder / "image_encoder"
)
prior_text_encoder = CLIPTextModelWithProjection.from_pretrained(
    prior_folder / "text_encoder"
)
prior_tokenizer = CLIPTokenizer.from_pretrained(prior_folder / "tokenizer")
prior_image_processor = CLIPImageProcessor.from_pretrained(
    prior_folder / "image_processor"
)

scheduler = DDPMScheduler.from_pretrained(decoder_folder / "scheduler")
unet = UNet2DConditionModel.from_pretrained(decoder_folder / "unet")
movq = VQModel.from_pretrained(decoder_folder / "movq")

pipe = KandinskyV22CombinedPipeline(
    prior_scheduler=prior_scheduler,
    prior_prior=prior_prior,
    prior_image_encoder=prior_image_encoder,
    prior_text_encoder=prior_text_encoder,
    prior_tokenizer=prior_tokenizer,
    prior_image_processor=prior_image_processor,
    scheduler=scheduler,
    unet=unet,
    movq=movq,
)

scheduler = DDPMScheduler.from_pretrained(decoder_inpaint_folder / "scheduler")
unet = UNet2DConditionModel.from_pretrained(decoder_inpaint_folder / "unet")
movq = VQModel.from_pretrained(decoder_inpaint_folder / "movq")

pipe = KandinskyV22InpaintCombinedPipeline(
    prior_scheduler=prior_scheduler,
    prior_prior=prior_prior,
    prior_image_encoder=prior_image_encoder,
    prior_text_encoder=prior_text_encoder,
    prior_tokenizer=prior_tokenizer,
    prior_image_processor=prior_image_processor,
    scheduler=scheduler,
    unet=unet,
    movq=movq,
)
