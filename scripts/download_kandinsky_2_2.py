import os
import torch
from diffusers import AutoPipelineForInpainting

TARGET_DIR = './hub_folder/'

os.makedirs(TARGET_DIR, exist_ok=True)

# Step 1: download weights & export folders

pipe = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", 
    torch_dtype=torch.float16,
    resume_download=True
)

pipe.prior_scheduler.save_pretrained(os.path.join(TARGET_DIR, 'kandinsky-2-2-prior', 'scheduler'))
pipe.prior_prior.save_pretrained(os.path.join(TARGET_DIR, 'kandinsky-2-2-prior', 'prior'))
pipe.prior_image_processor.save_pretrained(os.path.join(TARGET_DIR, 'kandinsky-2-2-prior', 'image_processor'))
pipe.prior_image_encoder.save_pretrained(os.path.join(TARGET_DIR, 'kandinsky-2-2-prior', 'image_encoder'))
pipe.prior_tokenizer.save_pretrained(os.path.join(TARGET_DIR, 'kandinsky-2-2-prior', 'tokenizer'))
pipe.prior_text_encoder.save_pretrained(os.path.join(TARGET_DIR, 'kandinsky-2-2-prior', 'text_encoder'))

pipe.scheduler.save_pretrained(os.path.join(TARGET_DIR, 'kandinsky-2-2-decoder', 'scheduler'))
pipe.unet.save_pretrained(os.path.join(TARGET_DIR, 'kandinsky-2-2-decoder', 'unet'))
pipe.movq.save_pretrained(os.path.join(TARGET_DIR, 'kandinsky-2-2-decoder', 'movq'))

pipe = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", 
    torch_dtype=torch.float16,
    resume_download=True
)

pipe.scheduler.save_pretrained(os.path.join(TARGET_DIR, 'kandinsky-2-2-decoder-inpaint', 'scheduler'))
pipe.unet.save_pretrained(os.path.join(TARGET_DIR, 'kandinsky-2-2-decoder-inpaint', 'unet'))
pipe.movq.save_pretrained(os.path.join(TARGET_DIR, 'kandinsky-2-2-decoder-inpaint', 'movq'))

# Step 2: pack folders

os.system(r'cd {TARGET_DIR} | zip -rq kandinsky-2-2.zip kandinsky-2-2-prior kandinsky-2-2-decoder kandinsky-2-2-decoder-inpaint')

# Step 3: load from folders

from diffusers import DDPMScheduler, UNet2DConditionModel, VQModel, PriorTransformer, UnCLIPScheduler
from diffusers import KandinskyV22CombinedPipeline, KandinskyV22InpaintCombinedPipeline
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPTokenizer, CLIPImageProcessor

prior_scheduler = UnCLIPScheduler.from_pretrained(os.path.join(TARGET_DIR, 'kandinsky-2-2-prior', 'scheduler'))
prior_prior = PriorTransformer.from_pretrained(os.path.join(TARGET_DIR, 'kandinsky-2-2-prior', 'prior'))
prior_image_encoder = CLIPVisionModelWithProjection.from_pretrained(os.path.join(TARGET_DIR, 'kandinsky-2-2-prior', 'image_encoder'))
prior_text_encoder = CLIPTextModelWithProjection.from_pretrained(os.path.join(TARGET_DIR, 'kandinsky-2-2-prior', 'text_encoder'))
prior_tokenizer = CLIPTokenizer.from_pretrained(os.path.join(TARGET_DIR, 'kandinsky-2-2-prior', 'tokenizer'))
prior_image_processor = CLIPImageProcessor.from_pretrained(os.path.join(TARGET_DIR, 'kandinsky-2-2-prior', 'image_processor'))

scheduler = DDPMScheduler.from_pretrained(os.path.join(TARGET_DIR, 'kandinsky-2-2-decoder', 'scheduler'))
unet = UNet2DConditionModel.from_pretrained(os.path.join(TARGET_DIR, 'kandinsky-2-2-decoder', 'unet'))
movq = VQModel.from_pretrained(os.path.join(TARGET_DIR, 'kandinsky-2-2-decoder', 'movq'))

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

scheduler = DDPMScheduler.from_pretrained(os.path.join(TARGET_DIR, 'kandinsky-2-2-decoder-inpaint', 'scheduler'))
unet = UNet2DConditionModel.from_pretrained(os.path.join(TARGET_DIR, 'kandinsky-2-2-decoder-inpaint', 'unet'))
movq = VQModel.from_pretrained(os.path.join(TARGET_DIR, 'kandinsky-2-2-decoder-inpaint', 'movq'))

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
