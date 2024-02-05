import os
import torch
from diffusers import AutoPipelineForInpainting

TARGET_DIR = './hub_folder/'

os.makedirs(TARGET_DIR, exist_ok=True)

# Step 1: download weights

pipe = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", 
    torch_dtype=torch.float16,
    resume_download=True
    # cache_dir='/home/user/',
    # local_files_only=True
)

pipe = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", 
    torch_dtype=torch.float16,
    resume_download=True
    # cache_dir='/home/user/',
    # local_files_only=True
)

# Step 2: export folders

os.system(f'cp -r ~/.cache/huggingface/hub/models--kandinsky-community--* {TARGET_DIR}')

# Step 3: pack folders

# I don't know if this step is necessary. 
# If minio or torrent supports uploading the entire folder, 
# this step is not needed.

os.system(r'cd {TARGET_DIR} | zip -rq kandinsky-2-2-prior.zip models--kandinsky-community--kandinsky-2-2-prior')
os.system(r'cd {TARGET_DIR} | zip -rq kandinsky-2-2-decoder.zip models--kandinsky-community--kandinsky-2-2-decoder')
os.system(r'cd {TARGET_DIR} | zip -rq kandinsky-2-2-decoder-inpaint.zip models--kandinsky-community--kandinsky-2-2-decoder-inpaint')

# Step 4: load from folders

pipe = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", 
    torch_dtype=torch.float16,
    cache_dir=TARGET_DIR,
    local_files_only=True
)

pipe = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", 
    torch_dtype=torch.float16,
    cache_dir=TARGET_DIR,
    local_files_only=True
)

