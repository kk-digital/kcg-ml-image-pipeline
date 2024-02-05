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
)

pipe = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", 
    torch_dtype=torch.float16,
    resume_download=True
)

# Step 2: export folders

os.system(f'cp -r ~/.cache/huggingface/hub/models--kandinsky-community--* {TARGET_DIR}')

# Step 3: pack folders

# I don't know if this step is necessary. 
# If minio or torrent supports uploading the entire folder, 
# this step is not needed.

os.system(f'cd {TARGET_DIR} | zip -r kandinsky-2-2.zip models--kandinsky-community--kandinsky-2-2-prior/ models--kandinsky-community--kandinsky-2-2-decoder/ models--kandinsky-community--kandinsky-2-2-decoder-inpaint/')

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

