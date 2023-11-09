import os
import sys
from typing import List
import torch
from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextConfig

sys.path.insert(0, os.getcwd())

from stable_diffusion.model_paths import CLIP_TEXT_EMBEDDER_PATH, CLIP_TOKENIZER_DIR_PATH, CLIP_TEXT_MODEL_DIR_PATH, CLIP_MODEL_PATH

class CLIPTextEmbedder(nn.Module):
    def __init__(self, device='cpu', max_length: int = 77, tokenizer=None, transformer=None):
        super().__init__()

        self.model_name = "openai/clip-vit-L-14"

        self.device = device

        self.tokenizer = tokenizer
        self.transformer = transformer

        self.max_length = max_length
        self.to(self.device)

    def init_submodels(self, tokenizer_path: str = CLIP_TOKENIZER_DIR_PATH, transformer_path: str = CLIP_TEXT_MODEL_DIR_PATH):
        config = CLIPTextConfig.from_pretrained(transformer_path, local_files_only=True)
        self.transformer = CLIPTextModel(config).eval().to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    def forward(self, prompts: List[str]):
        batch_encoding = self.tokenizer(prompts, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        self.transformer = self.transformer.to(device=self.device)
        return self.transformer(input_ids=tokens).last_hidden_state


if __name__ == "__main__":
    prompts = "environmental, pixel art, concept art, side scrolling, video game, neo city, (1 girl), white box, puffy lips, cinematic lighting, colorful, steampunk, partially submerged, original, 1girl, night, ribbon choker, see through top, black tissues, a masterpiece, high heel, hand on own crotch"
    phrase= "environmental"
    embedder = CLIPTextEmbedder()
    embedder.init_submodels()
    with torch.no_grad():
        prompt_embeddings = embedder(prompts)
        phrase_embeddings = embedder(phrase)
        print("Shape of produced embedding for prompt:", prompt_embeddings.shape)
        print("Shape of produced embeddings for phrase:", phrase_embeddings.shape)
