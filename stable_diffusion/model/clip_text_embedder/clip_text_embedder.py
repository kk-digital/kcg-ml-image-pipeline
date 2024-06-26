# %%


"""
---
title: CLIP Text Embedder
summary: >
 CLIP embedder to get prompt embeddings for stable diffusion
---

# CLIP Text Embedder

This is used to get prompt embeddings for [stable diffusion](../index.html).
It uses HuggingFace Transformers CLIP model.
"""
import os
import sys
from typing import List
import safetensors
import torch
# import clip
from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextConfig, CLIPModel

sys.path.insert(0, os.getcwd())
from stable_diffusion.model_paths import CLIP_TEXT_EMBEDDER_PATH, CLIP_TOKENIZER_DIR_PATH, CLIP_TEXT_MODEL_DIR_PATH, CLIP_MODEL_PATH
from stable_diffusion.utils_backend import get_device
from utility.labml.monit import section
from utility.utils_logger import logger

MAX_LENGTH = 77

class CLIPTextEmbedder(nn.Module):
    """
    ## CLIP Text Embedder
    """

    def __init__(self, path_tree=None, device=None, max_length: int = MAX_LENGTH, tokenizer=None, transformer=None):
        """
        :param version: is the model version
        :param device: is the device
        :param max_length: is the max length of the tokenized prompt
        """
        super().__init__()

        self.model_name = "openai/clip-vit-L-14"

        self.path_tree = path_tree
        self.device = get_device(device)

        self.tokenizer = tokenizer
        self.transformer = transformer

        self.max_length = max_length
        self.to(self.device)

    def init_submodels(self, tokenizer_path: str = CLIP_TOKENIZER_DIR_PATH, transformer_path: str = CLIP_TEXT_MODEL_DIR_PATH):

        config = CLIPTextConfig.from_pretrained(transformer_path, local_files_only=True)
        self.transformer = CLIPTextModel(config).eval().to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

        self.transformer = self.transformer.to(device=self.device)

        return self

    def save_submodels(self, tokenizer_path: str = CLIP_TOKENIZER_DIR_PATH, transformer_path: str = CLIP_TEXT_MODEL_DIR_PATH):
        # self.tokenizer.save_pretrained(tokenizer_path, safe_serialization=True)
        # print("tokenizer saved to: ", tokenizer_path)
        self.transformer.save_pretrained(transformer_path, safe_serialization=True)
        # safetensors.torch.save_model(self.transformer, os.path.join(transformer_path, '/model.safetensors'))

        self.transformer = self.transformer.to(device=self.device)
        print("transformer saved to: ", transformer_path)

    def load_submodels(self, tokenizer_path=CLIP_TOKENIZER_DIR_PATH, transformer_path=CLIP_TEXT_MODEL_DIR_PATH):

        with section("Loading tokenizer and transformer"):
            self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, local_files_only=True, return_tensors="pt", padding=True, truncation=True)
            logger.debug(f"Tokenizer successfully loaded from : {tokenizer_path}")
            self.transformer = CLIPTextModel.from_pretrained(transformer_path, local_files_only=True,
                                                             use_safetensors=True).eval().to(self.device)
            # self.init_submodels(tokenizer_path = tokenizer_path, transformer_path = transformer_path)
            # safetensors.torch.load_model(self.transformer, os.path.join(transformer_path, '/model.safetensors'))
            self.transformer = self.transformer.to(device=self.device)
            print(self.device)
            logger.debug(f"CLIP text model successfully loaded from : {transformer_path}")

            return self

    def load_submodels_auto(self):

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(self.device)
        return self

    def unload_submodels(self):
        if self.tokenizer is not None:
            self.tokenizer
            del self.tokenizer
            self.tokenizer = None
        if self.transformer is not None:
            self.transformer.to("cpu")
            del self.transformer
            self.transformer = None
        torch.cuda.empty_cache()

    def save(self, embedder_path: str = CLIP_TEXT_EMBEDDER_PATH):
        try:
            safetensors.torch.save_model(self, embedder_path)
            print(f"CLIPTextEmbedder saved to: {embedder_path}")
        except Exception as e:
            print(f"CLIPTextEmbedder not saved. Error: {e}")

    def load(self, embedder_path: str = CLIP_TEXT_EMBEDDER_PATH):
        try:
            safetensors.torch.load_model(self, embedder_path, strict=True)
            logger.debug(f"CLIPTextEmbedder loaded from: {embedder_path}")
            return self
        except Exception as e:
            logger.error(f"CLIPTextEmbedder not loaded. Error: {e}")

    def forward(self, prompts: List[str]):
        """
        :param prompts: are the list of prompts to embed
        """
        # Tokenize the prompts
        batch_encoding = self.tokenizer(prompts, truncation=False, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        
        # Check if any tokenized input exceeds the maximum length
        assert not any(len(input_ids) > self.max_length for input_ids in batch_encoding['input_ids']), "Token length exceeds the maximum limit"

        # Get token ids
        tokens = batch_encoding["input_ids"].to(self.device)

        self.transformer = self.transformer.to(self.device)

        # Get CLIP embeddings
        return self.transformer(input_ids=tokens).last_hidden_state

    def compute_token_length(self, prompts: List[str]):
        # Tokenize the prompts
        batch_encoding = self.tokenizer(prompts, truncation=False, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")

        # Check if any tokenized input exceeds the maximum length
        input_ids = batch_encoding['input_ids']
        # Numer of elements in array
        num_tokens = input_ids.numel()

        return num_tokens

    def tokenize(self, prompts: List[str], max_token_length : int):
        # Tokenize the prompts
        batch_encoding = self.tokenizer(prompts, truncation=False, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")

        # Check if any tokenized input exceeds the maximum length
        input_ids = batch_encoding['input_ids']
        # Numer of elements in array
        num_tokens = input_ids.numel()

        assert num_tokens <= max_token_length, f"Token length {num_tokens} exceeds maximum {max_token_length}\nprompt : {prompts}"

        return batch_encoding

    def compute_embeddings(self, batch_encoding):
        # Get token ids and move to device
        tokens = batch_encoding["input_ids"].to(self.device)

        self.transformer = self.transformer.to(self.device)

        # Get CLIP embeddings
        clip_output = self.transformer(input_ids=tokens)

        return clip_output.last_hidden_state, clip_output.pooler_output, batch_encoding['attention_mask'].to(self.device)

    # NOTE(): deprecated
    # TODO(): Remove this
    def forward_return_all(self, prompts: List[str], max_token_length : int = MAX_LENGTH):
        """
        :param prompts: are the list of prompts to embed
        """
        # Tokenize the prompts
        batch_encoding = self.tokenizer(prompts, truncation=False, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")

        # Check if any tokenized input exceeds the maximum length
        input_ids =  batch_encoding['input_ids']
        # Numer of elements in array
        num_tokens = input_ids.numel()

        # Check if any tokenized input exceeds the maximum length
        assert not any(len(input_ids) > self.max_length for input_ids in batch_encoding['input_ids']), "Token length exceeds the maximum limit"

        # Get token ids and move to device
        tokens = batch_encoding["input_ids"].to(self.device)

        self.transformer = self.transformer.to(self.device)

        # Get CLIP embeddings
        clip_output = self.transformer(input_ids=tokens)
        
        return clip_output.last_hidden_state, clip_output.pooler_output, batch_encoding['attention_mask'].to(self.device)