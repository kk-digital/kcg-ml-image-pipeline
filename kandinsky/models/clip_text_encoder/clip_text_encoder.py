import os
import sys
from typing import List
import torch
from torch import nn
from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection
)

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from kandinsky.model_paths import PRIOR_MODEL_PATH
from stable_diffusion.utils_backend import get_device
from utility.labml.monit import section
from utility.utils_logger import logger

class KandinskyCLIPTextEmbedder(nn.Module):
    """
    ## CLIP Text Embedder
    """

    def __init__(self, device=None, tokenizer=None, text_encoder=None):
        """
        """
        super().__init__()

        self.device = get_device(device)

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

        self.to(self.device)

    def load_submodels(self, encoder_path=PRIOR_MODEL_PATH):

        with section("Loading tokenizer and transformer"):
            self.tokenizer = CLIPTokenizer.from_pretrained(encoder_path, subfolder="tokenizer", local_files_only=True)
            logger.debug(f"Tokenizer successfully loaded from : {encoder_path}/tokenizer")
            self.text_encoder = CLIPTextModelWithProjection.from_pretrained(encoder_path, subfolder="text_encoder", use_safetensors=True, local_files_only=True).eval().to(self.device)

            self.text_encoder = self.text_encoder.to(device=self.device)
            print(self.device)
            logger.debug(f"CLIP text model successfully loaded from : {encoder_path}/text_encoder")

            return self

    def unload_submodels(self):
        if self.tokenizer is not None:
            self.tokenizer
            del self.tokenizer
            self.tokenizer = None
        if self.text_encoder is not None:
            self.text_encoder.to("cpu")
            del self.text_encoder
            self.text_encoder = None
        torch.cuda.empty_cache()

    def forward(
        self,
        prompts,
        num_embeddings_per_prompt=1,
    ):
        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        text_mask = text_inputs.attention_mask.bool().to(self.device)

        untruncated_ids = self.tokenizer(prompts, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]

        text_encoder_output = self.text_encoder(text_input_ids.to(self.device))

        prompt_embeds = text_encoder_output.text_embeds
        text_encoder_hidden_states = text_encoder_output.last_hidden_state

        prompt_embeds = prompt_embeds.repeat_interleave(num_embeddings_per_prompt, dim=0)
        text_encoder_hidden_states = text_encoder_hidden_states.repeat_interleave(num_embeddings_per_prompt, dim=0)
        text_mask = text_mask.repeat_interleave(num_embeddings_per_prompt, dim=0)

        return text_encoder_hidden_states
    
    def compute_embeddings(self, 
                           prompts,
                           num_embeddings_per_prompt=1):
        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        text_mask = text_inputs.attention_mask.bool().to(self.device)

        untruncated_ids = self.tokenizer(prompts, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]

        text_encoder_output = self.text_encoder(text_input_ids.to(self.device))

        prompt_embeds = text_encoder_output.text_embeds
        text_encoder_hidden_states = text_encoder_output.last_hidden_state

        prompt_embeds = prompt_embeds.repeat_interleave(num_embeddings_per_prompt, dim=0)
        text_encoder_hidden_states = text_encoder_hidden_states.repeat_interleave(num_embeddings_per_prompt, dim=0)
        text_mask = text_mask.repeat_interleave(num_embeddings_per_prompt, dim=0)

        return text_encoder_hidden_states, prompt_embeds, text_mask

    def compute_token_length(self, prompts: List[str]):
        # Tokenize the prompts
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Check if any tokenized input exceeds the maximum length
        input_ids = text_inputs.input_ids
        # Numer of elements in array
        num_tokens = input_ids.numel()

        return num_tokens

    def tokenize(self, prompts: List[str], max_token_length : int):
        # Tokenize the prompts
        batch_encoding = self.tokenizer(prompts, truncation=False, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")

        # Check if any tokenized input exceeds the maximum length
        input_ids = batch_encoding.input_ids
        # Numer of elements in array
        num_tokens = input_ids.numel()

        assert num_tokens <= max_token_length, f"Token length {num_tokens} exceeds maximum {max_token_length}\nprompt : {prompts}"

        return batch_encoding
