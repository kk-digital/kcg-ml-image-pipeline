import os
import sys
from typing import List
import torch
from torch import nn
from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection
)

sys.path.insert(0, os.getcwd())
from stable_diffusion.model_paths import CLIP_TEXT_EMBEDDER_PATH, CLIP_TOKENIZER_DIR_PATH, CLIP_TEXT_MODEL_DIR_PATH, CLIP_MODEL_PATH
from stable_diffusion.utils_backend import get_device
from utility.labml.monit import section
from utility.utils_logger import logger

class KandinskyCLIPTextEmbedder(nn.Module):
    """
    ## CLIP Text Embedder
    """

    def __init__(self, device=None, tokenizer=None, transformer=None):
        """
        """
        super().__init__()

        self.device = get_device(device)

        self.tokenizer = tokenizer
        self.transformer = transformer

        self.to(self.device)

    def load_submodels(self, tokenizer_path=CLIP_TOKENIZER_DIR_PATH, transformer_path=CLIP_TEXT_MODEL_DIR_PATH):

        with section("Loading tokenizer and transformer"):
            self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            logger.debug(f"Tokenizer successfully loaded from : {tokenizer_path}")
            self.transformer = CLIPTextModelWithProjection.from_pretrained(transformer_path, use_safetensors=True, local_files_only=True).eval().to(self.device)

            self.transformer = self.transformer.to(device=self.device)
            print(self.device)
            logger.debug(f"CLIP text model successfully loaded from : {transformer_path}")

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

    def forward(
        self,
        prompt,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
        negative_prompt=None
    ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        text_mask = text_inputs.attention_mask.bool().to(self.device)

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

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

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        text_encoder_hidden_states = text_encoder_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
        text_mask = text_mask.repeat_interleave(num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_text_mask = uncond_input.attention_mask.bool().to(self.device)
            negative_prompt_embeds_text_encoder_output = self.text_encoder(uncond_input.input_ids.to(self.device))

            negative_prompt_embeds = negative_prompt_embeds_text_encoder_output.text_embeds
            uncond_text_encoder_hidden_states = negative_prompt_embeds_text_encoder_output.last_hidden_state

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method

            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len)

            seq_len = uncond_text_encoder_hidden_states.shape[1]
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.repeat(1, num_images_per_prompt, 1)
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            uncond_text_mask = uncond_text_mask.repeat_interleave(num_images_per_prompt, dim=0)

            # done duplicates

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            text_encoder_hidden_states = torch.cat([uncond_text_encoder_hidden_states, text_encoder_hidden_states])

            text_mask = torch.cat([uncond_text_mask, text_mask])

        return prompt_embeds, text_encoder_hidden_states, text_mask

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
        input_ids = batch_encoding['input_ids']
        # Numer of elements in array
        num_tokens = input_ids.numel()

        assert num_tokens <= max_token_length, f"Token length {num_tokens} exceeds maximum {max_token_length}\nprompt : {prompts}"

        return batch_encoding
