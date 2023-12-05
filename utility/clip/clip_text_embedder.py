import torch
import numpy as np
from typing import List


# pooling functions
def tensor_average_pooling(text_embedding):
    pooled_embedding = text_embedding.mean(dim=-2)
    return pooled_embedding

def tensor_max_pooling(text_embedding):
    pooled_embedding = text_embedding.max(dim=-2).values
    return pooled_embedding

def tensor_max_abs_pooling(text_embedding):
    embedding_abs = torch.abs(text_embedding)
    embedding_max_indices = torch.max(embedding_abs, dim=-2).indices
    pooled_embedding = text_embedding.gather(-2, embedding_max_indices.unsqueeze(-2)).squeeze(-2)
    return pooled_embedding

def tensor_attention_pooling(text_embedding, attention_mask):
    attention_mask = attention_mask.unsqueeze(-1)
    pooled_embedding = (text_embedding * attention_mask).sum(dim=-2) / attention_mask.sum(dim=-2)
    return pooled_embedding

def tensor_clip_pooling(text_embedding, attention_mask):
    eos_indices = attention_mask.sum(dim=-1) - 1
    eos_indices = eos_indices.unsqueeze(-1).repeat(*([1] * len(eos_indices.shape)), text_embedding.shape[-1]).unsqueeze(-2)
    pooled_embedding = text_embedding.gather(-2, index=eos_indices).squeeze(-2)
    return pooled_embedding

def ndarray_average_pooling(text_embedding):
    pooled_embedding = text_embedding.mean(axis=-2)
    return pooled_embedding

def ndarray_max_pooling(text_embedding):
    pooled_embedding = text_embedding.max(axis=-2)
    return pooled_embedding

def ndarray_max_abs_pooling(text_embedding):
    embedding_abs = np.abs(text_embedding)
    embedding_max_indices = np.argmax(embedding_abs, axis=-2)
    pooled_embedding = np.take_along_axis(text_embedding, embedding_max_indices[..., None, :], axis=-2)[..., 0, :]
    return pooled_embedding

def ndarray_attention_pooling(text_embedding, attention_mask):
    attention_mask = attention_mask[..., None]
    pooled_embedding = (text_embedding * attention_mask).sum(axis=-2) / attention_mask.sum(axis=-2)
    return pooled_embedding

def ndarray_clip_pooling(text_embedding, attention_mask):
    eos_indices = attention_mask.sum(axis=-1) - 1
    pooled_embedding = np.take_along_axis(text_embedding, eos_indices[..., None, None], axis=-2)[..., 0, :]
    return pooled_embedding

# pooler class

class CLIPTextPooler:

    @staticmethod
    def ndarray_pooling(text_embedding: np.ndarray, attention_mask: np.ndarray = None, pooling_strategy: str = 'AVERAGE_POOLING'):
        """
        Pooling given ndarray text embedding.

        Args:
        - text_embedding (np.ndarray): The input text embedding, either signle or batch, (*,77,768).
        - attention_mask (Optional[np.ndarray]): The attention mask of text embedding, (*,77).
        - pooling_strategy (Optional[str]): The name of pooling strategy, should be one of:
            - 'AVERAGE_POOLING'
            - 'MAX_POOLING'
            - 'MAX_ABS_POOLING'
            - 'ATTENTION_POOLING'
            - 'CLIP_POOLING'
            .

        Returns:
        - pooled_embedding (np.ndarray): pooling result of given text embedding using specific pooling strategy, (*,768).
        """
        if pooling_strategy == 'AVERAGE_POOLING':

            return ndarray_average_pooling(text_embedding)

        if pooling_strategy == 'MAX_POOLING':

            return ndarray_max_pooling(text_embedding)

        if pooling_strategy == 'MAX_ABS_POOLING':

            return ndarray_max_abs_pooling(text_embedding)

        if pooling_strategy == 'ATTENTION_POOLING':

            return ndarray_attention_pooling(text_embedding, attention_mask)

        if pooling_strategy == 'CLIP_POOLING':

            return ndarray_clip_pooling(text_embedding, attention_mask)
            
        raise(f'ERROR! Unknown pooling strateg: {pooling_strategy}')

    @staticmethod
    def tensor_pooling(text_embedding: torch.tensor, attention_mask: torch.tensor = None, pooling_strategy: str = 'AVERAGE_POOLING'):
        """
        Pooling given tensor text embedding.

        Args:
        - text_embedding (torch.tensor): The input text embedding, either signle or batch, (*,77,768).
        - attention_mask (Optional[torch.tensor]): The attention mask of text embedding, (*,77).
        - pooling_strategy (Optional[str]): The name of pooling strategy, should be one of:
            - 'AVERAGE_POOLING'
            - 'MAX_POOLING'
            - 'MAX_ABS_POOLING'
            - 'ATTENTION_POOLING'
            - 'CLIP_POOLING'
            .

        Returns:
        - pooled_embedding (torch.tensor): pooling result of given text embedding using specific pooling strategy, (*,768).
        """
        if pooling_strategy == 'AVERAGE_POOLING':

            return tensor_average_pooling(text_embedding)

        if pooling_strategy == 'MAX_POOLING':

            return tensor_max_pooling(text_embedding)

        if pooling_strategy == 'MAX_ABS_POOLING':

            return tensor_max_abs_pooling(text_embedding)

        if pooling_strategy == 'ATTENTION_POOLING':

            return tensor_attention_pooling(text_embedding, attention_mask)

        if pooling_strategy == 'CLIP_POOLING':

            return tensor_clip_pooling(text_embedding, attention_mask)
            
        raise(f'ERROR! Unknown pooling strateg: {pooling_strategy}')
