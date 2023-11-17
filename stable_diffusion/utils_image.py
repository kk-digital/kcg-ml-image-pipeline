import hashlib
from pathlib import Path
from typing import Union, BinaryIO, List, Optional
import io
import PIL
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms import ToPILImage
import os
from utility.path import separate_bucket_and_file_path
from utility.minio import cmd
from data_loader.generated_image_data import GeneratedImageData
from data_loader.prompt_embedding import PromptEmbedding
from utility.clip.clip_text_embedder import tensor_attention_pooling, tensor_max_pooling, tensor_max_abs_pooling


def calculate_sha256(tensor):
    if tensor.device == "cpu":
        tensor_bytes = tensor.numpy().tobytes()  # Convert tensor to a byte array
    else:
        tensor_bytes = tensor.cpu().numpy().tobytes()  # Convert tensor to a byte array
    sha256_hash = hashlib.sha256(tensor_bytes)
    return sha256_hash.hexdigest()


def to_pil(image):
    return ToPILImage()(torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0))


def save_images(images: torch.Tensor, dest_path: str, img_format: str = 'jpeg'):
    """
    ### Save a images

    :param images: is the tensor with images of shape `[batch_size, channels, height, width]`
    :param dest_path: is the folder to save images in
    :param img_format: is the image format
    """

    # Map images to `[0, 1]` space and clip
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    # Transpose to `[batch_size, height, width, channels]` and convert to numpy
    images = images.cpu()
    images = images.permute(0, 2, 3, 1)
    images = images.detach().float().numpy()

    image_hash_list = []
    image_list = []
    # Save images
    for i, img in enumerate(images):
        img = Image.fromarray((255. * img).astype(np.uint8))
        img.save(dest_path, format=img_format)
        image_data = img.tobytes()
        image_hash = (hashlib.sha256(image_data)).hexdigest()
        image_hash_list.append(image_hash)
        image_list.append(img)

    return (image_list, image_hash_list)


def save_image_data_to_minio(minio_client, job_uuid, creation_time, dataset, file_path, file_hash, positive_prompt,
                             negative_prompt,
                             cfg_strength, seed, image_width, image_height, sampler, sampler_steps,
                             prompt_scoring_model, prompt_score, prompt_generation_policy, top_k):
    bucket_name, file_path = separate_bucket_and_file_path(file_path)

    generated_image_data = GeneratedImageData(job_uuid, creation_time, dataset, file_path, file_hash, positive_prompt,
                                              negative_prompt,
                                              cfg_strength, seed, image_width, image_height, sampler, sampler_steps,
                                              prompt_scoring_model, prompt_score, prompt_generation_policy, top_k)


    msgpack_string = generated_image_data.get_msgpack_string()

    buffer = io.BytesIO()
    buffer.write(msgpack_string)
    buffer.seek(0)

    cmd.upload_data(minio_client, bucket_name, file_path.replace('.jpg', '_data.msgpack'), buffer)

def save_prompt_embedding_to_minio(minio_client, prompt_embedding, file_path):
    msgpack_string = prompt_embedding.get_msgpack_string()

    buffer = io.BytesIO()
    buffer.write(msgpack_string)
    buffer.seek(0)

    cmd.upload_data(minio_client, "datasets", file_path, buffer)

def get_embeddings(job_uuid,
                  creation_time,
                  dataset,
                  file_path,
                  file_hash,
                  positive_prompt,
                  negative_prompt,
                  text_embedder):
    # calculate new embeddings
    positive_embedding, _, positive_attention_mask = text_embedder.forward_return_all(positive_prompt)
    negative_embedding, _, negative_attention_mask = text_embedder.forward_return_all(negative_prompt)

    positive_attention_mask_detached = positive_attention_mask.detach().cpu().numpy()
    negative_attention_mask_detached = negative_attention_mask.detach().cpu().numpy()
    prompt_embedding = PromptEmbedding(job_uuid,
                                       creation_time,
                                       dataset,
                                       file_path,
                                       file_hash,
                                       positive_prompt,
                                       negative_prompt,
                                       positive_embedding.detach().cpu().numpy(),
                                       negative_embedding.detach().cpu().numpy(),
                                       positive_attention_mask_detached,
                                       negative_attention_mask_detached)

    # average
    positive_average_pooled = tensor_attention_pooling(positive_embedding, positive_attention_mask)
    negative_average_pooled = tensor_attention_pooling(negative_embedding, negative_attention_mask)

    prompt_embedding_average_pooled = PromptEmbedding(job_uuid,
                                       creation_time,
                                       dataset,
                                       file_path,
                                       file_hash,
                                       positive_prompt,
                                       negative_prompt,
                                       positive_average_pooled.detach().cpu().numpy(),
                                       negative_average_pooled.detach().cpu().numpy(),
                                       positive_attention_mask_detached,
                                       negative_attention_mask_detached)

    # max
    positive_max_pooled = tensor_max_pooling(positive_embedding)
    negative_max_pooled = tensor_max_pooling(negative_embedding)
    prompt_embedding_max_pooled = PromptEmbedding(job_uuid,
                                                  creation_time,
                                                  dataset,
                                                  file_path,
                                                  file_hash,
                                                  positive_prompt,
                                                  negative_prompt,
                                                  positive_max_pooled.detach().cpu().numpy(),
                                                  negative_max_pooled.detach().cpu().numpy(),
                                                  positive_attention_mask_detached,
                                                  negative_attention_mask_detached)

    # signed max
    positive_signed_max_pooled = tensor_max_abs_pooling(positive_embedding)
    negative_signed_max_pooled = tensor_max_abs_pooling(negative_embedding)
    prompt_embedding_signed_max_pooled = PromptEmbedding(job_uuid,
                                                          creation_time,
                                                          dataset,
                                                          file_path,
                                                          file_hash,
                                                          positive_prompt,
                                                          negative_prompt,
                                                          positive_signed_max_pooled.detach().cpu().numpy(),
                                                          negative_signed_max_pooled.detach().cpu().numpy(),
                                                          positive_attention_mask_detached,
                                                          negative_attention_mask_detached)

    return prompt_embedding, prompt_embedding_average_pooled, prompt_embedding_max_pooled, prompt_embedding_signed_max_pooled


def save_image_embedding_to_minio(minio_client,
                                  dataset,
                                  file_path,
                                  prompt_embedding,
                                  prompt_embedding_average_pooled,
                                  prompt_embedding_max_pooled,
                                  prompt_embedding_signed_max_pooled):
    bucket_name, file_path = separate_bucket_and_file_path(file_path)

    # get filename
    filename = os.path.split(file_path)[-1]
    filename = filename.replace(".jpg", "")
    # get parent dir
    parent_dir = os.path.dirname(file_path)
    parent_dir = os.path.split(parent_dir)[-1]

    text_embedding_path_1 = file_path.replace(".jpg", "_embedding.msgpack")
    text_embedding_path_2 = os.path.join(dataset, "embeddings/text-embedding", parent_dir,
                                       filename + "-text-embedding.msgpack")
    text_embedding_average_pooled_path = os.path.join(dataset, "embeddings/text-embedding", parent_dir,
                                                      filename + "-text-embedding-average-pooled.msgpack")
    text_embedding_max_pooled_path = os.path.join(dataset, "embeddings/text-embedding", parent_dir,
                                                  filename + "-text-embedding-max-pooled.msgpack")
    text_embedding_signed_max_pooled_path = os.path.join(dataset, "embeddings/text-embedding", parent_dir,
                                                         filename + "-text-embedding-signed-max-pooled.msgpack")


    # save normal embedding 77*768
    save_prompt_embedding_to_minio(minio_client, prompt_embedding, text_embedding_path_1)
    save_prompt_embedding_to_minio(minio_client, prompt_embedding, text_embedding_path_2)

    # average
    save_prompt_embedding_to_minio(minio_client, prompt_embedding_average_pooled, text_embedding_average_pooled_path)

    # max
    save_prompt_embedding_to_minio(minio_client, prompt_embedding_max_pooled, text_embedding_max_pooled_path)

    # signed max
    save_prompt_embedding_to_minio(minio_client, prompt_embedding_signed_max_pooled, text_embedding_signed_max_pooled_path)


def save_images_to_minio(minio_client, images: torch.Tensor, dest_path: str, img_format: str = 'jpeg'):
    """
    ### Save images to minio

    :param images: is the tensor with images of shape `[batch_size, channels, height, width]`
    :param dest_path: is the folder to save images in
    :param img_format: is the image format
    """

    # Map images to `[0, 1]` space and clip
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    # Transpose to `[batch_size, height, width, channels]` and convert to numpy
    images = images.cpu()
    images = images.permute(0, 2, 3, 1)
    images = images.detach().float().numpy()

    # Save images
    for i, img in enumerate(images):
        img = Image.fromarray((255. * img).astype(np.uint8))
        # convert to bytes arr
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img_format)
        img_byte_arr.seek(0)

        # get hash
        image_data = img.tobytes()
        output_file_hash = (hashlib.sha256(image_data)).hexdigest()

        # save to minio server
        output_file_path = dest_path
        bucket_name, file_path = separate_bucket_and_file_path(output_file_path)
        cmd.upload_data(minio_client, bucket_name, file_path, img_byte_arr)

    return output_file_hash


def get_image_hash(img_byte_arr):
    # Calculate the hash for the given image byte array
    return (hashlib.sha256(img_byte_arr)).hexdigest()
    

def get_image_data(images: torch.Tensor, img_format: str = 'jpeg'):
    # Map images to `[0, 1]` space and clip
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    # Transpose to `[batch_size, height, width, channels]` and convert to numpy
    images = images.cpu()
    images = images.permute(0, 2, 3, 1)
    images = images.detach().float().numpy()

    # Save images
    for i, img in enumerate(images):
        img = Image.fromarray((255. * img).astype(np.uint8))
        # convert to bytes arr
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img_format)
        img_byte_arr.seek(0)

        # get hash
        output_file_hash = (hashlib.sha256(img_byte_arr.getbuffer())).hexdigest()

    return output_file_hash, img_byte_arr


def save_image_grid(
        tensor: Union[torch.Tensor, List[torch.Tensor]],
        fp: Union[str, Path, BinaryIO],
        format: Optional[str] = None,
        **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = torchvision.utils.make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)


def show_image_grid(
        tensor: Union[torch.Tensor, List[torch.Tensor]],
        **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = torchvision.utils.make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


def load_img(path: str):
    """
    ### Load an image

    This loads an image from a file and returns a PyTorch tensor.

    :param path: is the path of the image
    """
    # Open Image
    image = Image.open(path).convert("RGB")
    # Get image size
    w, h = image.size
    # Resize to a multiple of 32
    w = w - w % 32
    h = h - h % 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    # Convert to numpy and map to `[-1, 1]` for `[0, 255]`
    image = np.array(image).astype(np.float32) * (2. / 255.0) - 1
    # Transpose to shape `[batch_size, channels, height, width]`
    image = image[None].transpose(0, 3, 1, 2)
    # Convert to torch
    return torch.from_numpy(image)
