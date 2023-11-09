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

from utility.path import separate_bucket_and_file_path
from utility.minio import cmd
from worker.image_generation.generation_data.generated_image_data import GeneratedImageData
from worker.image_generation.generation_data.prompt_embedding import PromptEmbedding


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

    cmd.upload_data(minio_client, bucket_name, file_path, buffer)

def save_image_embedding_to_minio(minio_client, job_uuid, creation_time, dataset, file_path, file_hash, positive_prompt,
                                              negative_prompt, positive_embedding, negative_embedding):

    bucket_name, file_path = separate_bucket_and_file_path(file_path)

    generated_image_data = PromptEmbedding(job_uuid, creation_time, dataset, file_path, file_hash, positive_prompt,
                                           negative_prompt, positive_embedding, negative_embedding)


    msgpack_string = generated_image_data.get_msgpack_string()

    buffer = io.BytesIO()
    buffer.write(msgpack_string)
    buffer.seek(0)

    cmd.upload_data(minio_client, bucket_name, file_path, buffer)


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

# commented the new hash function for now

# def get_image_data(images: torch.Tensor, img_format: str = 'jpeg'):
#     # Map images to `[0, 1]` space and clip
#     images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
#     # Transpose to `[batch_size, height, width, channels]` and convert to numpy
#     images = images.cpu()
#     images = images.permute(0, 2, 3, 1)
#     images = images.detach().float().numpy()

#     img_hashes = []
#     img_byte_arrs = []

#     # Save images and calculate hashes
#     for img in images:
#         img = Image.fromarray((255. * img).astype(np.uint8))
#         # Convert to bytes array
#         img_byte_arr = io.BytesIO()
#         img.save(img_byte_arr, format=img_format)
#         img_byte_arr.seek(0)

#         # Calculate hash
#         output_file_hash = (hashlib.sha256(img_byte_arr.getbuffer())).hexdigest()

#         img_hashes.append(output_file_hash)
#         img_byte_arrs.append(img_byte_arr)

#     return img_hashes, img_byte_arrs


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
