from io import BytesIO
import io
import os
import sys
from minio import Minio
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import msgpack
from torchvision.transforms.v2 import functional as VF
from torchvision.transforms.v2 import RandomResizedCrop

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())
from kandinsky.utils_image import save_latent_to_minio
from utility.minio import cmd
from utility.path import separate_bucket_and_file_path
from utility.http import request, external_images_request


EXTRACT_BUCKET= "extracts"

def extract_square_images(minio_client: Minio, 
                        external_image_data: list,
                        target_size: int = 512):
    
    print("Extracting images.........")
    # get file paths
    file_paths= [image['file_path'] for image in external_image_data]

    extracted_images=[]
    for path in tqdm(file_paths):
        # get image from minio server
        bucket_name, file_path = separate_bucket_and_file_path(path)
        try:
            response = minio_client.get_object(bucket_name, file_path)
            image_data = BytesIO(response.data)
            img = Image.open(image_data)
            img = img.convert("RGB")
        except Exception as e:
            raise e
        finally:
            response.close()
            response.release_conn()
        
        if img.size != (target_size, target_size):
            # get feature
            scale = min((target_size / min(img.size)) ** 2, .5)

            params = RandomResizedCrop.get_params(img, scale=(scale, 1), ratio=(1.,1.))
            img = VF.resized_crop(img, *params, size=target_size, interpolation=VF.InterpolationMode.BICUBIC, antialias=True)

            # Convert the resized image back to bytes
            image_data = BytesIO()
            img.save(image_data, format='JPEG')
            image_data.seek(0)  # Reset buffer position to the beginning
        
        extracted_images.append({
                "image": img,
                "image_data": image_data
        })
    
    return extracted_images

def upload_extract_data(minio_client: Minio, extract_data: dict):
    # get latent and clip vector
    image_hash= extract_data["image_hash"]
    image_uuid= extract_data["image_uuid"]
    image= extract_data["image"]
    clip_vector= extract_data["clip_vector"]
    vae_latent= extract_data["vae_latent"]
    source_image_hash= extract_data["source_image_hash"]
    source_image_uuid= extract_data["source_image_uuid"]
    dataset= extract_data["dataset"]

    # get image file path with sequential ids
    sequential_ids = request.http_get_sequential_id(f"{EXTRACT_BUCKET}_{dataset}", 1)
    file_path= f"{dataset}/{sequential_ids[0]+'.jpg'}"

    try:    
        # upload the image
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        cmd.upload_data(minio_client, EXTRACT_BUCKET, file_path, img_byte_arr)
        
        # upload latent
        save_latent_to_minio(minio_client, EXTRACT_BUCKET, image_uuid, image_hash, vae_latent, f"{EXTRACT_BUCKET}/{file_path}")

        # upload clip vector
        clip_vector = clip_vector.cpu().numpy().tolist()
        clip_feature_dict = {"clip-feature-vector": clip_vector}
        clip_feature_msgpack = msgpack.packb(clip_feature_dict)

        clip_feature_msgpack_buffer = BytesIO()
        clip_feature_msgpack_buffer.write(clip_feature_msgpack)
        clip_feature_msgpack_buffer.seek(0)

        cmd.upload_data(minio_client, EXTRACT_BUCKET, file_path.replace('.jpg', '_vae_clip-h.msgpack'), clip_feature_msgpack_buffer)

        # upload the image to mongoDB
        extract_data={
            "uuid": image_uuid,
            "image_hash": image_hash,
            "dataset": dataset,
            "file_path": file_path,
            "source_image_uuid": source_image_uuid,
            "source_image_hash": source_image_hash,
        }

        external_images_request.http_add_extract(extract_data)
        
    except Exception as e:
        print(e)


def save_latents_and_vectors(minio_client, dataset, clip_vectors, vae_latents, batch_size=10000):
    # Get the current batch information
    batch_info = external_images_request.http_get_current_extract_batch_sequential_id(dataset)
    print(batch_info)
    batch_num = batch_info["sequence_number"]
    is_complete = batch_info["complete"]

    # Determine the output folder based on batch number
    output_folder = f"latents/{str(batch_num).zfill(4)}"
    clip_vector_path = output_folder + "_clip-h.npy"
    vae_latent_path = output_folder + "_vae_latents.npy"

    # Convert tensors to numpy arrays
    clip_vectors_np = torch.stack(clip_vectors).cpu().numpy()
    vae_latents_np = torch.stack(vae_latents).cpu().numpy()

    if is_complete:
        # Current batch is complete, start a new batch
        batch_num = external_images_request.http_get_next_extract_batch_sequential_id(dataset, len(clip_vectors)==batch_size)
        output_folder = f"latents/{str(batch_num).zfill(4)}"
        # Save the new data directly as the start of a new batch
        save_batch_to_minio(minio_client, output_folder, clip_vectors_np, vae_latents_np)
    else:
        # Current batch is not complete, load existing data
        current_clip_batch = load_batch_from_minio(minio_client, EXTRACT_BUCKET, clip_vector_path) or np.array([])
        current_vae_batch = load_batch_from_minio(minio_client, EXTRACT_BUCKET, vae_latent_path) or np.array([])

        # Append new data
        updated_clip_batch = np.concatenate((current_clip_batch, clip_vectors_np))
        updated_vae_batch = np.concatenate((current_vae_batch, vae_latents_np))

        # Check if updated batch exceeds batch size
        if len(updated_clip_batch) > batch_size:
            # Save only up to batch size
            save_batch_to_minio(minio_client, output_folder, updated_clip_batch[:batch_size], updated_vae_batch[:batch_size])

            # Handle the overflow
            overflow_clip = updated_clip_batch[batch_size:]
            overflow_vae = updated_vae_batch[batch_size:]
            new_batch_num = external_images_request.http_get_next_extract_batch_sequential_id(dataset, False)
            new_output_folder = f"latents/{str(new_batch_num).zfill(4)}"
            save_batch_to_minio(minio_client, new_output_folder, overflow_clip, overflow_vae)
        else:
            # Save updated batch normally
            save_batch_to_minio(minio_client, output_folder, updated_clip_batch, updated_vae_batch)

    print(f"Saved CLIP vectors to {clip_vector_path}")
    print(f"Saved VAE latents to {vae_latent_path}")

def save_batch_to_minio(minio_client: Minio, output_folder: str, clip_vectors: np.array, vae_latents: np.array):
    clip_vector_buffer = BytesIO()
    np.save(clip_vector_buffer, clip_vectors)
    clip_vector_buffer.seek(0)  # Reset buffer position to the beginning

    vae_latent_buffer = BytesIO()
    np.save(vae_latent_buffer, vae_latents)
    vae_latent_buffer.seek(0)  # Reset buffer position to the beginning

    # Save to numpy files
    clip_vector_path= output_folder + "_clip-h.npy"
    vae_latent_path= output_folder + "_vae_latents.npy"
    cmd.upload_data(minio_client, EXTRACT_BUCKET, clip_vector_path, clip_vector_buffer)
    cmd.upload_data(minio_client, EXTRACT_BUCKET, vae_latent_path, vae_latent_buffer)

def load_batch_from_minio(minio_client: Minio, bucket_name: str, file_path: str):
    try:
        # Get the object from MinIO
        response = minio_client.get_object(bucket_name, file_path)
        
        # Read data into bytes buffer
        data = BytesIO(response.read())
        
        # Load array using NumPy
        array = list(np.load(data))
        
        return array
    except Exception as e:
        print("Failed to retrieve or parse the file:", e)
        return None
    

     



