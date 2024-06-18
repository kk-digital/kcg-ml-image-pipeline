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
    
def save_latents_and_vectors(minio_client, dataset, clip_vectors, vae_latents, image_hashes, batch_size=10000):
    batch_info = external_images_request.http_get_current_extract_batch_sequential_id(dataset)
    batch_num = batch_info["sequence_number"]
    is_complete = batch_info["complete"]

    # Convert tensors to numpy arrays
    clip_vectors_np = [vec.cpu().numpy() for vec in clip_vectors]
    vae_latents_np = [vec.cpu().numpy() for vec in vae_latents]

    # Prepare data for saving
    combined_data = [
        {"image_hash": img_hash, "clip_vector": clip_vec, "vae_latent": vae_lat}
        for img_hash, clip_vec, vae_lat in zip(image_hashes, clip_vectors_np, vae_latents_np)
    ]

    # Determine the output folder based on batch number
    output_folder = f"latents/{str(batch_num).zfill(4)}"
    data_path = output_folder + "_data.msgpack"

    if is_complete:
        # Current batch is complete, start a new batch
        batch_info = external_images_request.http_get_next_extract_batch_sequential_id(dataset, len(clip_vectors)==batch_size)
        batch_num = batch_info["sequence_number"]
        output_folder = f"latents/{str(batch_num).zfill(4)}"
        data_path = output_folder + "_data.msgpack"
        # Save the new data directly as the start of a new batch
        save_data_to_minio(minio_client, data_path, combined_data)
    else:
        # Current batch is not complete, load existing data, append, and save
        existing_data = load_data_from_minio(minio_client, EXTRACT_BUCKET, data_path) or []
        updated_data = existing_data + combined_data

        # Check if updated data exceeds batch size
        if len(updated_data) > batch_size:
            # Split and save the full batch, then the overflow
            save_data_to_minio(minio_client, data_path, updated_data[:batch_size])
            overflow_data = updated_data[batch_size:]
            new_batch_info = external_images_request.http_get_next_extract_batch_sequential_id(dataset, False)
            new_batch_num = new_batch_info["sequence_number"]
            new_output_folder = f"latents/{str(new_batch_num).zfill(4)}"
            new_data_path = new_output_folder + "_latent_data.msgpack"
            save_data_to_minio(minio_client, new_data_path, overflow_data)
        else:
            # Save updated data batch
            save_data_to_minio(minio_client, data_path, updated_data)

    print(f"Data saved in {data_path}")

def save_data_to_minio(minio_client, data_path, data):
    data_buffer = BytesIO()
    packed_data = msgpack.packb(data, use_bin_type=True)
    data_buffer.write(packed_data)
    data_buffer.seek(0)
    minio_client.put_object(EXTRACT_BUCKET, data_path, data_buffer, len(data_buffer.getvalue()))

def load_data_from_minio(minio_client, bucket_name, file_path):
    try:
        response = minio_client.get_object(bucket_name, file_path)
        data = BytesIO(response.read())
        return msgpack.unpackb(data.getvalue(), raw=False)
    except Exception as e:
        print(f"Failed to retrieve or parse the file: {e}")
        return []
    

     



