from io import BytesIO
import io
import os
import sys
from minio import Minio
from PIL import Image
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
    sequential_ids = request.http_get_sequential_id(dataset, 1)
    file_path= f"{dataset}/{sequential_ids[0]+'.jpg'}"

    # upload the image
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='jpg')
    img_byte_arr = img_byte_arr.getvalue()
    cmd.upload_data(minio_client, "extracts", file_path, img_byte_arr)
    
    # upload latent
    save_latent_to_minio(minio_client, "extracts", image_uuid, image_hash, vae_latent, file_path)

    # upload clip vector
    clip_feature_dict = {"clip-feature-vector": clip_vector}
    clip_feature_msgpack = msgpack.packb(clip_feature_dict)

    clip_feature_msgpack_buffer = BytesIO()
    clip_feature_msgpack_buffer.write(clip_feature_msgpack)
    clip_feature_msgpack_buffer.seek(0)

    # Upload the clip vector data
    cmd.upload_data(minio_client, "extracts", file_path.replace('.jpg', '_vae_clip-h.msgpack'), clip_feature_msgpack_buffer)

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
     


