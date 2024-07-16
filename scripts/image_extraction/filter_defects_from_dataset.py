import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import os
import sys
import torch
import msgpack
from tqdm import tqdm

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())
from utility.http import request
from utility.minio import cmd
from utility.path import separate_bucket_and_file_path
from training_worker.classifiers.models.elm_regression import ELMRegression
from utility.http.external_images_request import http_delete_extract, http_get_extract_image_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minio-addr', help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', help='Minio access key')
    parser.add_argument('--minio-secret-key', help='Minio secret key')
    parser.add_argument('--dataset', type=str, help='Dataset to extract from')
    parser.add_argument('--defect-tag', type=str, help='name of the defect tag to filter for')
    parser.add_argument('--defect-threshold', type=float, default=0.7, help='Minimum defect threshold')

    return parser.parse_args()

def delete_extract(minio_client, hash, file_path):
    # delete in mongodb
    http_delete_extract(hash)

    try:
        # delete files associated to the image
        bucket_name, image_path = separate_bucket_and_file_path(file_path)
        filename = os.path.splitext(image_path)[0]
        clip_path= filename + "_clip-h.msgpack"
        latent_path= filename + "_vae_latent.msgpack"
        # delete the image
        cmd.remove_an_object(minio_client, bucket_name, image_path)
        # delete vae and clip latents
        cmd.remove_an_object(minio_client, bucket_name, clip_path)
        cmd.remove_an_object(minio_client, bucket_name, latent_path)
    except Exception as e:
        print(f"This error {e} occured while deleting image file for {file_path}")

def load_clip_vector(minio_client, file_path, device):
    bucket_name, input_file_path = separate_bucket_and_file_path(file_path)
    file_path = os.path.splitext(input_file_path)[0]
    
    output_clip_path = file_path + "_clip-h.msgpack"
    features_data = cmd.get_file_from_minio(minio_client, bucket_name, output_clip_path)
    features_vector = msgpack.unpackb(features_data.data)["clip-feature-vector"]
    clip_vector= torch.tensor(features_vector).to(device)
     
    return clip_vector

def get_classifier_model(minio_client, tag_name, device):
    input_path = f"environmental/models/classifiers/{tag_name}/"
    file_suffix = "elm-regression-clip-h.pth"

    # Use the MinIO client's list_objects method directly with recursive=True
    model_files = [obj.object_name for obj in minio_client.list_objects('datasets', prefix=input_path, recursive=True) if obj.object_name.endswith(file_suffix)]
    
    if not model_files:
        print(f"No .safetensors models found for tag: {tag_name}")
        return None

    # Assuming there's only one model per tag or choosing the first one
    model_files.sort(reverse=True)
    model_file = model_files[0]
    print(f"Loading model: {model_file}")
    
    model_data = minio_client.get_object('datasets', model_file)
    
    clip_model = ELMRegression(device=device)
    
    # Create a BytesIO object from the model data
    byte_buffer = io.BytesIO(model_data.data)
    clip_model.load_safetensors(byte_buffer)

    print(f"Model loaded for tag: {tag_name}")
    
    return clip_model

def get_untagged_images(dataset: str, tag_name: str):
    print("Loading paths for untagged images")
    # get all images from the dataset
    image_data = http_get_extract_image_list(dataset= dataset)

    # get images tagged with the defect
    tag_list = request.http_get_tag_list()
    tag_id= None
    for tag in tag_list:
        if tag['tag_string']==tag_name:
            tag_id= tag["tag_id"]
    
    if tag_id is None:
        raise Exception(f"there is no tag with the name {tag_name}")

    tagged_images = request.http_get_tagged_extracts(tag_id)
    tagged_file_paths= [image['file_path'] for image in tagged_images]

    untagged_images= [image for image in image_data if image['file_path'] not in tagged_file_paths]

    file_paths= [image['file_path'] for image in untagged_images]
    image_hashes= [image['image_hash'] for image in untagged_images]

    return file_paths, image_hashes

def filter_defects(minio_client, dataset, defect_tag, defect_threshold, device):
    # load classifier model
    classifier_model = get_classifier_model(minio_client, defect_tag, device)

    # get all images except those tagged with the defect tag
    file_paths, image_hashes = get_untagged_images(dataset, defect_tag)

    # filter images
    images_to_delete=0
    delete_tasks = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        for file_path, hash in tqdm(zip(file_paths, image_hashes), total=len(file_paths)):
            clip_vector = load_clip_vector(minio_client, file_path, device)

            # Calculate the score
            score = classifier_model.classify(clip_vector)

            # Check if the defect is detected in the image
            if score > defect_threshold:
                # Schedule the deletion of the image
                # delete_tasks.append(executor.submit(delete_extract, minio_client, hash, file_path))
                images_to_delete += 1

    # Ensure all delete tasks are completed
    for task in delete_tasks:
        task.result()  # This will re-raise any exceptions caught during deletion

    print("images to delete: ", images_to_delete)

def main():
    args= parse_args()

    # get parameters
    dataset = args.dataset
    defect_tag= args.defect_tag
    defect_threshold= args.defect_threshold

    # get minio client
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                            minio_secret_key=args.minio_secret_key)
    
    # get device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    device = torch.device(device)

    # filter defects
    filter_defects(minio_client, dataset, defect_tag, defect_threshold, device)

if __name__ == "__main__":
    main()
