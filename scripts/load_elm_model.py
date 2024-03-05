import random
import os
import sys
base_directory = os.getcwd()
sys.path.insert(0, base_directory)
from utility.minio import cmd
from training_worker.classifiers.models.elm_regression import ELMRegression
from minio import Minio
import torch
import msgpack


# Define MinIO access
MINIO_ADDRESS = '123.176.98.90:9000'
ACCESS_KEY = '3lUCPCfLMgQoxrYaxgoz'
SECRET_KEY = 'MXszqU6KFV6X95Lo5jhMeuu5Xm85R79YImgI3Xmp'
MODEL_DATASET = 'environmental'  
TAG_NAME = 'concept-surreal'  
MODEL_TYPE = 'clip'
SCORING_MODEL = 'elm'
not_include = 'kandinsky'

def get_clip_vectors(minio_client, base_path, num_samples=10):
    objects_list = list(minio_client.list_objects(bucket_name='datasets', prefix=base_path, recursive=True))
    clip_objects = [obj for obj in objects_list if obj.object_name.endswith('_clip.msgpack')]

    # Randomly select num_samples clip vectors if available
    selected_clip_objects = random.sample(clip_objects, min(len(clip_objects), num_samples))
    clip_vectors = []

    for clip_obj in selected_clip_objects:
        obj_data = minio_client.get_object('datasets', clip_obj.object_name)
        obj_content = obj_data.read()
        # Deserialize the MessagePack data
        unpacked_data = msgpack.unpackb(obj_content, raw=False)
        # Extract the vector from the unpacked data
        vector = unpacked_data['clip-feature-vector'][0]  # Adjusted to directly access the vector
        clip_vectors.append(torch.tensor(vector))
    return clip_vectors

def get_unique_tag_names(minio_client, model_dataset):
    prefix = f"{model_dataset}/models/classifiers/"
    objects = minio_client.list_objects(bucket_name='datasets', prefix=prefix, recursive=True)
    tag_names = set()  # Use a set to avoid duplicates
    for obj in objects:
        parts = obj.object_name.split('/')
        if len(parts) > 3:  # Ensures that the path is deep enough to include a tag_name
            tag_name = parts[3]  # Assumes tag_name is the fourth element in the path
            tag_names.add(tag_name)
    print(tag_names)
    return list(tag_names)

    
# Initialize MinIO client
minio_client = cmd.connect_to_minio_client(MINIO_ADDRESS, access_key=ACCESS_KEY, secret_key=SECRET_KEY)

# Initialize your ELMRegression model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
elm_model = ELMRegression(device=device)

#for tag_name in get_unique_tag_names(minio_client, MODEL_DATASET):
    # Load the model for the specified tag
loaded_model = elm_model.load_model(minio_client, MODEL_DATASET, TAG_NAME, MODEL_TYPE, SCORING_MODEL, not_include, device=device)

if loaded_model:
    print(f"Model for tag '{TAG_NAME}' loaded successfully.")
    base_path = f"{MODEL_DATASET}/0004/"
    clip_vectors = get_clip_vectors(minio_client, base_path)
    
    for vector in clip_vectors:
        vector = vector.to(device).unsqueeze(0)  
        classification_score = loaded_model.classify(vector)
        print(f"Classification score for {TAG_NAME}: {classification_score}")
else:
     print(f"Failed to load the model for tag: {TAG_NAME}.")

