import sys
import msgpack
import torch
from io import BytesIO
import os

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.clip.clip import ClipModel

class Phrase:
    def __init__(self, id, phrase):
        self.id = id
        self.phrase = phrase

class ClipVector:
    def __init__(self, phrase, clip_vector):
        self.phrase = phrase
        self.clip_vector = clip_vector

class ClipCache:
    def __init__(self, device, minio_client, clip_cache_directory):
        self.minio_client = minio_client
        self.clip_model = ClipModel(device=device)
        self.device = device
        self.clip_cache_directory = clip_cache_directory

    def create_cache_directory_if_not_exists(self):
        # Check if the directory exists
        if not os.path.exists(directory_path):
            # If not, create the directory
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created.")
        else:
            print(f"Directory '{directory_path}' already exists.")



