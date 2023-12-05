import sys
import msgpack
import torch
from io import BytesIO
import os
import json
import msgpack

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.clip.clip import ClipModel
from clip_utils import get_image_clip_from_minio
from clip_constants import BUCKET_NAME

class ClipFile:
    def __init__(self, clip_vector_max_count, clip_vector_list):
        self.clip_vector_count = clip_vector_max_count
        self.clip_vector_list = clip_vector_list

    def to_json(self):
        # Serialize object to JSON string
        return json.dumps(self.__dict__)

    def to_mspack(self):
        # Serialize JSON string to mspack binary format
        json_data = self.to_json()
        packed_data = msgpack.packb(json_data)
        return packed_data

    @classmethod
    def from_mspack(cls, data):
        # Deserialize mspack binary data to JSON string
        json_data = msgpack.unpackb(data, encoding='utf-8')

        # Deserialize JSON string to object
        json_dict = json.loads(json_data)
        return cls(**json_dict)

    def save_to_mspack(self, file_path):
        try:
            with open(file_path, 'wb') as file:
                file.write(self.to_mspack())
            print(f"ClipFile saved to {file_path}")
        except Exception as e:
            print(f"An error occurred while saving: {str(e)}")

    @classmethod
    def load_from_mspack(cls, file_path):
        try:
            with open(file_path, 'rb') as file:
                data = file.read()
            return cls.from_mspack(data)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        except Exception as e:
            print(f"An error occurred while loading: {str(e)}")
            return None


class ClipCache:
    def __init__(self, device, minio_client, clip_cache_directory):
        self.minio_client = minio_client
        self.clip_model = ClipModel(device=device)
        self.device = device
        self.clip_cache_directory = clip_cache_directory
        self.clip_vector_dictionary = {}

    # the cache has two levels
    # the first one is the memory
    # second one is the hard drive
    def cache_clip_vector(self, image_path, clip_vector):
        # TODO(): implement ssh cache
        # TODO(): memory mapped clip vector caching
        if image_path not in self.clip_vector_dictionary:
            self.clip_cache_directory[image_path] = clip_vector

    def get_clip_vector(self, image_path):
        # if its already in the cache just return it
        if image_path in self.clip_vector_dictionary:
            return self.clip_vector_dictionary[image_path]

        # if its not in the cache
        # get the clip vector from minio
        # and store it in the clip cache
        image_clip_vector_numpy = get_image_clip_from_minio(self.minio_client, image_path, BUCKET_NAME)

        if image_clip_vector_numpy is None:
            # could not find clip vector for image
            print(f'image clip {image_path} not found')
            return None

        # the image clip vector was loaded correctly
        self.cache_clip_vector(image_path, image_clip_vector_numpy)

    def create_cache_directory_if_not_exists(self):
        clip_cache_directory = self.clip_cache_directory

        # Check if the directory exists
        if not os.path.exists(clip_cache_directory):
            # If not, create the directory
            os.makedirs(clip_cache_directory)
            print(f"Directory '{clip_cache_directory}' created.")

    def list_cache_directory_files(self):

        clip_cache_directory = self.clip_cache_directory
        file_list = []

        try:
            # Get a list of all files in the directory
            files = os.listdir(clip_cache_directory)

            for file in files:
                file_list.append(file)

        except FileNotFoundError:
            print(f"Directory '{clip_cache_directory}' not found.")
        except PermissionError:
            print(f"Permission denied for directory '{clip_cache_directory}'.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

        return file_list

