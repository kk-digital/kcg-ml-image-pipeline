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

