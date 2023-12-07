import sys
import msgpack
import numpy as np
import torch
from io import BytesIO
import math

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.clip.clip import ClipModel
from utility.minio.cmd import get_file_from_minio, is_object_exists
from clip_cache import ClipCache
from clip_constants import CLIP_CACHE_DIRECTORY


class Phrase:
    def __init__(self, id, phrase):
        self.id = id
        self.phrase = phrase

class ClipVector:
    def __init__(self, phrase, clip_vector):
        self.phrase = phrase
        self.clip_vector = clip_vector

class ClipServer:
    def __init__(self, device, minio_client):
        self.minio_client = minio_client
        self.id_counter = 0
        self.phrase_dictionary = {}
        self.clip_vector_dictionary = {}
        self.image_clip_vector_cache = {}
        self.clip_model = ClipModel(device=device)
        self.device = device
        self.clip_cache = ClipCache(device, minio_client, CLIP_CACHE_DIRECTORY)

    def load_clip_model(self):
        self.clip_model.load_clip()
        self.clip_model.load_tokenizer()

    def generate_id(self):
        new_id = self.id_counter
        self.id_counter = self.id_counter + 1

        return new_id

    def add_phrase(self, phrase):
        new_id = self.generate_id()
        clip_vector = self.compute_clip_vector(phrase)

        new_phrase = Phrase(new_id, phrase)
        new_clip_vector = ClipVector(phrase, clip_vector)

        self.phrase_dictionary[new_id] = new_phrase
        self.clip_vector_dictionary[phrase] = new_clip_vector

        return new_phrase

    def get_clip_vector(self, phrase):
        if phrase in self.clip_vector_dictionary:
            return self.clip_vector_dictionary[phrase]

        return None

    def get_image_clip_vector(self, image_path):
        return self.clip_cache.get_clip_vector(image_path)

    def get_phrase_list(self, offset, limit):
        result = []
        count = 0
        for key, value in self.phrase_dictionary.items():
            if count >= offset:
                if count < offset + limit:
                    result.append(value)
                else:
                    break
            count += 1
        return result


    def get_image_clip_from_minio(self, image_path, bucket_name):

        # if its in the cache return from cache
        if image_path in self.image_clip_vector_cache:
            clip_vector = self.image_clip_vector_cache[image_path]
            return clip_vector

        # Removes the last 4 characters from the path
        # image.jpg => image
        base_path = image_path.rstrip(image_path[-4:])

        # finds the clip file associated with the image
        # example image => image_clip.msgpack
        image_clip_vector_path = f'{base_path}_clip.msgpack'

        print(f'image clip vector path : {image_clip_vector_path}')
        # get the clip.msgpack from minio
        file_exists = is_object_exists(self.minio_client, bucket_name, image_clip_vector_path)

        if not file_exists:
            print(f'{image_clip_vector_path} does not exist')
            return None

        clip_vector_data_msgpack = get_file_from_minio(self.minio_client, bucket_name, image_clip_vector_path)

        if clip_vector_data_msgpack is None:
            print(f'image not found {image_path}')
            return None

        # read file_data_into memory
        clip_vector_data_msgpack_memory = clip_vector_data_msgpack.read()

        try:
            # uncompress the msgpack data
            clip_vector = msgpack.unpackb(clip_vector_data_msgpack_memory)
            clip_vector = clip_vector["clip-feature-vector"]
            # add to chache
            self.image_clip_vector_cache[image_path] = clip_vector

            return clip_vector
        except Exception as e:
            print('Exception details : ', e)

        return None


    def compute_cosine_match_value(self, phrase, image_path):
        print('computing cosine match value for ', phrase, ' and ', image_path)

        phrase_cip_vector_struct = self.get_clip_vector(phrase)
        # the score is zero if we cant find the phrase clip vector
        if phrase_cip_vector_struct is None:
            print(f'phrase {phrase} not found ')
            return 0

        phrase_clip_vector_numpy = phrase_cip_vector_struct.clip_vector

        image_clip_vector_numpy = self.get_image_clip_vector(image_path)

        # the score is zero if we cant find the image clip vector
        if image_clip_vector_numpy is None:
            print(f'image clip {image_path} not found')
            return 0

        # convert numpy array to tensors
        phrase_clip_vector = torch.tensor(phrase_clip_vector_numpy, dtype=torch.float32, device=self.device)
        image_clip_vector = torch.tensor(image_clip_vector_numpy, dtype=torch.float32, device=self.device)

        # removing the extra dimension
        # from shape (1, 768) => (768)
        phrase_clip_vector = phrase_clip_vector.squeeze(0)
        image_clip_vector = image_clip_vector.squeeze(0)

        # Normalizing the tensor
        normalized_phrase_clip_vector = torch.nn.functional.normalize(phrase_clip_vector.unsqueeze(0), p=2, dim=1)
        normalized_image_clip_vector = torch.nn.functional.normalize(image_clip_vector.unsqueeze(0), p=2, dim=1)

        # removing the extra dimension
        # from shape (1, 768) => (768)
        normalized_phrase_clip_vector = normalized_phrase_clip_vector.squeeze(0)
        normalized_image_clip_vector = normalized_image_clip_vector.squeeze(0)

        # cosine similarity
        similarity = torch.dot(normalized_phrase_clip_vector, normalized_image_clip_vector)

        # cleanup
        del phrase_clip_vector
        del image_clip_vector
        del normalized_phrase_clip_vector
        del normalized_image_clip_vector

        return similarity.item()

    def compute_cosine_match_value_list(self, phrase, image_path_list):
        num_images = len(image_path_list)

        cosine_match_list = [0] * num_images

        phrase_cip_vector_struct = self.get_clip_vector(phrase)
        # the score is zero if we cant find the phrase clip vector
        if phrase_cip_vector_struct is None:
            print(f'phrase {phrase} not found ')
            return cosine_match_list

        phrase_clip_vector_numpy = phrase_cip_vector_struct.clip_vector

        # convert numpy array to tensors
        phrase_clip_vector = torch.tensor(phrase_clip_vector_numpy, dtype=torch.float32, device=self.device)
        # Normalizing the tensor
        normalized_phrase_clip_vector = torch.nn.functional.normalize(phrase_clip_vector, p=2, dim=1)

        # removing the extra dimension
        # from shape (1, 768) => (768)
        normalized_phrase_clip_vector = normalized_phrase_clip_vector.squeeze(0)

        # for each batch do
        for image_index in range(0, num_images):
            image_path = image_path_list[image_index]
            image_clip_vector = self.get_image_clip_vector(image_path)
            # if the clip_vector was not found
            # or couldn't load for some network reason
            # we must provide an empty vector as replacement
            if image_clip_vector is None:
                # this syntax is weird but its just list full of zeros
                image_clip_vector = [0] * 768

            # now that we have the clip vectors we need to construct our tensors
            image_clip_vector = torch.tensor(image_clip_vector, dtype=torch.float32, device=self.device)
            normalized_image_clip_vector = torch.nn.functional.normalize(image_clip_vector, p=2, dim=1)
            # removing the extra dimension
            # from shape (1, 768) => (768)
            normalized_image_clip_vector = normalized_image_clip_vector.squeeze(0)

            # cosine similarity
            similarity = torch.dot(normalized_phrase_clip_vector, normalized_image_clip_vector)
            similarity_value = similarity.item()
            cosine_match_list[image_index] = similarity_value

            # cleanup
            del image_clip_vector
            del normalized_image_clip_vector
            del similarity
            # After your GPU-related operations, clean up the GPU memory
            torch.cuda.empty_cache()

        del phrase_clip_vector
        del normalized_phrase_clip_vector
        # After your GPU-related operations, clean up the GPU memory
        torch.cuda.empty_cache()

        return cosine_match_list


    def compute_clip_vector(self, text):
        clip_vector_gpu = self.clip_model.get_text_features(text)
        clip_vector_cpu = clip_vector_gpu.cpu()

        del clip_vector_gpu

        clip_vector = clip_vector_cpu.tolist()
        return clip_vector





