import sys
import os
import json
import numpy as np
import time
import torch
from torch.nn.functional import normalize as torch_normalize
import msgpack
from random import shuffle, choice, sample
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.minio import cmd
from utility.http import request
from training_worker.ab_ranking.model import constants
from data_loader.utils import *

class TaggedDatasetLoader:
    def __init__(self,
                 minio_ip_addr=None,
                 minio_access_key=None,
                 minio_secret_key=None,
                 tag_name="",
                 input_type=constants.CLIP,
                 server_addr=None,
                 pooling_strategy=constants.AVERAGE_POOLING,
                 train_percent=0.9):
        if server_addr is None:
            self.server_addr = request.SERVER_ADDRESS
        if minio_access_key is not None:
            self.minio_access_key = minio_access_key
            self.minio_secret_key = minio_secret_key
            self.minio_client = cmd.get_minio_client(minio_access_key=self.minio_access_key,
                                                     minio_secret_key=self.minio_secret_key,
                                                     minio_ip_addr=minio_ip_addr)

        self.server_addr = server_addr
        self.tag_name = tag_name
        self.input_type = input_type
        self.pooling_strategy = pooling_strategy
        self.train_percent = train_percent

        self.positive_training_features = None
        self.positive_validation_features = None
        self.negative_training_features = None
        self.negative_validation_features = None

    def get_tagged_data(self, path, index=0):
        input_type_extension = "-text-embedding.msgpack"
        if self.input_type == constants.CLIP:
            input_type_extension = "_clip.msgpack"
        elif self.input_type in [constants.EMBEDDING, constants.EMBEDDING_POSITIVE, constants.EMBEDDING_NEGATIVE]:
            # replace with new /embeddings
            splits = path.split("/")
            dataset_name = splits[1]
            print("dataset_name=", dataset_name)
            path = path.replace(dataset_name, os.path.join(dataset_name, "embeddings/text-embedding"))

            input_type_extension = "-text-embedding.msgpack"
            if self.pooling_strategy == constants.AVERAGE_POOLING:
                input_type_extension = "-text-embedding-average-pooled.msgpack"
            elif self.pooling_strategy == constants.MAX_POOLING:
                input_type_extension = "-text-embedding-max-pooled.msgpack"
            elif self.pooling_strategy == constants.MAX_ABS_POOLING:
                input_type_extension = "-text-embedding-signed-max-pooled.msgpack"

        # get .msgpack data
        path = path.replace(".jpg", input_type_extension)
        path = path.replace("datasets/", "")

        features_data = get_object(self.minio_client, path)
        features_data = msgpack.unpackb(features_data)
        features_vector = []

        if self.input_type in [constants.EMBEDDING, constants.EMBEDDING_POSITIVE]:
            features_vector.extend(features_data["positive_embedding"]["__ndarray__"])
        if self.input_type in [constants.EMBEDDING, constants.EMBEDDING_NEGATIVE]:
            features_vector.extend(features_data["negative_embedding"]["__ndarray__"])
        if self.input_type == constants.CLIP:
            features_vector.extend(features_data["clip-feature-vector"])

        features_vector = np.array(features_vector)

        # check if feature is nan
        if np.isnan(features_vector).all():
            raise Exception("Features from {} is nan.".format(path))

        return features_vector, index

    def load_data(self, paths_list, pre_shuffle=True):
        print("Loading data to ram...")
        start_time = time.time()

        len_paths_list = len(paths_list)
        data_features = [None] * len_paths_list

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []

            count = 0
            for path in paths_list:
                futures.append(executor.submit(self.get_tagged_data, path=path, index=count))
                count += 1

            for future in tqdm(as_completed(futures), total=len(paths_list)):
                feature, index = future.result()
                data_features[index] = feature

        if pre_shuffle:
            # shuffle
            shuffled_training_data = []
            index_shuf = list(range(len_paths_list))
            shuffle(index_shuf)
            for i in index_shuf:
                shuffled_training_data.append(data_features[i])

            data_features = shuffled_training_data

        time_elapsed = time.time() - start_time
        print("Time elapsed: {0}s".format(format(time_elapsed, ".2f")))

        return data_features

    def separate_training_and_validation_features(self, features):
        training_features = []
        validation_features = []

        len_features = len(features)
        # calculate num validations
        num_validations = round((len_features * (1.0 - self.train_percent)))

        # get random index for validations
        validation_indices = sample(range(0, len_features - 1), num_validations)
        for i in range(len_features):
            if i in validation_indices:
                validation_features.append(features[i])
            else:
                training_features.append(features[i])

        return training_features, validation_features


    def load_dataset(self):
        # get all data based on tag
        tag_list = request.http_get_tag_list()

        tag_id = None
        for data in tag_list:
            if data["tag_string"] == self.tag_name:
                tag_id = data["tag_id"]
                break
        if tag_id is None:
            raise Exception("Tag name not found")

        tagged_images = request.http_get_tagged_images(tag_id)

        positive_tagged_dataset = []
        negative_tagged_dataset = []
        for data in tagged_images:
            # separate data into two, positive and negative
            positive_tagged_dataset.append(data["file_path"])
            negative_tagged_dataset.append(data["file_path"])

        # make sure positive and negative have the same length
        # for now we want them to be 50/50
        min_length = min(len(positive_tagged_dataset), len(negative_tagged_dataset))
        positive_tagged_dataset = positive_tagged_dataset[:min_length]
        negative_tagged_dataset = negative_tagged_dataset[:min_length]

        # load proper input type: either clip image embedding or text embedding
        positive_tagged_features = self.load_data(positive_tagged_dataset)
        negative_tagged_features = self.load_data(negative_tagged_dataset)

        (self.positive_training_features,
         self.positive_validation_features) = self.separate_training_and_validation_features(positive_tagged_features)
        (self.negative_training_features,
         self.negative_validation_features) = self.separate_training_and_validation_features(negative_tagged_features)


    def get_training_positive_features(self, target=1.0):
        # return positive training data
        target_features = [target] * len(self.positive_training_features)
        return self.positive_training_features, target_features

    def get_validation_positive_features(self, target=1.0):
        # return positive validation data
        target_features = [target] * len(self.positive_validation_features)
        return self.positive_validation_features, target_features

    def get_training_negative_features(self, target=0.0):
        # return negative training data
        target_features = [target] * len(self.negative_training_features)
        return self.negative_training_features, target_features

    def get_validation_negative_features(self, target=0.0):
        # return negative validation data
        target_features = [target] * len(self.negative_validation_features)
        return self.negative_validation_features, target_features