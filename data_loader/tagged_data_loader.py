from datetime import datetime, timedelta
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

from utility.path import separate_bucket_and_file_path
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
                 train_percent=0.9,
                 epochs=100):
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
        self.epochs = epochs

        self.positive_training_features = None
        self.positive_validation_features = None
        self.negative_training_features_pool = None
        self.negative_validation_features_pool = None

        # sizes
        self.positive_dataset_total_size = None
        self.negative_dataset_total_size = None
        self.training_positive_size = None
        self.validation_positive_size = None
        self.training_negative_size = None
        self.validation_negative_size = None

        # we currently get dataset name based on a sample image path
        # since image tag instance doesn't have dataset name value
        self.dataset_name = None

    def get_tagged_data(self, path, index=0):
        input_type_extension = "-text-embedding.msgpack"
        if self.input_type in [constants.CLIP, constants.CLIP_WITH_LENGTH]:
            input_type_extension = "_clip.msgpack"
        elif self.input_type in [constants.KANDINSKY_CLIP, constants.KANDINSKY_CLIP_WITH_LENGTH]:
            input_type_extension = "_clip_kandinsky.msgpack"
        elif self.input_type in [constants.EMBEDDING, constants.EMBEDDING_POSITIVE, constants.EMBEDDING_NEGATIVE]:
            # replace with new /embeddings
            splits = path.split("/")
            dataset_name = splits[1]

            path = path.replace(dataset_name, os.path.join(dataset_name, "embeddings/text-embedding"))

            input_type_extension = "-text-embedding.msgpack"
            if self.pooling_strategy == constants.AVERAGE_POOLING:
                input_type_extension = "-text-embedding-average-pooled.msgpack"
            elif self.pooling_strategy == constants.MAX_POOLING:
                input_type_extension = "-text-embedding-max-pooled.msgpack"
            elif self.pooling_strategy == constants.MAX_ABS_POOLING:
                input_type_extension = "-text-embedding-signed-max-pooled.msgpack"

        # get .msgpack data
        file_path = path.replace(".jpg", input_type_extension)
        bucket_name, file_path = separate_bucket_and_file_path(file_path)
        
        try:
            features_data = get_object_with_bucket(self.minio_client, bucket_name, file_path)
        except Exception as e:
            print(f"Error: {e} when loading {file_path}")
        
        features_data = msgpack.unpackb(features_data)
        features_vector = []

        if self.input_type in [constants.EMBEDDING, constants.EMBEDDING_POSITIVE]:
            features_vector.extend(features_data["positive_embedding"]["__ndarray__"])
        if self.input_type in [constants.EMBEDDING, constants.EMBEDDING_NEGATIVE]:
            features_vector.extend(features_data["negative_embedding"]["__ndarray__"])
        if self.input_type in [constants.CLIP, constants.KANDINSKY_CLIP, 
                               constants.KANDINSKY_CLIP_WITH_LENGTH,
                               constants.CLIP_WITH_LENGTH]:
            features_vector.extend(features_data["clip-feature-vector"])

        features_vector = np.array(features_vector, dtype=np.float32)
        features_vector = torch.tensor(features_vector).to(torch.float)

        if self.input_type in [constants.KANDINSKY_CLIP_WITH_LENGTH, constants.CLIP_WITH_LENGTH] and len(features_vector)!= 0:
            vector_length= torch.norm(features_vector, dim=1).unsqueeze(1)
            features_vector= torch.cat([features_vector , vector_length], dim=1)

        # concatenate if len more than 2
        features_vector = features_vector.reshape(1, -1)
        features_vector = features_vector.squeeze()

        # check if feature is nan
        if torch.isnan(features_vector).all():
            raise Exception("Features from {} is nan or has nan values.".format(feature_filepath))

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
                if feature is not None:
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


    def load_dataset(self, image_type="all_resolutions"): # resolution ['all_resolutions', '512*512_resolution']
        # get all data based on tag
        tag_list = request.http_get_tag_list()

        tag_id = None
        for data in tag_list:
            if data["tag_string"] == self.tag_name:
                tag_id = data["tag_id"]
                break
        if tag_id is None:
            raise Exception("Tag name not found")

        tagged_images = request.http_get_tagged_images_by_image_type(tag_id, image_type)

        positive_tagged_dataset = []
        negative_tagged_dataset = []
        for data in tagged_images:
            # separate data into two, positive and negative
            if data["tag_type"] == 1:
                positive_tagged_dataset.append(data["file_path"])
            else:
                negative_tagged_dataset.append(data["file_path"])

        # get dataset name from a sample path
        splits = positive_tagged_dataset[0].split("/")
        dataset_name = splits[1]
        self.dataset_name = "environmental"

        # get random images for negatives
        # get from environmental for now
        # Format today's date as a string
        today = datetime.now()

        # Subtract one day to today's date and format as a string
        end_date = (today - timedelta(days=1)).strftime('%Y-%m-%d')

        random_image_list = request.http_get_random_image_by_date(dataset="environmental", size=len(positive_tagged_dataset) * self.epochs, end_date=end_date)
        # get paths only
        for image_data in random_image_list:
            negative_tagged_dataset.append(image_data["task_output_file_dict"]["output_file_path"])

        # load proper input type: either clip image embedding or text embedding
        positive_tagged_features = self.load_data(positive_tagged_dataset)
        negative_tagged_features = self.load_data(negative_tagged_dataset)

        (self.positive_training_features,
         self.positive_validation_features) = self.separate_training_and_validation_features(positive_tagged_features)
        (self.negative_training_features_pool,
         self.negative_validation_features_pool) = self.separate_training_and_validation_features(negative_tagged_features)

        # len
        self.positive_dataset_total_size = len(self.positive_training_features)
        self.negative_dataset_total_size = len(self.negative_training_features_pool)
        self.training_positive_size = len(self.positive_training_features)
        self.validation_positive_size = len(self.positive_validation_features)
        self.training_negative_size = len(self.positive_training_features)  # should be same as positive dataset
        self.validation_negative_size = len(self.positive_validation_features)  # should be same as positive dataset

    def get_training_positive_features(self, target=1.0):
        # return positive training data
        target_features = [torch.tensor(target)] * len(self.positive_training_features)
        return torch.stack(self.positive_training_features), torch.stack(target_features)

    def get_validation_positive_features(self, target=1.0):
        # return positive validation data
        target_features = [torch.tensor(target)] * len(self.positive_validation_features)
        return torch.stack(self.positive_validation_features), torch.stack(target_features)

    def get_training_negative_features(self, target=0.0):
        # return negative training data
        target_features = [torch.tensor(target)] * self.training_negative_size
        training_features = sample(self.negative_training_features_pool, self.training_negative_size)

        return torch.stack(training_features), torch.stack(target_features)

    def get_validation_negative_features(self, target=0.0):
        # return negative validation data
        target_features = [torch.tensor(target)] * self.validation_negative_size
        training_features = sample(self.negative_validation_features_pool, self.validation_negative_size)

        return torch.stack(training_features), torch.stack(target_features)

    def get_shuffled_positive_and_negative_training(self):
        training_positive_features, training_positive_targets = self.get_training_positive_features()
        training_negative_features, training_negative_targets = self.get_training_negative_features()

        stacked_features = torch.cat((training_positive_features, training_negative_features))
        stacked_targets = torch.cat((training_positive_targets, training_negative_targets))

        shuffled_idx = torch.randperm(len(stacked_features))
        stacked_features = stacked_features[shuffled_idx]
        stacked_targets = stacked_targets[shuffled_idx]

        stacked_targets = stacked_targets.unsqueeze(1)

        return stacked_features, stacked_targets

    def get_shuffled_positive_and_negative_validation(self):
        validation_positive_features, validation_positive_targets = self.get_validation_positive_features()
        validation_negative_features, validation_negative_targets = self.get_validation_negative_features()

        stacked_features = torch.cat((validation_positive_features, validation_negative_features))
        stacked_targets = torch.cat((validation_positive_targets, validation_negative_targets))

        shuffled_idx = torch.randperm(len(stacked_features))
        stacked_features = stacked_features[shuffled_idx]
        stacked_targets = stacked_targets[shuffled_idx]

        stacked_targets = stacked_targets.unsqueeze(1)

        return stacked_features, stacked_targets