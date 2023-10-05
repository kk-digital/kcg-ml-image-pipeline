from random import shuffle
import os
import zipfile
import sys
import json
import random
import numpy as np
import time
import torch
from queue import Queue
from threading import Semaphore
import msgpack

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.minio import cmd


DATASETS_BUCKET = "datasets"


def get_datasets(minio_client):
    datasets = cmd.get_list_of_objects(minio_client, DATASETS_BUCKET)
    return datasets


def get_aggregated_selection_datapoints(minio_client, dataset_name):
    prefix = os.path.join(dataset_name, "data/ranking")
    datasets = cmd.get_list_of_objects_with_prefix(minio_client, DATASETS_BUCKET, prefix=prefix)

    return datasets


def get_object(client, file_path):
    response = client.get_object(DATASETS_BUCKET, file_path)
    data = response.data

    return data

class ABData:
    def __init__(self, task, username, hash_image_1, hash_image_2, selected_image_index, selected_image_hash,
                 image_archive, image_1_path, image_2_path, datetime, selected_clip_feature_vector,
                 other_clip_feature_vector):
        self.task = task
        self.username = username
        self.hash_image_1 = hash_image_1
        self.hash_image_2 = hash_image_2
        self.selected_image_index = selected_image_index
        self.selected_image_hash = selected_image_hash
        self.image_archive = image_archive
        self.image_1_path = image_1_path
        self.image_2_path = image_2_path
        self.datetime = datetime
        self.selected_clip_feature_vector = selected_clip_feature_vector
        self.other_clip_feature_vector = other_clip_feature_vector


def split_ab_data_vectors(image_pair_data):
    image_x_feature_vector = image_pair_data[0]
    image_y_feature_vector = image_pair_data[1]
    target_probability = image_pair_data[2]

    return image_x_feature_vector, image_y_feature_vector, target_probability


class ABRankingDatasetLoader:
    def __init__(self,
                 dataset_name,
                 minio_access_key,
                 minio_secret_key,
                 buffer_size=20000,
                 train_percent=0.9):
        self.dataset_name = dataset_name

        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.minio_client = cmd.get_minio_client(self.minio_access_key, self.minio_secret_key)

        self.train_percent = train_percent
        self.dataset_paths = Queue()

        # these will contain features and targets with limit buffer size
        self.training_image_pair_data_buffer = Queue()
        self.validation_image_pair_data_buffer = Queue()

        # buffer size
        self.buffer_size = buffer_size  # N datapoints
        self.num_concurrent_loading = 8

        self.fill_semaphore = Semaphore(1)  # One filling only

    def load_dataset(self):
        start_time = time.time()
        print("Loading dataset references...")

        dataset_list = get_datasets(self.minio_client)
        if self.dataset_name not in dataset_list:
            raise Exception("Dataset is not in minio server")

        # if exist then get paths for aggregated selection datapoints
        dataset_paths = get_aggregated_selection_datapoints(self.minio_client, self.dataset_name)
        print(dataset_paths)

        for path in dataset_paths:
            self.dataset_paths.put(path)

        print("Dataset loaded...")
        print("Time elapsed: {0}s".format(format(time.time() - start_time, ".2f")))

    # TODO: remove hardcoded 200 when validations' proper len is implemented
    def get_total_validation_features_len(self):
        return min(self.dataset_paths.qsize(), 200)

    def get_selection_datapoint_image_pair(self, data_path):
        print("Getting object...")
        image_pair_data_list = []

        # load json object from minio
        data = get_object(self.minio_client, data_path)
        decoded_data = data.decode().replace("'", '"')
        json_data = json.loads(decoded_data)

        for item in json_data:
            selected_image_index = item["selected_image_index"]

            # features vector is in file_path.clip.msgpack
            file_path_img_1 = item["image_1_metadata"]["file_path"]
            file_path_img_2 = item["image_2_metadata"]["file_path"]

            clip_path_img_1 = file_path_img_1.replace(".jpg", "_clip.msgpack")
            clip_path_img_1 = clip_path_img_1.replace("datasets/", "")

            clip_path_img_2 = file_path_img_2.replace(".jpg", "_clip.msgpack")
            clip_path_img_2 = clip_path_img_2.replace("datasets/", "")

            clip_img_1_data = get_object(self.minio_client, clip_path_img_1)
            clip_img_1_data = msgpack.unpackb(clip_img_1_data)
            clip_img_1_data = clip_img_1_data["clip-feature-vector"]
            clip_img_2_data = get_object(self.minio_client, clip_path_img_2)
            clip_img_2_data = msgpack.unpackb(clip_img_2_data)
            clip_img_2_data = clip_img_2_data["clip-feature-vector"]

            # if image 1 is the selected
            if selected_image_index == 0:
                selected_clip_feature_vector = clip_img_1_data
                other_clip_feature_vector = clip_img_2_data

            # image 2 is selected
            else:
                selected_clip_feature_vector = clip_img_2_data
                other_clip_feature_vector = clip_img_1_data

            # ab_data = ABData(task=item["task"],
            #                  username=item["username"],
            #                  hash_image_1=item["image_1_metadata"]["file_hash"],
            #                  hash_image_2=item["image_2_metadata"]["file_hash"],
            #                  selected_image_index=selected_image_index,
            #                  selected_image_hash=item["selected_image_hash"],
            #                  image_archive="",
            #                  image_1_path=file_path_img_1,
            #                  image_2_path=file_path_img_2,
            #                  datetime=item["datetime"],
            #                  selected_clip_feature_vector=selected_clip_feature_vector,
            #                  other_clip_feature_vector=other_clip_feature_vector,
            #)


            # (x, y, 1.0)
            image_pair_target_1 = (selected_clip_feature_vector, other_clip_feature_vector, [1.0])
            image_pair_data_list.append(image_pair_target_1)

            # (y, x) = 0.0
            image_pair_target_0 = (other_clip_feature_vector, selected_clip_feature_vector, [0.0])
            image_pair_data_list.append(image_pair_target_0)

        return image_pair_data_list

    def get_training_data_and_save_to_buffer(self, path):
        # get data
        image_pair_data_list = self.get_selection_datapoint_image_pair(path)

        # add to training data buffer
        for data in image_pair_data_list:
            self.training_image_pair_data_buffer.put(data)

    def fill_training_data_buffer(self):
        if not self.fill_semaphore.acquire(blocking=False):
            return

        while self.dataset_paths.qsize() > 0:
            start_time = time.time()
            print("Filling training data buffer in background...")

            while self.training_image_pair_data_buffer.qsize() < self.buffer_size:
                if self.dataset_paths.qsize() <= 0:
                    break

                path = self.dataset_paths.get()
                self.get_training_data_and_save_to_buffer(path)

            print("Training data buffer filled...")
            print("Time elapsed: {0}s".format(format(time.time() - start_time, ".2f")))

            # check every 1s if queue needs to be refilled
            while self.dataset_paths.qsize() > 0:
                time.sleep(1)
                if self.training_image_pair_data_buffer.qsize() < self.buffer_size:
                    break

        self.fill_semaphore.release()

    def get_next_training_feature_vectors_and_target(self, num_data):
        image_x_feature_vectors = []
        image_y_feature_vectors = []
        target_probabilities = []

        for _ in range(num_data):
            training_image_pair_data = self.training_image_pair_data_buffer.get()
            image_x_feature_vector, image_y_feature_vector, target_probability = split_ab_data_vectors(training_image_pair_data)
            image_x_feature_vectors.append(image_x_feature_vector)
            image_y_feature_vectors.append(image_y_feature_vector)
            target_probabilities.append(target_probability)


        image_x_feature_vectors = np.array(image_x_feature_vectors, dtype=np.float32)
        image_y_feature_vectors = np.array(image_y_feature_vectors, dtype=np.float32)
        target_probabilities = np.array(target_probabilities)

        image_x_feature_vectors = torch.tensor(image_x_feature_vectors).to(torch.float).squeeze()
        image_y_feature_vectors = torch.tensor(image_y_feature_vectors).to(torch.float).squeeze()
        target_probabilities = torch.tensor(target_probabilities).to(torch.float)

        return image_x_feature_vectors, image_y_feature_vectors, target_probabilities

    #
    # # TODO: finalize how many validations to use
    # # hard code to use 200 validation data for now
    # # get random 200 data from validation buffer
    # def get_validation_feature_vector_and_target(self, num_data=200):
    #     validation_path = []
    #     validation_feature_vector = []
    #     validation_targets = []
    #
    #     num_validations = min(self.validation_dataset_paths.qsize(), num_data)
    #     # get 200 paths
    #     while len(validation_path) < num_validations:
    #         path = self.validation_dataset_paths.get()
    #         validation_path.append(path)
    #
    #     while len(validation_targets) < num_validations:
    #         # get data
    #         data_path = validation_path.pop(-1)
    #         feature, target = self.get_selection_datapoint_image_pair(data_path)
    #
    #         # add to training data buffer
    #         validation_feature_vector.append(feature)
    #         validation_targets.append(target)
    #
    #     validation_feature_vector = torch.tensor(validation_feature_vector)
    #     if self.input_type == "clip":
    #         validation_feature_vector = validation_feature_vector.unsqueeze(1)
    #
    #     validation_targets = torch.tensor(validation_targets)
    #     if self.target_type == "chad-score" or self.target_type == "score":
    #         validation_targets = torch.tensor(validation_targets).unsqueeze(1)
    #     elif self.target_type == "clip-feature-vector":
    #         validation_targets = torch.tensor(validation_targets).squeeze(1)
    #
    #     return validation_feature_vector, validation_targets

