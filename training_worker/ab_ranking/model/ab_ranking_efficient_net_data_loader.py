import os
import sys
import json
import numpy as np
import time
import torch
from queue import Queue
from threading import Semaphore
import msgpack
import threading

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.minio import cmd


DATASETS_BUCKET = "datasets"


def get_datasets(minio_client):
    datasets = cmd.get_list_of_objects(minio_client, DATASETS_BUCKET)
    return datasets


def get_aggregated_selection_datapoints(minio_client, dataset_name):
    prefix = os.path.join(dataset_name, "data/ranking/aggregate")
    datasets = cmd.get_list_of_objects_with_prefix(minio_client, DATASETS_BUCKET, prefix=prefix)

    return datasets


def get_object(client, file_path):
    response = client.get_object(DATASETS_BUCKET, file_path)
    data = response.data

    return data


class ABData:
    def __init__(self, task, username, hash_image_1, hash_image_2, selected_image_index, selected_image_hash,
                 image_archive, image_1_path, image_2_path, datetime):
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
        self.training_dataset_paths_copy = []
        self.training_dataset_paths_queue = Queue()
        self.validation_dataset_paths_queue = Queue()

        # these will contain features and targets with limit buffer size
        self.training_image_pair_data_buffer = Queue()
        self.validation_image_pair_data_buffer = Queue()

        # buffer size
        self.buffer_size = buffer_size  # N datapoints
        self.num_concurrent_loading = 8

        self.num_filling_workers = 5
        self.fill_semaphore = Semaphore(self.num_filling_workers)  # One filling only

    def load_dataset(self):
        start_time = time.time()
        print("Loading dataset references...")

        dataset_list = get_datasets(self.minio_client)
        if self.dataset_name not in dataset_list:
            raise Exception("Dataset is not in minio server")

        # if exist then get paths for aggregated selection datapoints
        dataset_paths = get_aggregated_selection_datapoints(self.minio_client, self.dataset_name)
        print("# of dataset paths retrieved=", len(dataset_paths))
        if len(dataset_paths) == 0:
            raise Exception("No selection datapoints json found.")

        # calculate num validations
        num_validations = round((len(dataset_paths) * (1.0 - self.train_percent)))
        validation_ab_data_list = dataset_paths[:num_validations]
        training_ab_data_list = dataset_paths[num_validations:]
        self.training_dataset_paths_copy = training_ab_data_list

        # put to their queue
        for data in validation_ab_data_list:
            self.validation_dataset_paths_queue.put(data)

        for data in training_ab_data_list:
            self.training_dataset_paths_queue.put(data)

        print("Dataset loaded...")
        print("Time elapsed: {0}s".format(format(time.time() - start_time, ".2f")))

    def fill_training_ab_data(self):
        for data in self.training_dataset_paths_copy:
            self.training_dataset_paths_queue.put(data)

    def get_len_training_ab_data(self):
        return self.training_dataset_paths_queue.qsize()

    def get_len_validation_ab_data(self):
        return self.validation_dataset_paths_queue.qsize()

    def get_selection_datapoint_image_pair(self, dataset_path):
        image_pair_data_list = []

        # load json object from minio
        data = get_object(self.minio_client, dataset_path)
        decoded_data = data.decode().replace("'", '"')
        item = json.loads(decoded_data)
        ab_data = ABData(task=item["task"],
                         username=item["username"],
                         hash_image_1=item["image_1_metadata"]["file_hash"],
                         hash_image_2=item["image_2_metadata"]["file_hash"],
                         selected_image_index=item["selected_image_index"],
                         selected_image_hash=item["selected_image_hash"],
                         image_archive="",
                         image_1_path=item["image_1_metadata"]["file_path"],
                         image_2_path=item["image_2_metadata"]["file_path"],
                         datetime=item["datetime"])

        selected_image_index = ab_data.selected_image_index
        file_path_img_1 = ab_data.image_1_path
        file_path_img_2 = ab_data.image_2_path

        # embeddings are in file_path_embedding.msgpack
        embeddings_path_img_1 = file_path_img_1.replace(".jpg", "_embedding.msgpack")
        embeddings_path_img_1 = embeddings_path_img_1.replace("datasets/", "")

        embeddings_path_img_2 = file_path_img_2.replace(".jpg", "_embedding.msgpack")
        embeddings_path_img_2 = embeddings_path_img_2.replace("datasets/", "")

        embeddings_img_1_data = get_object(self.minio_client, embeddings_path_img_1)
        embeddings_img_1_data = msgpack.unpackb(embeddings_img_1_data)
        embeddings_img_1_embeddings_vector = []
        embeddings_img_1_embeddings_vector.extend(embeddings_img_1_data["positive_embedding"]["__ndarray__"])
        embeddings_img_1_embeddings_vector.extend(embeddings_img_1_data["negative_embedding"]["__ndarray__"])
        embeddings_img_1_embeddings_vector = np.array(embeddings_img_1_embeddings_vector)

        embeddings_img_2_data = get_object(self.minio_client, embeddings_path_img_2)
        embeddings_img_2_data = msgpack.unpackb(embeddings_img_2_data)
        embeddings_img_2_embeddings_vector = []
        embeddings_img_2_embeddings_vector.extend(embeddings_img_2_data["positive_embedding"]["__ndarray__"])
        embeddings_img_2_embeddings_vector.extend(embeddings_img_2_data["negative_embedding"]["__ndarray__"])
        embeddings_img_2_embeddings_vector = np.array(embeddings_img_2_embeddings_vector)

        # if image 1 is the selected
        if selected_image_index == 0:
            selected_embeddings_vector = embeddings_img_1_embeddings_vector
            other_embeddings_vector = embeddings_img_2_embeddings_vector

        # image 2 is selected
        else:
            selected_embeddings_vector = embeddings_img_2_embeddings_vector
            other_embeddings_vector = embeddings_img_1_embeddings_vector

        image_pair_target_1 = (selected_embeddings_vector, other_embeddings_vector, [1.0])
        image_pair_data_list.append(image_pair_target_1)

        # (y, x) = 0.0
        image_pair_target_0 = (other_embeddings_vector, selected_embeddings_vector, [0.0])
        image_pair_data_list.append(image_pair_target_0)

        return image_pair_data_list

    def get_training_data_and_save_to_buffer(self, dataset_path):
        # get data
        image_pair_data_list = self.get_selection_datapoint_image_pair(dataset_path)

        # add to training data buffer
        for data in image_pair_data_list:
            self.training_image_pair_data_buffer.put(data)

    def fill_training_data_buffer(self):
        if not self.fill_semaphore.acquire(blocking=False):
            return

        while self.training_dataset_paths_queue.qsize() > 0:
            start_time = time.time()
            print("Filling training data buffer in background...")

            while self.training_image_pair_data_buffer.qsize() < self.buffer_size:
                if self.training_dataset_paths_queue.qsize() <= 0:
                    break

                dataset_path = self.training_dataset_paths_queue.get()
                self.get_training_data_and_save_to_buffer(dataset_path)

            print("Training data buffer filled...")
            print("Time elapsed: {0}s".format(format(time.time() - start_time, ".2f")))

            # check every 1s if queue needs to be refilled
            while self.training_dataset_paths_queue.qsize() > 0:
                time.sleep(1)
                if self.training_image_pair_data_buffer.qsize() < self.buffer_size:
                    break

        self.fill_semaphore.release()

    def spawn_filling_workers(self):
        for i in range(self.num_filling_workers):
            # fill data buffer
            # if buffer is empty, fill data
            fill_buffer_thread = threading.Thread(target=self.fill_training_data_buffer)
            fill_buffer_thread.start()

    def get_next_training_feature_vectors_and_target(self, num_data, device=None):
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

        image_x_feature_vectors = torch.tensor(image_x_feature_vectors).to(torch.float)
        image_y_feature_vectors = torch.tensor(image_y_feature_vectors).to(torch.float)

        target_probabilities = torch.tensor(target_probabilities).to(torch.float)

        if device is not None:
            image_x_feature_vectors = image_x_feature_vectors.to(device)
            image_y_feature_vectors = image_y_feature_vectors.to(device)
            target_probabilities = target_probabilities.to(device)

        return image_x_feature_vectors, image_y_feature_vectors, target_probabilities

    def get_validation_feature_vectors_and_target(self):
        image_x_feature_vectors = []
        image_y_feature_vectors = []
        target_probabilities = []

        # get ab data
        while self.validation_dataset_paths_queue.qsize() > 0:
            dataset_path = self.validation_dataset_paths_queue.get()
            image_pair_data_list = self.get_selection_datapoint_image_pair(dataset_path)
            for image_pair in image_pair_data_list:
                image_x_feature_vector, image_y_feature_vector, target_probability = split_ab_data_vectors(
                    image_pair)
                image_x_feature_vectors.append(image_x_feature_vector)
                image_y_feature_vectors.append(image_y_feature_vector)
                target_probabilities.append(target_probability)

        image_x_feature_vectors = np.array(image_x_feature_vectors, dtype=np.float32)
        image_y_feature_vectors = np.array(image_y_feature_vectors, dtype=np.float32)
        target_probabilities = np.array(target_probabilities)

        image_x_feature_vectors = torch.tensor(image_x_feature_vectors).to(torch.float)
        image_y_feature_vectors = torch.tensor(image_y_feature_vectors).to(torch.float)
        target_probabilities = torch.tensor(target_probabilities).to(torch.float)

        return image_x_feature_vectors, image_y_feature_vectors, target_probabilities
