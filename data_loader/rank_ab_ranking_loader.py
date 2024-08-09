import os
import sys
import json
import numpy as np
import time
import torch
from torch.nn.functional import normalize as torch_normalize
import msgpack
from random import shuffle, choice, sample
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.minio import cmd
from training_worker.ab_ranking.model import constants
from data_loader.ab_data import ABData
from data_loader.utils import *

# import request service for getting rank model list
from utility.http.request import http_get_rank_list


class ABRankingDatasetLoader:
    def __init__(self,
                 rank_model_id,
                 minio_ip_addr=None,
                 minio_access_key=None,
                 minio_secret_key=None,
                 input_type="embedding",
                 train_percent=0.9,
                 load_to_ram=False,
                 pooling_strategy=constants.AVERAGE_POOLING,
                 normalize_vectors=True,
                 target_option=constants.TARGET_1_AND_0,
                 duplicate_flip_option=constants.DUPLICATE_AND_FLIP_ALL):
        self.rank_model_id = rank_model_id
        self.input_type = input_type

        if minio_access_key is not None:
            self.minio_access_key = minio_access_key
            self.minio_secret_key = minio_secret_key
            self.minio_client = cmd.get_minio_client(minio_access_key=self.minio_access_key,
                                                     minio_secret_key=self.minio_secret_key,
                                                     minio_ip_addr=minio_ip_addr)

        # config
        self.pooling_strategy = pooling_strategy
        self.normalize_vectors = normalize_vectors
        self.target_option = target_option
        self.duplicate_flip_option = duplicate_flip_option

        self.train_percent = train_percent
        self.total_selection_datapoints = 0
        self.training_data_total = 0
        self.validation_data_total = 0
        self.total_num_data = 0

        # for hyperparam
        self.training_dataset_paths_arr = []
        self.validation_dataset_paths_arr = []

        # load all data to ram
        self.load_to_ram = load_to_ram
        self.training_ab_data_paths_list = []
        self.validation_ab_data_paths_list = []
        self.current_training_data_index = 0
        self.training_image_pair_data_arr = []
        self.validation_image_pair_data_arr = []
        self.datapoints_per_sec = 0

        # for chronological data scores graph
        self.training_data_paths_indices = []
        self.validation_data_paths_indices = []
        self.training_data_paths_indices_shuffled = []
        self.validation_data_paths_indices_shuffled = []

        # for adding scores and residuals
        self.training_image_hashes = []
        self.validation_image_hashes = []

        # image data selected index count
        self.image_selected_index_0_count = 0
        self.image_selected_index_1_count = 0

    def load_dataset(self, pre_shuffle=True):
        start_time = time.time()
        print("Loading dataset references...")

        # Getting existing rank model list 
        rank_model_list = http_get_rank_list()
        rank_model_ids = [rank_model["rank_model_id"] for rank_model in rank_model_list]

        if self.rank_model_id not in rank_model_ids:
            raise Exception("Dataset is not in minio server")

        # if exist then get paths for aggregated selection datapoints
        dataset = get_aggregated_selection_datapoints_v1(self.minio_client, self.rank_model_id)
        len_dataset = len(dataset)
        print("# of dataset retrieved=", len_dataset)
        if len(dataset) == 0:
            print("No selection datapoints json found.")
            return False

        self.total_selection_datapoints = len_dataset

        # test
        # dataset = dataset[:5]

        # calculate num validations
        num_validations = round((len_dataset * (1.0 - self.train_percent)))

        # get random index for validations
        training_data_paths_indices = []
        validation_data_paths_indices = []
        validation_ab_data_list = []
        training_ab_data_list = []
        validation_indices = sample(range(0, len_dataset - 1), num_validations)
        for i in range(len_dataset):
            if i in validation_indices:
                validation_ab_data_list.append(dataset[i])
                validation_data_paths_indices.append(i)
            else:
                training_ab_data_list.append(dataset[i])
                training_data_paths_indices.append(i)

        self.training_ab_data_paths_list = training_ab_data_list
        self.validation_ab_data_paths_list = validation_ab_data_list

        self.training_data_paths_indices = training_data_paths_indices
        self.validation_data_paths_indices = validation_data_paths_indices

        # always load to ram
        self.load_all_training_data(self.training_ab_data_paths_list, pre_shuffle=pre_shuffle)
        self.load_all_validation_data(self.validation_ab_data_paths_list)
        self.total_num_data = self.validation_data_total + self.training_data_total

        print("Dataset loaded...")
        print("Time elapsed: {0}s".format(format(time.time() - start_time, ".2f")))

        return True

    def get_len_training_ab_data(self):
        return self.training_data_total

    def get_len_validation_ab_data(self):
        return self.validation_data_total

    def get_image_selected_index_data(self):
        selected_index_0_count = self.image_selected_index_0_count
        selected_index_1_count = self.image_selected_index_1_count
        total_count = selected_index_0_count + selected_index_1_count

        return selected_index_0_count, selected_index_1_count, total_count

    def get_selection_datapoint_image_pair(self, dataset, index=0):
        image_pairs = []
        ab_data = dataset

        selected_image_index = ab_data.selected_image_index
        file_path_img_1 = ab_data.image_1_path
        file_path_img_2 = ab_data.image_2_path

        input_type_extension = "-text-embedding.msgpack"
        if self.input_type == constants.CLIP:
            input_type_extension = "_clip.msgpack"
        elif self.input_type == constants.KANDINSKY_CLIP:
            input_type_extension = "_clip_kandinsky.msgpack"
        elif self.input_type in [constants.EMBEDDING, constants.EMBEDDING_POSITIVE, constants.EMBEDDING_NEGATIVE]:
            # replace with new /embeddings

            dataset_name = Path(file_path_img_1).parent.parent.name
            file_path_img_1 = file_path_img_1.replace(dataset_name,
                                                      os.path.join(dataset_name, "embeddings/text-embedding"))
            
            dataset_name = Path(file_path_img_1).parent.parent.name
            file_path_img_2 = file_path_img_2.replace(dataset_name,
                                                      os.path.join(dataset_name, "embeddings/text-embedding"))

            input_type_extension = "-text-embedding.msgpack"
            if self.pooling_strategy == constants.AVERAGE_POOLING:
                input_type_extension = "-text-embedding-average-pooled.msgpack"
            elif self.pooling_strategy == constants.MAX_POOLING:
                input_type_extension = "-text-embedding-max-pooled.msgpack"
            elif self.pooling_strategy == constants.MAX_ABS_POOLING:
                input_type_extension = "-text-embedding-signed-max-pooled.msgpack"

        # get .msgpack data
        features_path_img_1 = file_path_img_1.replace(".jpg", input_type_extension)
        bucket_img_1, features_path_img_1 = separate_bucket_and_file_path(features_path_img_1)

        features_path_img_2 = file_path_img_2.replace(".jpg", input_type_extension)
        bucket_img_2, features_path_img_2 = separate_bucket_and_file_path(features_path_img_2)

        features_image_1_response = cmd.get_file_from_minio(self.minio_client, bucket_img_1, features_path_img_1)
        if features_image_1_response is None:
            return None, None, None, None
        else:
            features_img_1_data = features_image_1_response.data
            
        features_img_1_data = msgpack.unpackb(features_img_1_data)
        features_vector_img_1 = []

        if self.input_type in [constants.EMBEDDING, constants.EMBEDDING_POSITIVE]:
            features_vector_img_1.extend(features_img_1_data["positive_embedding"]["__ndarray__"])
        if self.input_type in [constants.EMBEDDING, constants.EMBEDDING_NEGATIVE]:
            features_vector_img_1.extend(features_img_1_data["negative_embedding"]["__ndarray__"])
        if self.input_type in [constants.CLIP, constants.KANDINSKY_CLIP]:
            features_vector_img_1.extend(features_img_1_data["clip-feature-vector"])

        features_vector_img_1 = np.array(features_vector_img_1)
        
        features_image_2_response = cmd.get_file_from_minio(self.minio_client, bucket_img_2, features_path_img_2)
        if features_image_2_response is None:
            return None, None, None, None
        else:
            features_img_2_data = features_image_2_response.data
        
        features_img_2_data = msgpack.unpackb(features_img_2_data)
        features_vector_img_2 = []

        if self.input_type in [constants.EMBEDDING, constants.EMBEDDING_POSITIVE]:
            features_vector_img_2.extend(features_img_2_data["positive_embedding"]["__ndarray__"])
        if self.input_type in [constants.EMBEDDING, constants.EMBEDDING_NEGATIVE]:
            features_vector_img_2.extend(features_img_2_data["negative_embedding"]["__ndarray__"])
        if self.input_type in [constants.CLIP, constants.KANDINSKY_CLIP]:
            features_vector_img_2.extend(features_img_2_data["clip-feature-vector"])

        features_vector_img_2 = np.array(features_vector_img_2)

        # check if feature is nan
        if np.isnan(features_vector_img_1).all():
            raise Exception("Features from {} is nan.".format(features_path_img_1))
        if np.isnan(features_vector_img_2).all():
            raise Exception("Features from {} is nan.".format(features_path_img_2))

        # if image 1 is the selected
        if selected_image_index == 0:
            selected_features_vector = features_vector_img_1
            other_features_vector = features_vector_img_2
            self.image_selected_index_0_count += 1

            selected_img_hash = ab_data.hash_image_1
            other_img_hash = ab_data.hash_image_2
        # image 2 is selected
        else:
            selected_features_vector = features_vector_img_2
            other_features_vector = features_vector_img_1

            selected_img_hash = ab_data.hash_image_2
            other_img_hash = ab_data.hash_image_1
            self.image_selected_index_1_count += 1

        if (self.target_option == constants.TARGET_1_AND_0) or (
                self.target_option == constants.TARGET_1_ONLY):
            image_pair = (selected_features_vector, other_features_vector, [1.0])
            image_pairs.append(image_pair)

        if (self.target_option == constants.TARGET_1_AND_0) or (self.target_option == constants.TARGET_0_ONLY):
            use_target_0 = True
            if (self.target_option == constants.TARGET_1_AND_0) and (
                    self.duplicate_flip_option == constants.DUPLICATE_AND_FLIP_RANDOM):
                # then should have 50/50 chance of being duplicated or not
                rand_int = choice([0, 1])
                if rand_int == 1:
                    # then don't duplicate
                    use_target_0 = False

            if use_target_0:
                image_pair = (other_features_vector, selected_features_vector, [0.0])
                image_pairs.append(image_pair)

        return image_pairs, index, selected_img_hash, other_img_hash

    def load_all_training_data(self, paths_list, pre_shuffle=True):
        print("Loading all training data to ram...")
        start_time = time.time()
        new_training_data_paths_indices = []
        training_image_hashes = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []

            count = 0
            for path in paths_list:
                futures.append(executor.submit(self.get_selection_datapoint_image_pair, dataset=path, index=count))
                count += 1

            for future in tqdm(as_completed(futures), total=len(paths_list)):
                image_pairs, index, selected_img_hash, other_img_hash = future.result()
                if image_pairs is None:
                    continue
                for pair in image_pairs:
                    self.training_image_pair_data_arr.append(pair)
                    new_training_data_paths_indices.append(self.training_data_paths_indices[index])

                    if pair[2] == [1.0]:
                        training_image_hashes.append(selected_img_hash)
                    else:
                        training_image_hashes.append(other_img_hash)

        self.training_data_paths_indices = new_training_data_paths_indices

        len_training_data_paths = len(self.training_data_paths_indices)
        if pre_shuffle is False:
            self.training_data_paths_indices_shuffled = self.training_data_paths_indices
            self.training_image_hashes = training_image_hashes

        else:
            # shuffle
            shuffled_training_data = []
            shuffled_training_data_indices = []
            shuffled_training_image_hashes = []
            index_shuf = list(range(len_training_data_paths))
            shuffle(index_shuf)
            for i in index_shuf:
                shuffled_training_data.append(self.training_image_pair_data_arr[i])
                shuffled_training_data_indices.append(self.training_data_paths_indices[i])
                shuffled_training_image_hashes.append(training_image_hashes[i])

            self.training_data_paths_indices_shuffled = shuffled_training_data_indices
            self.training_image_pair_data_arr = shuffled_training_data
            self.training_image_hashes = shuffled_training_image_hashes

        self.training_data_total = len_training_data_paths
        time_elapsed = time.time() - start_time
        print("Time elapsed: {0}s".format(format(time_elapsed, ".2f")))
        self.datapoints_per_sec = len_training_data_paths / time_elapsed

    def load_all_validation_data(self, paths_list):
        print("Loading all validation data to ram...")
        start_time = time.time()

        new_validation_data_paths_indices = []
        validation_image_hashes = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []

            count = 0
            for path in paths_list:
                futures.append(executor.submit(self.get_selection_datapoint_image_pair, dataset=path, index=count))
                count += 1

            for future in tqdm(as_completed(futures), total=len(paths_list)):
                image_pairs, index, selected_img_hash, other_img_hash = future.result()
                if image_pairs is None:
                    continue
                for pair in image_pairs:
                    self.validation_image_pair_data_arr.append(pair)
                    new_validation_data_paths_indices.append(self.validation_data_paths_indices[index])

                    if pair[2] == [1.0]:
                        validation_image_hashes.append(selected_img_hash)
                    else:
                        validation_image_hashes.append(other_img_hash)

        self.validation_data_paths_indices = new_validation_data_paths_indices

        # shuffle
        shuffled_validation_data = []
        shuffled_validation_data_indices = []
        shuffle_validation_image_hashes = []
        len_validation_data_paths = len(self.validation_data_paths_indices)
        index_shuf = list(range(len_validation_data_paths))
        shuffle(index_shuf)
        for i in index_shuf:
            shuffled_validation_data.append(self.validation_image_pair_data_arr[i])
            shuffled_validation_data_indices.append(self.validation_data_paths_indices[i])
            shuffle_validation_image_hashes.append(validation_image_hashes[i])

        self.validation_data_paths_indices_shuffled = shuffled_validation_data_indices
        self.validation_image_pair_data_arr = shuffled_validation_data
        self.validation_data_total = len_validation_data_paths
        self.validation_image_hashes = shuffle_validation_image_hashes

        time_elapsed = time.time() - start_time
        print("Time elapsed: {0}s".format(format(time_elapsed, ".2f")))

    def shuffle_training_data(self):
        print("Shuffling training data...")
        # shuffle
        new_shuffled_indices = []
        shuffled_training = []
        shuffled_training_image_hashes = []
        index_shuf = list(range(len(self.training_image_pair_data_arr)))
        shuffle(index_shuf)
        for i in index_shuf:
            shuffled_training.append(self.training_image_pair_data_arr[i])
            new_shuffled_indices.append(self.training_data_paths_indices_shuffled[i])
            shuffled_training_image_hashes.append(self.training_image_hashes[i])

        self.training_data_paths_indices_shuffled = new_shuffled_indices
        self.training_image_pair_data_arr = shuffled_training
        self.training_image_hashes = shuffled_training_image_hashes

    # ------------------------------- For AB Ranking Efficient Net -------------------------------
    def get_next_training_feature_vectors_and_target_efficient_net(self, num_data, device=None):
        image_x_feature_vectors = []
        image_y_feature_vectors = []
        target_probabilities = []

        for _ in range(num_data):
            training_image_pair_data = self.training_image_pair_data_arr[self.current_training_data_index]
            image_x_feature_vector, image_y_feature_vector, target_probability = split_ab_data_vectors(
                training_image_pair_data)
            image_x_feature_vectors.append(image_x_feature_vector)
            image_y_feature_vectors.append(image_y_feature_vector)
            target_probabilities.append(target_probability)
            self.current_training_data_index += 1

        image_x_feature_vectors = np.array(image_x_feature_vectors, dtype=np.float32)
        image_y_feature_vectors = np.array(image_y_feature_vectors, dtype=np.float32)

        target_probabilities = np.array(target_probabilities)

        image_x_feature_vectors = torch.tensor(image_x_feature_vectors).to(torch.float)
        image_y_feature_vectors = torch.tensor(image_y_feature_vectors).to(torch.float)
        target_probabilities = torch.tensor(target_probabilities).to(torch.float)

        if self.normalize_vectors:
            image_x_feature_vectors = torch_normalize(image_x_feature_vectors, p=1.0, dim=2)
            image_y_feature_vectors = torch_normalize(image_y_feature_vectors, p=1.0, dim=2)

        # then concatenate
        image_x_feature_vectors = image_x_feature_vectors.reshape(len(image_x_feature_vectors), -1)
        image_y_feature_vectors = image_y_feature_vectors.reshape(len(image_y_feature_vectors), -1)

        image_x_feature_vectors = image_x_feature_vectors.unsqueeze(1)
        image_y_feature_vectors = image_y_feature_vectors.unsqueeze(1)
        image_x_feature_vectors = image_x_feature_vectors.unsqueeze(1)
        image_y_feature_vectors = image_y_feature_vectors.unsqueeze(1)

        if device is not None:
            image_x_feature_vectors = image_x_feature_vectors.to(device)
            image_y_feature_vectors = image_y_feature_vectors.to(device)
            target_probabilities = target_probabilities.to(device)

        return image_x_feature_vectors, image_y_feature_vectors, target_probabilities

    def get_validation_feature_vectors_and_target_efficient_net(self):
        image_x_feature_vectors = []
        image_y_feature_vectors = []
        target_probabilities = []

        # get ab data
        for i in range(len(self.validation_image_pair_data_arr)):
            validation_image_pair_data = self.validation_image_pair_data_arr[i]
            image_x_feature_vector, image_y_feature_vector, target_probability = split_ab_data_vectors(
                validation_image_pair_data)
            image_x_feature_vectors.append(image_x_feature_vector)
            image_y_feature_vectors.append(image_y_feature_vector)
            target_probabilities.append(target_probability)

        image_x_feature_vectors = np.array(image_x_feature_vectors, dtype=np.float32)
        image_y_feature_vectors = np.array(image_y_feature_vectors, dtype=np.float32)
        target_probabilities = np.array(target_probabilities)

        image_x_feature_vectors = torch.tensor(image_x_feature_vectors).to(torch.float)
        image_y_feature_vectors = torch.tensor(image_y_feature_vectors).to(torch.float)
        target_probabilities = torch.tensor(target_probabilities).to(torch.float)
        print("feature shape =", image_x_feature_vectors.shape)

        if self.normalize_vectors:
            image_x_feature_vectors = torch_normalize(image_x_feature_vectors, p=1.0, dim=2)
            image_y_feature_vectors = torch_normalize(image_y_feature_vectors, p=1.0, dim=2)
            print("feature shape after normalizing=", image_x_feature_vectors.shape)

        # then concatenate
        image_x_feature_vectors = image_x_feature_vectors.reshape(len(image_x_feature_vectors), -1)
        image_y_feature_vectors = image_y_feature_vectors.reshape(len(image_y_feature_vectors), -1)

        image_x_feature_vectors = image_x_feature_vectors.unsqueeze(1)
        image_y_feature_vectors = image_y_feature_vectors.unsqueeze(1)
        image_x_feature_vectors = image_x_feature_vectors.unsqueeze(1)
        image_y_feature_vectors = image_y_feature_vectors.unsqueeze(1)

        print("feature shape after pooling and unsqueeze=", image_x_feature_vectors.shape)

        return image_x_feature_vectors, image_y_feature_vectors, target_probabilities

    # ------------------------------- For AB Ranking Linear -------------------------------
    def get_next_training_feature_vectors_and_target_linear(self, num_data, device=None):
        image_x_feature_vectors = []
        image_y_feature_vectors = []
        target_probabilities = []

        for _ in range(num_data):
            training_image_pair_data = self.training_image_pair_data_arr[self.current_training_data_index]
            image_x_feature_vector, image_y_feature_vector, target_probability = split_ab_data_vectors(
                training_image_pair_data)
            image_x_feature_vectors.append(image_x_feature_vector)
            image_y_feature_vectors.append(image_y_feature_vector)
            target_probabilities.append(target_probability)
            self.current_training_data_index += 1

        image_x_feature_vectors = np.array(image_x_feature_vectors, dtype=np.float32)
        image_y_feature_vectors = np.array(image_y_feature_vectors, dtype=np.float32)

        target_probabilities = np.array(target_probabilities)

        image_x_feature_vectors = torch.tensor(image_x_feature_vectors).to(torch.float)
        image_y_feature_vectors = torch.tensor(image_y_feature_vectors).to(torch.float)

        target_probabilities = torch.tensor(target_probabilities).to(torch.float)

        # then concatenate
        image_x_feature_vectors = image_x_feature_vectors.reshape(len(image_x_feature_vectors), -1)
        image_y_feature_vectors = image_y_feature_vectors.reshape(len(image_y_feature_vectors), -1)

        if device is not None:
            image_x_feature_vectors = image_x_feature_vectors.to(device)
            image_y_feature_vectors = image_y_feature_vectors.to(device)
            target_probabilities = target_probabilities.to(device)

        return image_x_feature_vectors, image_y_feature_vectors, target_probabilities

    def get_validation_feature_vectors_and_target_linear(self, device=None):
        image_x_feature_vectors = []
        image_y_feature_vectors = []
        target_probabilities = []

        # get ab data
        for i in range(len(self.validation_image_pair_data_arr)):
            validation_image_pair_data = self.validation_image_pair_data_arr[i]
            image_x_feature_vector, image_y_feature_vector, target_probability = split_ab_data_vectors(
                validation_image_pair_data)
            image_x_feature_vectors.append(image_x_feature_vector)
            image_y_feature_vectors.append(image_y_feature_vector)
            target_probabilities.append(target_probability)

        image_x_feature_vectors = np.array(image_x_feature_vectors, dtype=np.float32)
        image_y_feature_vectors = np.array(image_y_feature_vectors, dtype=np.float32)
        target_probabilities = np.array(target_probabilities)

        image_x_feature_vectors = torch.tensor(image_x_feature_vectors).to(torch.float)
        image_y_feature_vectors = torch.tensor(image_y_feature_vectors).to(torch.float)
        target_probabilities = torch.tensor(target_probabilities).to(torch.float)

        # then concatenate
        image_x_feature_vectors = image_x_feature_vectors.reshape(len(image_x_feature_vectors), -1)
        image_y_feature_vectors = image_y_feature_vectors.reshape(len(image_y_feature_vectors), -1)
        print("feature shape after reshape=", image_x_feature_vectors.shape)

        if device is not None:
            image_x_feature_vectors = image_x_feature_vectors.to(device)
            image_y_feature_vectors = image_y_feature_vectors.to(device)
            target_probabilities = target_probabilities.to(device)

        return image_x_feature_vectors, image_y_feature_vectors, target_probabilities

    # ------------------------------- For Hyperparamter Search -------------------------------
    # ---------------------------------------- elm -------------------------------------------
    def get_len_training_ab_data_hyperparam(self):
        return len(self.training_dataset_paths_arr)

    def get_next_training_feature_vectors_and_target_hyperparam_elm(self, num_data, selection_datapoints_dict,
                                                                    features_dict, device=None):
        image_x_feature_vectors = []
        image_y_feature_vectors = []
        target_probabilities = []

        # get ab data
        for _ in range(num_data):
            dataset_path = self.training_dataset_paths_arr[self.current_training_data_index]
            image_pair = self.get_selection_datapoint_image_pair_hyperparameter(dataset_path, selection_datapoints_dict,
                                                                                features_dict)

            image_x_feature_vector, image_y_feature_vector, target_probability = split_ab_data_vectors(
                image_pair)
            image_x_feature_vectors.append(image_x_feature_vector)
            image_y_feature_vectors.append(image_y_feature_vector)
            target_probabilities.append(target_probability)
            self.current_training_data_index += 1

        image_x_feature_vectors = np.array(image_x_feature_vectors, dtype=np.float32)
        image_y_feature_vectors = np.array(image_y_feature_vectors, dtype=np.float32)

        target_probabilities = np.array(target_probabilities)

        image_x_feature_vectors = torch.tensor(image_x_feature_vectors).to(torch.float)
        image_y_feature_vectors = torch.tensor(image_y_feature_vectors).to(torch.float)

        target_probabilities = torch.tensor(target_probabilities).to(torch.float)

        # then concatenate
        image_x_feature_vectors = image_x_feature_vectors.reshape(len(image_x_feature_vectors), -1)
        image_y_feature_vectors = image_y_feature_vectors.reshape(len(image_y_feature_vectors), -1)

        if device is not None:
            image_x_feature_vectors = image_x_feature_vectors.to(device)
            image_y_feature_vectors = image_y_feature_vectors.to(device)
            target_probabilities = target_probabilities.to(device)

        return image_x_feature_vectors, image_y_feature_vectors, target_probabilities

    def get_validation_feature_vectors_and_target_hyperparam_elm(self, selection_datapoints_dict, features_dict,
                                                                 device=None):
        image_x_feature_vectors = []
        image_y_feature_vectors = []
        target_probabilities = []

        # get ab data
        for i in range(len(self.validation_dataset_paths_arr)):
            dataset_path = self.validation_dataset_paths_arr[i]
            image_pair = self.get_selection_datapoint_image_pair_hyperparameter(dataset_path, selection_datapoints_dict,
                                                                                features_dict)

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

        # then concatenate
        image_x_feature_vectors = image_x_feature_vectors.reshape(len(image_x_feature_vectors), -1)
        image_y_feature_vectors = image_y_feature_vectors.reshape(len(image_y_feature_vectors), -1)
        print("feature shape after reshape=", image_x_feature_vectors.shape)

        if device is not None:
            image_x_feature_vectors = image_x_feature_vectors.to(device)
            image_y_feature_vectors = image_y_feature_vectors.to(device)
            target_probabilities = target_probabilities.to(device)

        return image_x_feature_vectors, image_y_feature_vectors, target_probabilities

    def get_selection_datapoint_image_pair_hyperparameter(self, dataset, selection_datapoints_dict, features_dict):
        dataset_path = dataset[0]
        data_target = dataset[1]

        # load json object
        ab_data = selection_datapoints_dict[dataset_path]

        selected_image_index = ab_data.selected_image_index
        file_path_img_1 = ab_data.image_1_path
        file_path_img_2 = ab_data.image_2_path

        input_type_extension = "-text-embedding.msgpack"
        if self.input_type == constants.CLIP:
            input_type_extension = "_clip.msgpack"
        elif self.input_type == constants.KANDINSKY_CLIP:
            input_type_extension = "_clip_kandinsky.msgpack"
        elif self.input_type == constants.EMBEDDING:
            # replace with new /embeddings
            dataset_name = Path(file_path_img_1).parent.name
            file_path_img_1 = file_path_img_1.replace(dataset_name,
                                                      os.path.join(dataset_name, "embeddings/text-embedding"))
            
            dataset_name = Path(file_path_img_1).parent.name
            file_path_img_2 = file_path_img_2.replace(dataset_name,
                                                      os.path.join(dataset_name, "embeddings/text-embedding"))

            input_type_extension = "-text-embedding.msgpack"

            if self.pooling_strategy == constants.AVERAGE_POOLING:
                input_type_extension = "-text-embedding-average-pooled.msgpack"
            elif self.pooling_strategy == constants.MAX_POOLING:
                input_type_extension = "-text-embedding-max-pooled.msgpack"
            elif self.pooling_strategy == constants.MAX_ABS_POOLING:
                input_type_extension = "-text-embedding-signed-max-pooled.msgpack"

        features_path_img_1 = file_path_img_1.replace(".jpg", input_type_extension)
        features_path_img_1 = features_path_img_1.replace("datasets/", "")

        features_path_img_2 = file_path_img_2.replace(".jpg", input_type_extension)
        features_path_img_2 = features_path_img_2.replace("datasets/", "")

        features_img_1_data = features_dict[features_path_img_1]
        features_img_1_data = msgpack.unpackb(features_img_1_data)

        features_vector_img_1 = []
        if self.input_type in [constants.EMBEDDING, constants.EMBEDDING_POSITIVE]:
            features_vector_img_1.extend(features_img_1_data["positive_embedding"]["__ndarray__"])
        if self.input_type in [constants.EMBEDDING, constants.EMBEDDING_NEGATIVE]:
            features_vector_img_1.extend(features_img_1_data["negative_embedding"]["__ndarray__"])
        if self.input_type in [constants.CLIP, constants.KANDINSKY_CLIP]:
            features_vector_img_1.extend(features_img_1_data["clip-feature-vector"])

        features_vector_img_1 = np.array(features_vector_img_1)

        features_img_2_data = features_dict[features_path_img_2]
        features_img_2_data = msgpack.unpackb(features_img_2_data)

        features_vector_img_2 = []
        if self.input_type in [constants.EMBEDDING, constants.EMBEDDING_POSITIVE]:
            features_vector_img_2.extend(features_img_2_data["positive_embedding"]["__ndarray__"])
        if self.input_type in [constants.EMBEDDING, constants.EMBEDDING_NEGATIVE]:
            features_vector_img_2.extend(features_img_2_data["negative_embedding"]["__ndarray__"])
        if self.input_type in [constants.CLIP, constants.KANDINSKY_CLIP]:
            features_vector_img_2.extend(features_img_2_data["clip-feature-vector"])

        features_vector_img_2 = np.array(features_vector_img_2)

        # if image 1 is the selected
        if selected_image_index == 0:
            selected_features_vector = features_vector_img_1
            other_features_vector = features_vector_img_2

        # image 2 is selected
        else:
            selected_features_vector = features_vector_img_2
            other_features_vector = features_vector_img_1

        if data_target == 1.0:
            image_pair = (selected_features_vector, other_features_vector, [data_target])
        else:
            image_pair = (other_features_vector, selected_features_vector, [data_target])

        # add for training report
        if (self.image_selected_index_0_count + self.image_selected_index_1_count) < self.total_num_data:
            if selected_image_index == 0:
                self.image_selected_index_0_count += 1
            else:
                self.image_selected_index_1_count += 1

        return image_pair

    def load_dataset_hyperparameter(self, dataset_paths):
        start_time = time.time()

        print("# of dataset paths=", len(dataset_paths))
        if len(dataset_paths) == 0:
            raise Exception("No selection datapoints json found.")

        # calculate num validations
        num_validations = round((len(dataset_paths) * (1.0 - self.train_percent)))
        # get random index for validations
        validation_ab_data_list = []
        training_ab_data_list = []
        validation_indices = sample(range(0, len(dataset_paths) - 1), num_validations)
        for i in range(len(dataset_paths)):
            if i in validation_indices:
                validation_ab_data_list.append(dataset_paths[i])
            else:
                training_ab_data_list.append(dataset_paths[i])

        # training
        # duplicate each one
        # for target 1.0 and 0.0
        duplicated_training_list = []
        for path in training_ab_data_list:
            if (self.target_option == constants.TARGET_1_AND_0) or (self.target_option == constants.TARGET_1_ONLY):
                duplicated_training_list.append((path, 1.0))

            if (self.target_option == constants.TARGET_1_AND_0) or (self.target_option == constants.TARGET_0_ONLY):
                if (self.target_option == constants.TARGET_1_AND_0) and (
                        self.duplicate_flip_option == constants.DUPLICATE_AND_FLIP_RANDOM):
                    # then should have 50/50 chance of being duplicated or not
                    rand_int = choice([0, 1])
                    if rand_int == 1:
                        # then dont duplicate
                        continue

                duplicated_training_list.append((path, 0.0))

            # for test
            # if len(duplicated_training_list) >= 2:
            #     break

        # shuffle
        shuffled_training_list = []
        index_shuf = list(range(len(duplicated_training_list)))
        shuffle(index_shuf)
        for i in index_shuf:
            shuffled_training_list.append(duplicated_training_list[i])

        # validation
        # duplicate each one
        # for target 1.0 and 0.0
        duplicated_validation_list = []
        for path in validation_ab_data_list:
            if (self.target_option == constants.TARGET_1_AND_0) or (self.target_option == constants.TARGET_1_ONLY):
                duplicated_validation_list.append((path, 1.0))

            if (self.target_option == constants.TARGET_1_AND_0) or (self.target_option == constants.TARGET_0_ONLY):
                if (self.target_option == constants.TARGET_1_AND_0) and (
                        self.duplicate_flip_option == constants.DUPLICATE_AND_FLIP_RANDOM):
                    # then should have 50/50 chance of being duplicated or not
                    rand_int = choice([0, 1])
                    if rand_int == 1:
                        # then dont duplicate
                        continue
                duplicated_validation_list.append((path, 0.0))

            # for test
            # if len(duplicated_validation_list) >= 2:
            #     break

        # shuffle
        shuffled_validation_list = []
        index_shuf = list(range(len(duplicated_validation_list)))
        shuffle(index_shuf)
        for i in index_shuf:
            shuffled_validation_list.append(duplicated_validation_list[i])

        self.total_num_data = len(shuffled_training_list) + len(shuffled_validation_list)

        self.training_dataset_paths_arr = shuffled_training_list
        self.validation_dataset_paths_arr = shuffled_validation_list

        print("Dataset loaded...")
        print("Time elapsed: {0}s".format(format(time.time() - start_time, ".2f")))

    def shuffle_training_paths_hyperparam(self):
        print("Shuffling training data...")
        # shuffle
        shuffled_training_paths = []
        index_shuf = list(range(len(self.training_dataset_paths_arr)))
        shuffle(index_shuf)
        for i in index_shuf:
            shuffled_training_paths.append(self.training_dataset_paths_arr[i])

        self.training_dataset_paths_arr = shuffled_training_paths
