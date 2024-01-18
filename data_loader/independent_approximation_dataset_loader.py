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
import scipy as sp
base_directory = "./"
sys.path.insert(0, base_directory)

from utility.minio import cmd
from training_worker.ab_ranking.model import constants
from data_loader.ab_data import ABData
from data_loader.utils import get_datasets, get_aggregated_selection_datapoints, get_object, split_ab_data_vectors
from data_loader.generated_image_data import GeneratedImageData
from data_loader.phrase_vector_loader import PhraseVectorLoader

class IndependentApproximationDatasetLoader:
    def __init__(self,
                 dataset_name,
                 minio_ip_addr=None,
                 minio_access_key=None,
                 minio_secret_key=None,
                 train_percent=0.9,
                 phrase_vector_loader: PhraseVectorLoader =None,
                 target_option=constants.TARGET_1_AND_0,
                 duplicate_flip_option=constants.DUPLICATE_AND_FLIP_ALL,
                 input_type="positive",
                 ):
        self.dataset_name = dataset_name
        self.phrase_vector_loader = phrase_vector_loader
        self.input_type = input_type

        if minio_access_key is not None:
            self.minio_access_key = minio_access_key
            self.minio_secret_key = minio_secret_key
            self.minio_client = cmd.get_minio_client(minio_access_key=self.minio_access_key,
                                                     minio_secret_key=self.minio_secret_key,
                                                     minio_ip_addr=minio_ip_addr)

        # config
        self.train_percent = train_percent
        self.total_selection_datapoints = 0
        self.training_data_total = 0
        self.validation_data_total = 0
        self.total_num_data = 0
        self.target_option = target_option
        self.duplicate_flip_option = duplicate_flip_option

        # load all data to ram
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

        dataset_list = get_datasets(self.minio_client)
        if self.dataset_name not in dataset_list:
            raise Exception("Dataset is not in minio server")

        # if exist then get paths for aggregated selection datapoints
        dataset = get_aggregated_selection_datapoints(self.minio_client, self.dataset_name)
        len_dataset = len(dataset)
        print("# of dataset retrieved=", len_dataset)
        if len(dataset) == 0:
            raise Exception("No selection datapoints json found.")

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

    def get_len_training_ab_data(self):
        return self.training_data_total

    def get_len_validation_ab_data(self):
        return self.validation_data_total

    def get_image_selected_index_data(self):
        selected_index_0_count = self.image_selected_index_0_count
        selected_index_1_count = self.image_selected_index_1_count
        total_count = selected_index_0_count + selected_index_1_count

        return selected_index_0_count, selected_index_1_count, total_count

    def get_phrase_vector_of_image_pair(self, image_pair, index):
        prompt_img_1 = image_pair[0]
        prompt_img_2 = image_pair[1]
        prompt_target = image_pair[2]

        phrase_vector_img_1 = self.phrase_vector_loader.get_phrase_vector(prompt_img_1, input_type=self.input_type)
        phrase_vector_img_1 = np.fromiter(phrase_vector_img_1, dtype=bool)
        phrase_vector_img_1 = sp.sparse.coo_array(phrase_vector_img_1)

        phrase_vector_img_2 = self.phrase_vector_loader.get_phrase_vector(prompt_img_2, input_type=self.input_type)
        phrase_vector_img_2 = np.fromiter(phrase_vector_img_2, dtype=bool)
        phrase_vector_img_2 = sp.sparse.coo_array(phrase_vector_img_2)

        new_image_pair = (phrase_vector_img_1, phrase_vector_img_2, prompt_target)

        return new_image_pair, index

    def convert_training_data_to_phrase_vector_pairs(self):
        print("Converting training data to phrase vector pairs...")
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []

            count = 0
            for pair in self.training_image_pair_data_arr:
                futures.append(executor.submit(self.get_phrase_vector_of_image_pair, image_pair=pair, index=count))
                count += 1

            for future in tqdm(as_completed(futures), total=len(futures)):
                new_image_pair, index  = future.result()
                self.training_image_pair_data_arr[index] = new_image_pair

    def convert_validation_data_to_phrase_vector_pairs(self):
        print("Converting validation data to phrase vector pairs...")
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []

            count = 0
            for pair in self.validation_image_pair_data_arr:
                futures.append(executor.submit(self.get_phrase_vector_of_image_pair, image_pair=pair, index=count))
                count += 1

            for future in tqdm(as_completed(futures), total=len(futures)):
                new_image_pair, index  = future.result()
                self.validation_image_pair_data_arr[index] = new_image_pair


    def get_selection_datapoint_image_pair(self, dataset, index=0):
        image_pairs = []
        ab_data = dataset

        selected_image_index = ab_data.selected_image_index
        file_path_img_1 = ab_data.image_1_path
        file_path_img_2 = ab_data.image_2_path

        input_type_extension = "_data.msgpack"

        # get .msgpack data
        features_path_img_1 = file_path_img_1.replace(".jpg", input_type_extension)
        features_path_img_1 = features_path_img_1.replace("datasets/", "")

        features_path_img_2 = file_path_img_2.replace(".jpg", input_type_extension)
        features_path_img_2 = features_path_img_2.replace("datasets/", "")

        features_img_1_data = get_object(self.minio_client, features_path_img_1)
        generated_image_data_1 = GeneratedImageData.from_msgpack_string(features_img_1_data)
        if self.input_type == "positive":
            prompt_img_1 = generated_image_data_1.positive_prompt
        else:
            prompt_img_1 = generated_image_data_1.negative_prompt

        features_vector_img_1 = prompt_img_1

        features_img_2_data = get_object(self.minio_client, features_path_img_2)
        generated_image_data_2 = GeneratedImageData.from_msgpack_string(features_img_2_data)
        if self.input_type == "positive":
            prompt_img_2 = generated_image_data_2.positive_prompt
        else:
            prompt_img_2 = generated_image_data_2.negative_prompt

        features_vector_img_2 = prompt_img_2

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

        self.convert_training_data_to_phrase_vector_pairs()

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

        self.convert_validation_data_to_phrase_vector_pairs()

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

    # ------------------------------- For AB Ranking Linear -------------------------------
    def get_next_training_feature_vectors_and_target(self, num_data, device=None):
        image_x_feature_vectors = []
        image_y_feature_vectors = []
        target_probabilities = []

        for _ in range(num_data):
            training_image_pair_data = self.training_image_pair_data_arr[self.current_training_data_index]
            image_x_feature_vector, image_y_feature_vector, target_probability = split_ab_data_vectors(
                training_image_pair_data)

            # since we're using sparsed tensor
            # we have to convert input to to_dense first
            image_x_feature_vector = image_x_feature_vector.todense()
            image_y_feature_vector = image_y_feature_vector.todense()

            image_x_feature_vectors.append(image_x_feature_vector)
            image_y_feature_vectors.append(image_y_feature_vector)
            target_probabilities.append(target_probability)
            self.current_training_data_index += 1

        image_x_feature_vectors = np.array(image_x_feature_vectors)
        image_y_feature_vectors = np.array(image_y_feature_vectors)

        image_x_feature_vectors = torch.tensor(image_x_feature_vectors).to(torch.bool)
        image_y_feature_vectors = torch.tensor(image_y_feature_vectors).to(torch.bool)

        target_probabilities = torch.tensor(target_probabilities).to(torch.float)

        image_x_feature_vectors = image_x_feature_vectors.squeeze(1)
        image_y_feature_vectors = image_y_feature_vectors.squeeze(1)

        if device is not None:
            image_x_feature_vectors = image_x_feature_vectors.to(device)
            image_y_feature_vectors = image_y_feature_vectors.to(device)
            target_probabilities = target_probabilities.to(device)

        return image_x_feature_vectors, image_y_feature_vectors, target_probabilities

    def get_validation_feature_vectors_and_target(self, device=None):
        image_x_feature_vectors = []
        image_y_feature_vectors = []
        target_probabilities = []

        # get ab data
        for i in range(len(self.validation_image_pair_data_arr)):
            validation_image_pair_data = self.validation_image_pair_data_arr[i]
            image_x_feature_vector, image_y_feature_vector, target_probability = split_ab_data_vectors(
                validation_image_pair_data)

            # since we're using sparsed tensor
            # we have to convert input to to_dense first
            image_x_feature_vector = image_x_feature_vector.todense()
            image_y_feature_vector = image_y_feature_vector.todense()

            image_x_feature_vectors.append(image_x_feature_vector)
            image_y_feature_vectors.append(image_y_feature_vector)

            target_probabilities.append(target_probability)

        image_x_feature_vectors = np.array(image_x_feature_vectors)
        image_y_feature_vectors = np.array(image_y_feature_vectors)

        image_x_feature_vectors = torch.tensor(image_x_feature_vectors).to(torch.bool)
        image_y_feature_vectors = torch.tensor(image_y_feature_vectors).to(torch.bool)

        target_probabilities = torch.tensor(target_probabilities).to(torch.float)

        image_x_feature_vectors = image_x_feature_vectors.squeeze(1)
        image_y_feature_vectors = image_y_feature_vectors.squeeze(1)
        print("feature shape=", image_x_feature_vectors.shape)

        # if device is not None:
        #     image_x_feature_vectors = image_x_feature_vectors.to(device)
        #     image_y_feature_vectors = image_y_feature_vectors.to(device)
        #     target_probabilities = target_probabilities.to(device)

        return image_x_feature_vectors, image_y_feature_vectors, target_probabilities

