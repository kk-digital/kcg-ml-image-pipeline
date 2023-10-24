import os
import sys
import json
import argparse
import numpy as np
import torch.nn as nn
import math
import torch
import torch.optim as optim
import sklearn.metrics
from torchinfo import summary
from tqdm import tqdm
from ray import tune
from ray.air import session
from ray.tune.search.optuna import OptunaSearch

base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from utility.regression_utils import torchinfo_summary
from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel, forward_bradley_terry
from training_worker.ab_ranking.model.reports.ab_ranking_linear_train_report import get_train_report
from training_worker.ab_ranking.model.reports.graph_report_ab_ranking_linear import *
from training_worker.ab_ranking.model.ab_ranking_data_loader import ABRankingDatasetLoader
from training_worker.ab_ranking.model.reports.get_model_card import get_model_card_buf
from utility.minio import cmd
from training_worker.ab_ranking.model import constants
from training_worker.ab_ranking.script.hyperparameter_utils import get_data_dicts


def train_elm_v1_hyperparameter(model,
                                dataset_loader: ABRankingDatasetLoader,
                                training_batch_size=1,
                                epochs=100,
                                learning_rate=0.001,
                                weight_decay=0.01,
                                add_loss_penalty=False,
                                randomize_data_per_epoch=True):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # get validation data
    validation_features_x, \
        validation_features_y, \
        validation_targets = dataset_loader.get_validation_feature_vectors_and_target_linear()

    # get total number of training features
    num_features = dataset_loader.get_len_training_ab_data()

    # get number of batches to do per epoch
    training_num_batches = math.ceil(num_features / training_batch_size)
    for epoch in tqdm(range(epochs), desc="Training epoch"):
        training_loss_arr = []
        validation_loss_arr = []
        epoch_training_loss = None
        epoch_validation_loss = None

        # Only train after 0th epoch
        if epoch != 0:
            # fill data buffer
            dataset_loader.spawn_filling_workers()

            for i in range(training_num_batches):
                num_data_to_get = training_batch_size
                # last batch
                if i == training_num_batches - 1:
                    num_data_to_get = num_features - (i * (training_batch_size))

                batch_features_x_orig, \
                    batch_features_y_orig, \
                    batch_targets_orig = dataset_loader.get_next_training_feature_vectors_and_target_linear(
                    num_data_to_get)

                batch_features_x = batch_features_x_orig.clone().requires_grad_(True)
                batch_features_y = batch_features_y_orig.clone().requires_grad_(True)
                batch_targets = batch_targets_orig.clone().requires_grad_(True)

                with torch.no_grad():
                    predicted_score_images_y = model.forward(batch_features_y)

                optimizer.zero_grad()
                predicted_score_images_x = model.forward(batch_features_x)

                predicted_score_images_y_copy = predicted_score_images_y.clone().requires_grad_(True)
                batch_pred_probabilities = forward_bradley_terry(predicted_score_images_x,
                                                                      predicted_score_images_y_copy)

                loss = model.l1_loss(batch_pred_probabilities, batch_targets)

                if add_loss_penalty:
                    # add loss penalty
                    neg_score = torch.multiply(predicted_score_images_x, -1.0)
                    negative_score_loss_penalty = torch.relu(neg_score)
                    loss = torch.add(loss, negative_score_loss_penalty)

                loss.backward()
                optimizer.step()

                training_loss_arr.append(loss.detach().cpu())

            if randomize_data_per_epoch:
                dataset_loader.shuffle_training_data()

            # refill training ab data
            dataset_loader.fill_training_ab_data()

        # Calculate Validation Loss
        with torch.no_grad():
            for i in range(len(validation_features_x)):
                validation_feature_x = validation_features_x[i]
                validation_feature_x = validation_feature_x.unsqueeze(0)
                validation_feature_y = validation_features_y[i]
                validation_feature_y = validation_feature_y.unsqueeze(0)

                validation_target = validation_targets[i]
                validation_target = validation_target.unsqueeze(0)

                predicted_score_image_x = model.forward(validation_feature_x)
                with torch.no_grad():
                    predicted_score_image_y = model.forward(validation_feature_y)

                validation_pred_probabilities = forward_bradley_terry(predicted_score_image_x,
                                                                           predicted_score_image_y)


                validation_loss = model.l1_loss(validation_pred_probabilities, validation_target)

                if add_loss_penalty:
                    # add loss penalty
                    neg_score = torch.multiply(predicted_score_image_x, -1.0)
                    negative_score_loss_penalty = torch.relu(neg_score)
                    validation_loss = torch.add(validation_loss, negative_score_loss_penalty)

                validation_loss_arr.append(validation_loss.detach().cpu())

        # calculate epoch loss
        # epoch's training loss
        if len(training_loss_arr) != 0:
            training_loss_arr = torch.stack(training_loss_arr)
            epoch_training_loss = torch.mean(training_loss_arr)

        # epoch's validation loss
        validation_loss_arr = torch.stack(validation_loss_arr)
        epoch_validation_loss = torch.mean(validation_loss_arr)

        if epoch_training_loss is None:
            epoch_training_loss = epoch_validation_loss

        session.report({"training-loss": epoch_training_loss, "validation-loss": epoch_validation_loss})

    print("Training Finished.")


def train_hyperparameter_search(config,
                                dataset_paths,
                                selection_datapoints_dict,
                                embeddings_dict):
    input_shape = 2 * 768

    epochs = config["epochs"]
    num_random_layers = config["num_random_layers"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    pooling_strategy = config["pooling_strategy"]
    normalize_vectors = config["normalize_vectors"]
    target_option = config["target_option"]
    duplicate_flip_option = config["duplicate_flip_option"]

    # load dataset
    dataset_loader = ABRankingDatasetLoader(dataset_name="",
                                            load_to_ram=True,
                                            pooling_strategy=pooling_strategy,
                                            normalize_vectors=normalize_vectors,
                                            target_option=target_option,
                                            duplicate_flip_option=duplicate_flip_option)

    dataset_loader.load_dataset_hyperparameter(dataset_paths=dataset_paths,
                                               selection_datapoints_dict=selection_datapoints_dict,
                                               embeddings_dict=embeddings_dict)

    # initialize model
    ab_model = ABRankingELMModel(inputs_shape=input_shape,
                                 num_random_layers=num_random_layers)

    # do training
    train_elm_v1_hyperparameter(model=ab_model,
                                dataset_loader=dataset_loader,
                                training_batch_size=1,
                                epochs=epochs,
                                learning_rate=learning_rate,
                                weight_decay=weight_decay)


def do_search(minio_access_key, minio_secret_key, dataset_name):
    # get data first
    dataset_paths, selection_datapoints_dict, embeddings_dict = get_data_dicts(minio_access_key=minio_access_key,
                                                                               minio_secret_key=minio_secret_key,
                                                                               dataset_name=dataset_name)

    search_space = {
        "epochs": tune.grid_search([5,
                                    6,
                                    7,
                                    8]),
        "num_random_layers": tune.grid_search([0,
                                               1,
                                               2,
                                               3]),
        "learning_rate": tune.grid_search([1.0,
                                           0.5,
                                           0.1,
                                           0.05,
                                           0.01,
                                           0.005,
                                           0.001,
                                           0.0001,
                                           0.0005]),
        "weight_decay": tune.grid_search([ 0.0,
                                           0.1,
                                           0.01,
                                           0.001,
                                           0.0001]),
        "pooling_strategy": tune.grid_search([constants.AVERAGE_POOLING, constants.MAX_POOLING]),
        "normalize_vectors": tune.grid_search([True, False]),
        "target_option": tune.grid_search([constants.TARGET_1_AND_0, constants.TARGET_1_ONLY, constants.TARGET_0_ONLY]),
        "duplicate_flip_option": tune.grid_search([constants.DUPLICATE_AND_FLIP_ALL, constants.DUPLICATE_AND_FLIP_RANDOM]),
    }

    trainable_with_cpu_gpu = tune.with_resources(train_hyperparameter_search, {"cpu": 1})
    tuner = tune.Tuner(tune.with_parameters(trainable_with_cpu_gpu,
                                            dataset_paths=dataset_paths,
                                            selection_datapoints_dict=selection_datapoints_dict,
                                            embeddings_dict=embeddings_dict),
                       tune_config=tune.TuneConfig(
                           metric="training-loss",
                           mode="min"
                       ),
                       param_space=search_space)

    results = tuner.fit()
    best_result_config = results.get_best_result(metric="score", mode="min").config

    print(best_result_config)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search. Searches optimum parameters to use")
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")
    subparsers.required = True

    # Subparser for 'search' command
    train_parser = subparsers.add_parser("search", help="Search optimum parameters for the model")
    train_parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    train_parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    train_parser.add_argument('--dataset-name', type=str, help='The dataset name to use for training',
                              default='environmental')

    return parser.parse_args()


def main():
    args = parse_arguments()

    command = args.subcommand

    if command == 'search':
        do_search(args.minio_access_key, args.minio_secret_key, args.dataset_name)


if __name__ == '__main__':
    main()
