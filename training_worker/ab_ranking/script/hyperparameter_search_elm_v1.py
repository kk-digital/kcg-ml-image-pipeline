import os
import sys
import argparse
import math
import torch.optim as optim
from tqdm import tqdm
from ray import tune
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.bayesopt import BayesOptSearch
from datetime import datetime
from pytz import timezone


base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel, forward_bradley_terry
from training_worker.ab_ranking.model.reports.graph_report_ab_ranking import *
from data_loader.ab_ranking_dataset_loader import ABRankingDatasetLoader
from training_worker.ab_ranking.model import constants
from training_worker.ab_ranking.script.hyperparameter_utils import get_data_dicts, get_performance_graph_report
from utility.minio import cmd


def train_elm_v1_hyperparameter(model,
                                dataset_loader: ABRankingDatasetLoader,
                                training_batch_size=1,
                                epochs=100,
                                learning_rate=0.05,
                                weight_decay=0.01,
                                add_loss_penalty=False,
                                randomize_data_per_epoch=True,
                                selection_datapoints_dict=None,
                                features_dict=None):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # get validation data
    validation_features_x, \
        validation_features_y, \
        validation_targets = dataset_loader.get_validation_feature_vectors_and_target_hyperparam_elm(selection_datapoints_dict=selection_datapoints_dict,
                                                                                                     features_dict=features_dict)

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
            for i in range(training_num_batches):
                num_data_to_get = training_batch_size
                # last batch
                if i == training_num_batches - 1:
                    num_data_to_get = num_features - (i * (training_batch_size))

                batch_features_x_orig, \
                    batch_features_y_orig, \
                    batch_targets_orig = dataset_loader.get_next_training_feature_vectors_and_target_hyperparam_elm(num_data_to_get,
                                                                                                                    selection_datapoints_dict=selection_datapoints_dict,
                                                                                                                    features_dict=features_dict)

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
                dataset_loader.shuffle_training_paths_hyperparam()

            # reset current index
            dataset_loader.current_training_data_index = 0

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

        session.report({"training-loss": epoch_training_loss.item(), "validation-loss": epoch_validation_loss.item()})

    print("Training Finished.")


def train_hyperparameter_search(config,
                                dataset_paths,
                                selection_datapoints_dict,
                                features_dict):

    epochs = config["epochs"]
    num_random_layers = config["num_random_layers"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    add_loss_penalty = config["add_loss_penalty"]
    randomize_data_per_epoch= config["randomize_data_per_epoch"]
    pooling_strategy = config["pooling_strategy"]
    normalize_vectors = config["normalize_vectors"]
    target_option = config["target_option"]
    duplicate_flip_option = config["duplicate_flip_option"]
    elm_sparsity = config["elm_sparsity"]
    input_type = config["input_type"]

    input_shape = 2 * 768
    if input_type==constants.CLIP:
        input_shape = 768

    # load dataset
    dataset_loader = ABRankingDatasetLoader(dataset_name="",
                                            load_to_ram=True,
                                            pooling_strategy=pooling_strategy,
                                            normalize_vectors=normalize_vectors,
                                            target_option=target_option,
                                            duplicate_flip_option=duplicate_flip_option,
                                            input_type=input_type)

    dataset_loader.load_dataset_hyperparameter(dataset_paths=dataset_paths)

    # initialize model
    ab_model = ABRankingELMModel(inputs_shape=input_shape,
                                 num_random_layers=num_random_layers,
                                 elm_sparsity=elm_sparsity)

    # do training
    train_elm_v1_hyperparameter(model=ab_model.model,
                                dataset_loader=dataset_loader,
                                training_batch_size=1,
                                epochs=epochs,
                                learning_rate=learning_rate,
                                weight_decay=weight_decay,
                                add_loss_penalty=add_loss_penalty,
                                randomize_data_per_epoch=randomize_data_per_epoch,
                                selection_datapoints_dict=selection_datapoints_dict,
                                features_dict=features_dict)


def do_search(minio_access_key, minio_secret_key, dataset_name, input_type, num_samples):
    # get minio client
    minio_client = cmd.get_minio_client(minio_access_key=minio_access_key,
                                        minio_secret_key=minio_secret_key)

    # get data first
    dataset_paths, selection_datapoints_dict, features_dict = get_data_dicts(minio_client=minio_client,
                                                                               dataset_name=dataset_name,
                                                                               input_type=input_type)

    search_space = {
        "input_type": input_type,
        "epochs": 8,
        "num_random_layers": 1,
        "learning_rate": 0.5,
        "weight_decay": 0.0,
        "elm_sparsity": 0.5,
        "add_loss_penalty": True,
        "randomize_data_per_epoch": True,
        "pooling_strategy": constants.AVERAGE_POOLING,
        "normalize_vectors": True,
        "target_option": constants.TARGET_1_AND_0,
        "duplicate_flip_option": constants.DUPLICATE_AND_FLIP_ALL
        }

    bayesian_opt = BayesOptSearch(utility_kwargs={"kind": "ucb", "kappa": 1.9, "xi": 0.0})

    trainable_with_cpu_gpu = tune.with_resources(train_hyperparameter_search, {"cpu": 1})
    tuner = tune.Tuner(tune.with_parameters(trainable_with_cpu_gpu,
                                            dataset_paths=dataset_paths,
                                            selection_datapoints_dict=selection_datapoints_dict,
                                            features_dict=features_dict),
                       tune_config=tune.TuneConfig(
                           metric="validation-loss",
                           mode="min",
                           search_alg=bayesian_opt,
                           num_samples=num_samples,
                       ),
                       param_space=search_space)

    results = tuner.fit()
    best_result_config = results.get_best_result(metric="validation-loss", mode="min").config

    print(best_result_config)

    # get final filename
    date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%Y-%m-%d')
    print("Current datetime: {}".format(datetime.now(tz=timezone("Asia/Hong_Kong"))))
    bucket_name = "datasets"
    network_type = "elm-v1-hyperparam"
    output_type = "score"
    output_path = "{}/models/ranking".format(dataset_name)
    sequence = 0
    # if exist, increment sequence
    while True:
        filename = "{}-{:02}-{}-{}-{}".format(date_now, sequence, output_type, network_type, input_type)
        exists = cmd.is_object_exists(minio_client, bucket_name,
                                      os.path.join(output_path, filename + ".pth"))
        if not exists:
            break

        sequence += 1
    graph_name = "{}-performance.png".format(filename)
    graph_output_path = os.path.join(output_path, graph_name)
    performance_graph = get_performance_graph_report(results)
    cmd.upload_data(minio_client, bucket_name, graph_output_path, performance_graph)


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
    train_parser.add_argument('--input-type', type=str, help='The input type to use',
                              default='embedding')
    train_parser.add_argument('--num-samples', type=int, help='The number of samples to do',
                              default=100)

    return parser.parse_args()


def main():
    args = parse_arguments()

    command = args.subcommand

    if command == 'search':
        do_search(args.minio_access_key, args.minio_secret_key, args.dataset_name, args.input_type, args.num_samples)


if __name__ == '__main__':
    main()
