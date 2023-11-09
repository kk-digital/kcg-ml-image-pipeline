import os
import torch
import sys
from datetime import datetime
from pytz import timezone
import numpy as np
from xgboost import XGBRegressor
import time

base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from utility.regression_utils import torchinfo_summary
from training_worker.ab_ranking.model.ab_ranking_linear import ABRankingModel
from training_worker.ab_ranking.model.reports.ab_ranking_linear_train_report import get_train_report
from training_worker.ab_ranking.model.reports.graph_report_ab_ranking_linear import *
from training_worker.ab_ranking.model.ab_ranking_data_loader import ABRankingDatasetLoader
from training_worker.ab_ranking.model.reports.get_model_card import get_model_card_buf
from utility.minio import cmd
from training_worker.ab_ranking.model import constants
from training_worker.ab_ranking.model.reports import upload_score_residual

import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error

def train_xgboost(dataset_name: str,
                  minio_ip_addr=None,
                  minio_access_key=None,
                  minio_secret_key=None,
                  input_type="clip",
                  epochs=10000,
                  learning_rate=0.05,
                  buffer_size=20000,
                  train_percent=0.9,
                  training_batch_size=1,
                  weight_decay=0.00,
                  load_data_to_ram=False,
                  debug_asserts=False,
                  normalize_vectors=True,
                  pooling_strategy=constants.AVERAGE_POOLING,
                  add_loss_penalty=True,
                  target_option=constants.TARGET_1_AND_0,
                  duplicate_flip_option=constants.DUPLICATE_AND_FLIP_ALL,
                  randomize_data_per_epoch=True,
                  ):
    # raise exception if input is not clip
    if input_type not in ["clip", "embedding"]:
        raise Exception("Only 'clip' and 'embedding' is supported for now.")

    date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%Y-%m-%d')
    print("Current datetime: {}".format(datetime.now(tz=timezone("Asia/Hong_Kong"))))
    bucket_name = "datasets"
    training_dataset_path = os.path.join(bucket_name, dataset_name)
    network_type = "xgboost"
    output_type = "score"
    output_path = "{}/models/ranking".format(dataset_name)

    # check input type
    if input_type not in constants.ALLOWED_INPUT_TYPES:
        raise Exception("input type is not supported: {}".format(input_type))

    input_shape = 2 * 768
    if input_type in [constants.EMBEDDING_POSITIVE, constants.EMBEDDING_NEGATIVE, constants.CLIP]:
        input_shape = 768

    # load dataset
    dataset_loader = ABRankingDatasetLoader(dataset_name=dataset_name,
                                            minio_ip_addr=minio_ip_addr,
                                            minio_access_key=minio_access_key,
                                            minio_secret_key=minio_secret_key,
                                            input_type=input_type,
                                            buffer_size=buffer_size,
                                            train_percent=train_percent,
                                            load_to_ram=load_data_to_ram,
                                            pooling_strategy=pooling_strategy,
                                            normalize_vectors=normalize_vectors,
                                            target_option=target_option,
                                            duplicate_flip_option=duplicate_flip_option)
    dataset_loader.load_dataset()

    # get final filename
    sequence = 0
    # if exist, increment sequence
    while True:
        filename = "{}-{:02}-{}-{}-{}".format(date_now, sequence, output_type, network_type, input_type)
        exists = cmd.is_object_exists(dataset_loader.minio_client, bucket_name,
                                      os.path.join(output_path, filename + ".pth"))
        if not exists:
            break

        sequence += 1

    training_total_size = dataset_loader.get_len_training_ab_data()
    validation_total_size = dataset_loader.get_len_validation_ab_data()

    # get training data and convert to numpy array
    training_features_x, \
        training_features_y, \
        training_targets = dataset_loader.get_next_training_feature_vectors_and_target_linear(training_total_size)

    training_features_x = training_features_x.numpy()
    training_features_y = training_features_y.numpy()
    training_targets = training_targets.numpy()

    training_data = []
    for i in range(len(training_features_x)):
        concatenated = np.concatenate((training_features_x[i], training_features_y[i]))
        training_data.append(concatenated)


    training_data = np.asarray(training_data)
    print("training data shape=", training_data.shape)
    print("training target shape=", training_targets.shape)

    # get validation data
    validation_features_x, \
        validation_features_y, \
        validation_targets = dataset_loader.get_validation_feature_vectors_and_target_linear()
    validation_features_x = validation_features_x.numpy()
    validation_features_y = validation_features_y.numpy()
    validation_targets = validation_targets.numpy()
    validation_data = []
    for i in range(len(validation_features_x)):
            concatenated = np.concatenate((validation_features_x[i], validation_features_y[i]))
            validation_data.append(concatenated)


    validation_data = np.asarray(validation_data)
    print("validation data shape=", validation_data.shape)
    print("validation target shape=", validation_targets.shape)

    # Create regression matrices
    dtrain_reg = xgb.DMatrix(training_data, training_targets)
    dtest_reg = xgb.DMatrix(validation_data, validation_targets)

    # Define hyperparameters
    params = {"objective": "reg:squarederror", "device": "cuda:0"}
    evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]
    n = 500
    xgboost_model = xgb.train(params=params,
                      dtrain=dtrain_reg,
                      num_boost_round=n,
                      evals=evals,
                      verbose_eval=10,  # Every ten rounds
                      early_stopping_rounds=50,  # Activate early stopping
                      )

    # make a prediction
    training_pred_results = xgboost_model.predict(dtrain_reg)
    validation_pred_results = xgboost_model.predict(dtest_reg)

    ab_model = ABRankingModel(inputs_shape=input_shape)
    training_predicted_score_images_x = training_pred_results
    training_predicted_score_images_y = training_pred_results
    training_predicted_probabilities = training_pred_results
    training_target_probabilities = training_targets
    validation_predicted_score_images_x = validation_pred_results
    validation_predicted_score_images_y = validation_pred_results
    validation_predicted_probabilities = validation_pred_results
    validation_target_probabilities = validation_targets
    training_loss_per_epoch = [0] * epochs
    validation_loss_per_epoch = [0] * epochs

    # data for chronological score graph
    training_shuffled_indices_origin = []
    for index in dataset_loader.training_data_paths_indices_shuffled:
        training_shuffled_indices_origin.append(index)

    validation_shuffled_indices_origin = []
    for index in dataset_loader.validation_data_paths_indices_shuffled:
        validation_shuffled_indices_origin.append(index)

    # Upload model to minio
    model_name = "{}.pth".format(filename)
    model_output_path = os.path.join(output_path, model_name)
    xgboost_model_buf = xgboost_model.save_raw(raw_format='json')
    buffer = BytesIO(xgboost_model_buf)
    buffer.seek(0)

    cmd.upload_data(dataset_loader.minio_client, "datasets", model_output_path, buffer)

    train_sum_correct = 0
    validation_sum_correct = 0

    # show and save graph
    graph_name = "{}.png".format(filename)
    graph_output_path = os.path.join(output_path, graph_name)

    graph_buffer = get_graph_report(ab_model,
                                    training_predicted_probabilities,
                                    training_target_probabilities,
                                    validation_predicted_probabilities,
                                    validation_target_probabilities,
                                    training_predicted_score_images_x,
                                    training_predicted_score_images_y,
                                    validation_predicted_score_images_x,
                                    validation_predicted_score_images_y,
                                    training_total_size,
                                    validation_total_size,
                                    training_loss_per_epoch,
                                    validation_loss_per_epoch,
                                    epochs,
                                    learning_rate,
                                    training_batch_size,
                                    weight_decay,
                                    date_now,
                                    network_type,
                                    input_type,
                                    input_shape,
                                    output_type,
                                    train_sum_correct,
                                    validation_sum_correct,
                                    "",
                                    dataset_name,
                                    pooling_strategy,
                                    normalize_vectors,
                                    -1,
                                    add_loss_penalty,
                                    target_option,
                                    duplicate_flip_option,
                                    randomize_data_per_epoch,
                                    -1,
                                    training_shuffled_indices_origin,
                                    validation_shuffled_indices_origin,
                                    dataset_loader.total_selection_datapoints)

    # upload the graph report
    cmd.upload_data(dataset_loader.minio_client, bucket_name, graph_output_path, graph_buffer)


if __name__ == '__main__':
    start_time = time.time()

    train_xgboost(dataset_name="environmental",
                  minio_ip_addr=None,  # will use default if none is given
                  minio_access_key="nkjYl5jO4QnpxQU0k0M1",
                  minio_secret_key="MYtmJ9jhdlyYx3T1McYy4Z0HB3FkxjmITXLEPKA1",
                  input_type="embedding",
                  )

    time_elapsed = time.time() - start_time
    print("Time elapsed: {0}s".format(format(time_elapsed, ".2f")))
