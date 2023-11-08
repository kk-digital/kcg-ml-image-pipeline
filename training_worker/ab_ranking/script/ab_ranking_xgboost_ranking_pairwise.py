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

def np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def forward_bradley_terry(predicted_score_images_x, predicted_score_images_y, use_sigmoid=True):
    if use_sigmoid:
        # scale the score
        # scaled_score_image_x = torch.multiply(1000.0, predicted_score_images_x)
        # scaled_score_image_y = torch.multiply(1000.0, predicted_score_images_y)

        # prob = sigmoid( (x-y) / 100 )
        diff_predicted_score = np.subtract(predicted_score_images_x, predicted_score_images_y)
        res_predicted_score = np.divide(diff_predicted_score, 50.0)
        pred_probabilities = np_sigmoid(res_predicted_score)
    else:
        epsilon = 0.000001

        # if score is negative N, make it 0
        # predicted_score_images_x = torch.max(predicted_score_images_x, torch.tensor([0.], device=self._device))
        # predicted_score_images_y = torch.max(predicted_score_images_y, torch.tensor([0.], device=self._device))

        # Calculate probability using Bradley Terry Formula: P(x>y) = score(x) / ( Score(x) + score(y))
        sum_predicted_score = np.add(predicted_score_images_x, predicted_score_images_y)
        sum_predicted_score = np.add(sum_predicted_score, epsilon)
        pred_probabilities = np.divide(predicted_score_images_x, sum_predicted_score)

    return pred_probabilities

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
    network_type = "xgboost-rank-pairwise"
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

    training_data = training_features_x

    print("training data shape=", training_data.shape)
    print("training target shape=", training_targets.shape)

    # get validation data
    validation_features_x, \
        validation_features_y, \
        validation_targets = dataset_loader.get_validation_feature_vectors_and_target_linear()
    validation_features_x = validation_features_x.numpy()
    validation_features_y = validation_features_y.numpy()
    validation_targets = validation_targets.numpy()

    validation_data = validation_features_x

    print("validation data shape=", validation_data.shape)
    print("validation target shape=", validation_targets.shape)

    # Create regression matrices
    dtrain_reg = xgb.DMatrix(training_data, training_targets)
    dtest_reg = xgb.DMatrix(validation_data, validation_targets)

    train_len= int(len(training_targets)/2)
    validate_len = int(len(validation_targets)/2)

    # group
    train_group = np.array([2 for _ in range(train_len)])
    validation_group = np.array([2 for _ in range(validate_len)])

    dtrain_reg.set_group(train_group)
    dtest_reg.set_group(validation_group)

    params = {'objective': 'rank:pairwise', 'eta': 0.1, 'gamma': 1.0,
              'min_child_weight': 0.1, 'max_depth': 6}

    evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]
    n = 500
    xgboost_model = xgb.train(params,
                              dtrain_reg,
                              num_boost_round=n,
                              evals=evals,
                              verbose_eval=10,  # Every ten rounds
                              early_stopping_rounds=50,  # Activate early stopping
                              )

    # make a prediction
    # training_pred_scores_results = xgboost_model.predict(dtrain_reg)
    # validation_pred_scores_results = xgboost_model.predict(dtest_reg)

    # get training predicted probability
    dtrain_x = xgb.DMatrix(training_features_x)
    dtrain_y = xgb.DMatrix(training_features_y)
    train_x_pred_scores = xgboost_model.predict(dtrain_x)
    train_y_pred_scores = xgboost_model.predict(dtrain_y)

    # add const
    train_x_pred_scores = train_x_pred_scores + 10
    train_y_pred_scores = train_y_pred_scores + 10

    train_pred_prob = []
    for i in range(len(train_x_pred_scores)):
        prob = forward_bradley_terry(train_x_pred_scores[i], train_y_pred_scores[i])
        train_pred_prob.append(prob)

    # get validation predicted probability
    dvalidation_x = xgb.DMatrix(validation_features_x)
    dvalidation_y = xgb.DMatrix(validation_features_y)
    validation_x_pred_scores = xgboost_model.predict(dvalidation_x)
    validation_y_pred_scores = xgboost_model.predict(dvalidation_y)

    # add const
    validation_x_pred_scores = validation_x_pred_scores + 10
    validation_y_pred_scores = validation_y_pred_scores + 10
    validation_pred_prob = []
    for i in range(len(validation_x_pred_scores)):
        prob = forward_bradley_terry(validation_x_pred_scores[i], validation_y_pred_scores[i])
        validation_pred_prob.append(prob)

    ab_model = ABRankingModel(inputs_shape=input_shape)
    training_predicted_score_images_x = train_x_pred_scores
    training_predicted_score_images_y = train_y_pred_scores
    training_predicted_probabilities = train_pred_prob
    training_target_probabilities = training_targets
    validation_predicted_score_images_x = validation_x_pred_scores
    validation_predicted_score_images_y = validation_y_pred_scores
    validation_predicted_probabilities = validation_pred_prob
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
    for i in range(len(training_target_probabilities)):
        if training_target_probabilities[i] == [1.0]:
            if training_predicted_score_images_x[i] > training_predicted_score_images_y[i]:
                train_sum_correct += 1
        else:
            if training_predicted_score_images_x[i] < training_predicted_score_images_y[i]:
                train_sum_correct += 1

    validation_sum_correct = 0
    for i in range(len(validation_target_probabilities)):
        if validation_target_probabilities[i] == [1.0]:
            if validation_predicted_score_images_x[i] > validation_predicted_score_images_y[i]:
                validation_sum_correct += 1
        else:
            if validation_predicted_score_images_x[i] < validation_predicted_score_images_y[i]:
                validation_sum_correct += 1

    selected_index_0_count, selected_index_1_count, total_images_count = dataset_loader.get_image_selected_index_data()
    # # save report
    # report_str = get_train_report(ab_model,
    #                               training_dataset_path,
    #                               train_percent,
    #                               training_total_size,
    #                               validation_total_size,
    #                               train_sum_correct,
    #                               validation_sum_correct,
    #                               nn_summary,
    #                               training_predicted_score_images_x,
    #                               training_predicted_score_images_y,
    #                               validation_predicted_score_images_x,
    #                               validation_predicted_score_images_y,
    #                               training_batch_size,
    #                               learning_rate,
    #                               weight_decay,
    #                               selected_index_0_count,
    #                               selected_index_1_count,
    #                               total_images_count,
    #                               dataset_loader.datapoints_per_sec)
    #
    # # Upload model to minio
    # report_name = "{}.txt".format(filename)
    # report_output_path = os.path.join(output_path, report_name)
    #
    # report_buffer = BytesIO(report_str.encode(encoding='UTF-8'))
    #
    # # upload the txt report
    # cmd.upload_data(dataset_loader.minio_client, bucket_name, report_output_path, report_buffer)

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

    # # get model card and upload
    # model_card_name = "{}.json".format(filename)
    # model_card_name_output_path = os.path.join(output_path, model_card_name)
    # model_card_buf, model_card = get_model_card_buf(ab_model,
    #                                                 training_total_size,
    #                                                 validation_total_size,
    #                                                 graph_output_path,
    #                                                 input_type,
    #                                                 output_type)
    # cmd.upload_data(dataset_loader.minio_client, bucket_name, model_card_name_output_path, model_card_buf)

    # # add model card
    # model_id = upload_score_residual.add_model_card(model_card)
    #
    # # upload score and residual
    # upload_score_residual.upload_score_residual(model_id,
    #                                             training_predicted_probabilities,
    #                                             training_target_probabilities,
    #                                             validation_predicted_probabilities,
    #                                             validation_target_probabilities,
    #                                             training_predicted_score_images_x,
    #                                             validation_predicted_score_images_x,
    #                                             dataset_loader.training_image_hashes,
    #                                             dataset_loader.validation_image_hashes,
    #                                             training_shuffled_indices_origin,
    #                                             validation_shuffled_indices_origin)



if __name__ == '__main__':
    start_time = time.time()

    train_xgboost(dataset_name="environmental",
                  minio_ip_addr=None,  # will use defualt if none is given
                  minio_access_key="nkjYl5jO4QnpxQU0k0M1",
                  minio_secret_key="MYtmJ9jhdlyYx3T1McYy4Z0HB3FkxjmITXLEPKA1",
                  input_type="embedding",
                  )

    time_elapsed = time.time() - start_time
    print("Time elapsed: {0}s".format(format(time_elapsed, ".2f")))
