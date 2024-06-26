import os
import sys
from datetime import datetime
from pytz import timezone
import time
import xgboost as xgb
import hashlib

base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from training_worker.ab_ranking.model.reports.graph_report_ab_ranking import *
from data_loader.ab_ranking_dataset_loader import ABRankingDatasetLoader
from utility.minio import cmd
from training_worker.ab_ranking.model import constants
from training_worker.ab_ranking.model.reports.get_model_card import get_xgboost_model_card_buf
from training_worker.ab_ranking.model.reports import score_residual, sigma_score


def np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def forward_bradley_terry(predicted_score_images_x, predicted_score_images_y, use_sigmoid=True):
    if use_sigmoid:
        # scale the score
        # scaled_score_image_x = torch.multiply(1000.0, predicted_score_images_x)
        # scaled_score_image_y = torch.multiply(1000.0, predicted_score_images_y)

        diff_predicted_score = np.subtract(predicted_score_images_x, predicted_score_images_y)
        res_predicted_score = np.divide(diff_predicted_score, 1.0)
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
                  train_percent=0.9,
                  load_data_to_ram=False,
                  normalize_vectors=True,
                  pooling_strategy=constants.AVERAGE_POOLING,
                  target_option=constants.TARGET_1_AND_0,
                  duplicate_flip_option=constants.DUPLICATE_AND_FLIP_ALL,
                  ):
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
    if input_type in [constants.KANDINSKY_CLIP]:
        input_shape = 1280

    # load dataset
    dataset_loader = ABRankingDatasetLoader(dataset_name=dataset_name,
                                            minio_ip_addr=minio_ip_addr,
                                            minio_access_key=minio_access_key,
                                            minio_secret_key=minio_secret_key,
                                            input_type=input_type,
                                            train_percent=train_percent,
                                            load_to_ram=load_data_to_ram,
                                            pooling_strategy=pooling_strategy,
                                            normalize_vectors=normalize_vectors,
                                            target_option=target_option,
                                            duplicate_flip_option=duplicate_flip_option)
    dataset_loader.load_dataset(pre_shuffle=False)

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

    train_len = int(len(training_targets) / 2)
    validate_len = int(len(validation_targets) / 2)

    # group
    train_group = np.array([2 for _ in range(train_len)])
    validation_group = np.array([2 for _ in range(validate_len)])

    dtrain_reg.set_group(train_group)
    dtest_reg.set_group(validation_group)

    # params based on hyperparam search result
    params = {'objective': 'rank:pairwise', 'eval_metric': 'error', 'max_depth': 8, 'min_child_weight': 3,
              'subsample': 0.5327713979402486, 'eta': 0.0998724001538154}

    evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]
    evals_result = {}
    n = 500
    xgboost_model = xgb.train(params,
                              dtrain_reg,
                              num_boost_round=n,
                              evals=evals,
                              verbose_eval=10,  # Every ten rounds
                              early_stopping_rounds=50,  # Activate early stopping
                              evals_result=evals_result
                              )

    train_loss_per_round = evals_result["train"]["error"]
    validation_loss_per_round = evals_result["validation"]["error"]
    epochs = len(train_loss_per_round)
    training_loss = train_loss_per_round[-1]
    validation_loss = validation_loss_per_round[-1]

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

    training_predicted_score_images_x = train_x_pred_scores
    training_predicted_score_images_y = train_y_pred_scores
    training_predicted_probabilities = train_pred_prob
    training_target_probabilities = training_targets
    validation_predicted_score_images_x = validation_x_pred_scores
    validation_predicted_score_images_y = validation_y_pred_scores
    validation_predicted_probabilities = validation_pred_prob
    validation_target_probabilities = validation_targets
    training_loss_per_epoch = train_loss_per_round
    validation_loss_per_epoch = validation_loss_per_round

    # data for chronological score graph
    training_shuffled_indices_origin = []
    for index in dataset_loader.training_data_paths_indices_shuffled:
        training_shuffled_indices_origin.append(index)

    validation_shuffled_indices_origin = []
    for index in dataset_loader.validation_data_paths_indices_shuffled:
        validation_shuffled_indices_origin.append(index)

    # get sigma scores
    (x_chronological_sigma_scores,
     x_chronological_image_hashes,
     y_chronological_sigma_scores,
     mean,
     standard_deviation) = sigma_score.get_chronological_sigma_scores(training_target_probabilities,
                                                                        validation_target_probabilities,
                                                                        training_predicted_score_images_x,
                                                                        validation_predicted_score_images_x,
                                                                        training_predicted_score_images_y,
                                                                        validation_predicted_score_images_y,
                                                                        dataset_loader.training_image_hashes,
                                                                        dataset_loader.validation_image_hashes,
                                                                        training_shuffled_indices_origin,
                                                                        validation_shuffled_indices_origin)

    # Upload model to minio
    model_name = "{}.json".format(filename)
    model_output_path = os.path.join(output_path, model_name)
    xgboost_model_buf = xgboost_model.save_raw(raw_format='json')
    buffer = BytesIO(xgboost_model_buf)
    buffer.seek(0)

    cmd.upload_data(dataset_loader.minio_client, "datasets", model_output_path, buffer)

    model_hash = hashlib.sha256(xgboost_model_buf).hexdigest()

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

    # show and save graph
    graph_name = "{}.png".format(filename)
    graph_output_path = os.path.join(output_path, graph_name)

    graph_buffer = get_graph_report(training_loss=training_loss,
                                    validation_loss=validation_loss,
                                    train_prob_predictions=training_predicted_probabilities,
                                    training_targets=training_target_probabilities,
                                    validation_prob_predictions=validation_predicted_probabilities,
                                    validation_targets=validation_target_probabilities,
                                    training_pred_scores_img_x=training_predicted_score_images_x,
                                    training_pred_scores_img_y=training_predicted_score_images_y,
                                    validation_pred_scores_img_x=validation_predicted_score_images_x,
                                    validation_pred_scores_img_y=validation_predicted_score_images_y,
                                    training_total_size=training_total_size,
                                    validation_total_size=validation_total_size,
                                    training_losses=training_loss_per_epoch,
                                    validation_losses=validation_loss_per_epoch,
                                    mean=mean,
                                    standard_deviation=standard_deviation,
                                    x_chronological_sigma_scores=x_chronological_sigma_scores,
                                    y_chronological_sigma_scores=y_chronological_sigma_scores,
                                    epochs=epochs,
                                    date=date_now,
                                    network_type=network_type,
                                    input_type=input_type,
                                    input_shape=input_shape,
                                    output_type=output_type,
                                    train_sum_correct=train_sum_correct,
                                    validation_sum_correct=validation_sum_correct,
                                    loss_func="",
                                    dataset_name=dataset_name,
                                    training_shuffled_indices_origin=training_shuffled_indices_origin,
                                    validation_shuffled_indices_origin=validation_shuffled_indices_origin,
                                    total_selection_datapoints=dataset_loader.total_selection_datapoints,
                                    pooling_strategy=pooling_strategy,
                                    target_option=target_option,
                                    duplicate_flip_option=duplicate_flip_option
                                    )

    # upload the graph report
    cmd.upload_data(dataset_loader.minio_client, bucket_name, graph_output_path, graph_buffer)

    # get model card and upload
    model_card_name = "{}.json".format(filename)
    model_card_name_output_path = os.path.join(output_path, model_card_name)
    model_card_buf, model_card = get_xgboost_model_card_buf(date_now,
                                                            network_type,
                                                            model_output_path,
                                                            model_hash,
                                                            input_type,
                                                            output_type,
                                                            training_total_size,
                                                            validation_total_size,
                                                            training_loss,
                                                            validation_loss,
                                                            graph_output_path)
    cmd.upload_data(dataset_loader.minio_client, bucket_name, model_card_name_output_path, model_card_buf)

    # add model card
    model_id = score_residual.add_model_card(model_card)


if __name__ == '__main__':
    start_time = time.time()

    train_xgboost(dataset_name="environmental",
                  minio_ip_addr=None,  # will use default if none is given
                  minio_access_key="nkjYl5jO4QnpxQU0k0M1",
                  minio_secret_key="MYtmJ9jhdlyYx3T1McYy4Z0HB3FkxjmITXLEPKA1",
                  input_type="embedding",
                  train_percent=0.9,
                  load_data_to_ram=True,
                  normalize_vectors=True,
                  pooling_strategy=constants.AVERAGE_POOLING,
                  target_option=constants.TARGET_1_AND_0,
                  duplicate_flip_option=constants.DUPLICATE_AND_FLIP_ALL,
                  )

    time_elapsed = time.time() - start_time
    print("Time elapsed: {0}s".format(format(time_elapsed, ".2f")))
