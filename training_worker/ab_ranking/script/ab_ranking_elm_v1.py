import os
import torch
import sys
from datetime import datetime
from pytz import timezone
import argparse

base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from utility.regression_utils import torchinfo_summary
from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from training_worker.ab_ranking.model.reports.ab_ranking_linear_train_report import get_train_report
from training_worker.ab_ranking.model.reports.graph_report_ab_ranking_linear import *
from training_worker.ab_ranking.model.ab_ranking_data_loader import ABRankingDatasetLoader
from training_worker.ab_ranking.model.reports.get_model_card import get_model_card_buf
from utility.minio import cmd
from training_worker.ab_ranking.model import constants
from training_worker.ab_ranking.model.reports import upload_score_residual


def train_ranking(dataset_name: str,
                  minio_ip_addr=None,
                  minio_access_key=None,
                  minio_secret_key=None,
                  input_type="embedding",
                  epochs=8,
                  learning_rate=0.05,
                  buffer_size=20000,
                  train_percent=0.9,
                  training_batch_size=1,
                  weight_decay=0.00,
                  load_data_to_ram=True,
                  debug_asserts=False,
                  normalize_vectors=True,
                  pooling_strategy=constants.AVERAGE_POOLING,
                  num_random_layers=1,
                  add_loss_penalty=True,
                  target_option=constants.TARGET_1_AND_0,
                  duplicate_flip_option=constants.DUPLICATE_AND_FLIP_ALL,
                  randomize_data_per_epoch=True,
                  elm_sparsity=0.5):
    date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%Y-%m-%d')
    print("Current datetime: {}".format(datetime.now(tz=timezone("Asia/Hong_Kong"))))
    bucket_name = "datasets"
    training_dataset_path = os.path.join(bucket_name, dataset_name)
    network_type = "elm-v1"
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

    ab_model = ABRankingELMModel(inputs_shape=input_shape,
                                 num_random_layers=num_random_layers,
                                 elm_sparsity=elm_sparsity)
    training_predicted_score_images_x, \
        training_predicted_score_images_y, \
        training_predicted_probabilities, \
        training_target_probabilities, \
        validation_predicted_score_images_x, \
        validation_predicted_score_images_y, \
        validation_predicted_probabilities, \
        validation_target_probabilities, \
        training_loss_per_epoch, \
        validation_loss_per_epoch = ab_model.train(dataset_loader=dataset_loader,
                                                   training_batch_size=training_batch_size,
                                                   epochs=epochs,
                                                   learning_rate=learning_rate,
                                                   weight_decay=weight_decay,
                                                   add_loss_penalty=add_loss_penalty,
                                                   randomize_data_per_epoch=randomize_data_per_epoch,
                                                   debug_asserts=debug_asserts)

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
    ab_model.save(dataset_loader.minio_client, bucket_name, model_output_path)

    # Generate report
    nn_summary = torchinfo_summary(ab_model.model)

    # get number of correct predictions
    training_target_probabilities = torch.stack(training_target_probabilities)
    training_predicted_probabilities = torch.stack(training_predicted_probabilities)
    training_predicted_score_images_x = torch.stack(training_predicted_score_images_x)
    training_predicted_score_images_y = torch.stack(training_predicted_score_images_y)
    training_loss_per_epoch = torch.stack(training_loss_per_epoch)
    validation_loss_per_epoch = torch.stack(validation_loss_per_epoch)

    validation_predicted_score_images_x = torch.stack(validation_predicted_score_images_x)
    validation_predicted_score_images_y = torch.stack(validation_predicted_score_images_y)
    validation_predicted_probabilities = torch.stack(validation_predicted_probabilities)

    training_target_probabilities = training_target_probabilities.detach().cpu().numpy()
    validation_target_probabilities = validation_target_probabilities.detach().cpu().numpy()
    training_predicted_score_images_x = training_predicted_score_images_x.detach().cpu().numpy()
    training_predicted_score_images_y = training_predicted_score_images_y.detach().cpu().numpy()
    validation_predicted_score_images_x = validation_predicted_score_images_x.detach().cpu().numpy()
    validation_predicted_score_images_y = validation_predicted_score_images_y.detach().cpu().numpy()

    training_predicted_probabilities = training_predicted_probabilities.detach().cpu()
    validation_predicted_probabilities = validation_predicted_probabilities.detach().cpu()

    training_loss_per_epoch = training_loss_per_epoch.detach().cpu()
    validation_loss_per_epoch = validation_loss_per_epoch.detach().cpu()

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
    # save report
    report_str = get_train_report(ab_model,
                                  training_dataset_path,
                                  train_percent,
                                  training_total_size,
                                  validation_total_size,
                                  train_sum_correct,
                                  validation_sum_correct,
                                  nn_summary,
                                  training_predicted_score_images_x,
                                  training_predicted_score_images_y,
                                  validation_predicted_score_images_x,
                                  validation_predicted_score_images_y,
                                  training_batch_size,
                                  learning_rate,
                                  weight_decay,
                                  selected_index_0_count,
                                  selected_index_1_count,
                                  total_images_count,
                                  dataset_loader.datapoints_per_sec)

    # Upload model to minio
    report_name = "{}.txt".format(filename)
    report_output_path = os.path.join(output_path, report_name)

    report_buffer = BytesIO(report_str.encode(encoding='UTF-8'))

    # upload the txt report
    cmd.upload_data(dataset_loader.minio_client, bucket_name, report_output_path, report_buffer)

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
                                    ab_model.loss_func_name,
                                    dataset_name,
                                    pooling_strategy,
                                    normalize_vectors,
                                    num_random_layers,
                                    add_loss_penalty,
                                    target_option,
                                    duplicate_flip_option,
                                    randomize_data_per_epoch,
                                    elm_sparsity,
                                    training_shuffled_indices_origin,
                                    validation_shuffled_indices_origin,
                                    dataset_loader.total_selection_datapoints)
    # upload the graph report
    cmd.upload_data(dataset_loader.minio_client, bucket_name, graph_output_path, graph_buffer)

    # get model card and upload
    model_card_name = "{}.json".format(filename)
    model_card_name_output_path = os.path.join(output_path, model_card_name)
    model_card_buf, model_card = get_model_card_buf(ab_model,
                                        training_total_size,
                                        validation_total_size,
                                        graph_output_path,
                                        input_type,
                                        output_type)
    cmd.upload_data(dataset_loader.minio_client, bucket_name, model_card_name_output_path, model_card_buf)

    # add model card
    model_id = upload_score_residual.add_model_card(model_card)

    # upload score and residual
    upload_score_residual.upload_score_residual(model_id,
                                                training_predicted_probabilities,
                                                training_target_probabilities,
                                                validation_predicted_probabilities,
                                                validation_target_probabilities,
                                                training_predicted_score_images_x,
                                                validation_predicted_score_images_x,
                                                dataset_loader.training_image_hashes,
                                                dataset_loader.validation_image_hashes)

    return model_output_path, report_output_path, graph_output_path


def run_ab_ranking_elm_v1_task(training_task, minio_access_key, minio_secret_key):
    model_output_path, \
        report_output_path, \
        graph_output_path = train_ranking(dataset_name=training_task["dataset_name"],
                                          minio_access_key=minio_access_key,
                                          minio_secret_key=minio_secret_key,
                                          epochs=training_task["epochs"],
                                          learning_rate=training_task["learning_rate"],
                                          buffer_size=training_task["buffer_size"],
                                          train_percent=training_task["train_percent"])

    return model_output_path, report_output_path, graph_output_path


def test_run():
    train_ranking(minio_ip_addr=None,  # will use default if none is given
                  minio_access_key="nkjYl5jO4QnpxQU0k0M1",
                  minio_secret_key="MYtmJ9jhdlyYx3T1McYy4Z0HB3FkxjmITXLEPKA1",
                  dataset_name="environmental",
                  input_type="embedding",
                  epochs=10,
                  learning_rate=0.1,
                  buffer_size=20000,
                  train_percent=0.9,
                  training_batch_size=1,
                  weight_decay=0.01,
                  load_data_to_ram=True,
                  debug_asserts=True,
                  normalize_vectors=True,
                  pooling_strategy=constants.AVERAGE_POOLING,
                  num_random_layers=2,
                  add_loss_penalty=True,
                  target_option=constants.TARGET_1_AND_0,
                  duplicate_flip_option=constants.DUPLICATE_AND_FLIP_RANDOM,
                  randomize_data_per_epoch=True,
                  elm_sparsity=0.0)

# if __name__ == '__main__':
#     test_run()
