import os
import torch
import sys
from datetime import datetime
from pytz import timezone
import argparse

base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from utility.http import request
from utility.regression_utils import torchinfo_summary
from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from training_worker.ab_ranking.model.reports.ab_ranking_train_report import get_train_report
from training_worker.ab_ranking.model.reports.graph_report_ab_ranking_v2 import *
from data_loader.rank_ab_ranking_loader import ABRankingDatasetLoader
from training_worker.ab_ranking.model.reports.get_model_card import get_model_card_buf, get_ranking_model_data
from utility.minio import cmd
from training_worker.ab_ranking.model import constants
from training_worker.ab_ranking.model.reports import score_residual, sigma_score

# import http request service for getting rank model list
from utility.http.request import http_get_rank_list

# import constants for training ranking model
from training_worker.ab_ranking.model import constants


def train_ranking(rank_model_info: dict, # rank_model_info must have rank_model_id and rank_model_string
                  minio_ip_addr=None,
                  minio_access_key=None,
                  minio_secret_key=None,
                  input_type="embedding",
                  epochs=8,
                  learning_rate=0.05,
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
                  elm_sparsity=0.5,
                  penalty_range = 5.0):
    date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%Y-%m-%d')
    print("Current datetime: {}".format(datetime.now(tz=timezone("Asia/Hong_Kong"))))
    bucket_name = "datasets"
    training_dataset_path = os.path.join(bucket_name, "ranks/{:05}/data/ranking/aggregate".format(rank_model_info["rank_model_id"]))
    network_type = "elm-v1"
    output_type = "score"
    rank_id= rank_model_info["rank_model_id"]
    output_path = "ranks/{:05}/models/ranking".format(rank_id)

    # check input type
    if input_type not in constants.ALLOWED_INPUT_TYPES:
        raise Exception("input type is not supported: {}".format(input_type))

    input_shape = 2 * 768
    if input_type in [constants.EMBEDDING_POSITIVE, constants.EMBEDDING_NEGATIVE, constants.CLIP]:
        input_shape = 768
    if input_type in [constants.KANDINSKY_CLIP]:
        input_shape = 1280

    # load dataset
    dataset_loader = ABRankingDatasetLoader(rank_model_id=rank_model_info["rank_model_id"],
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
    loaded = dataset_loader.load_dataset()

    # if dataset is not loaded, ranking will be cancelled
    if not loaded:
        return
    
    # get final filename
    sequence = 0
    # if exist, increment sequence
    while True:
        filename = "{}-{:02}-{}-{}-{}".format(date_now, sequence, output_type, network_type, input_type)
        exists = cmd.is_object_exists(dataset_loader.minio_client, bucket_name,
                                      os.path.join(output_path, filename + ".safetensors"))
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
                                                   debug_asserts=debug_asserts,
                                                   penalty_range=penalty_range)

    # data for chronological score graph
    training_shuffled_indices_origin = []
    for index in dataset_loader.training_data_paths_indices_shuffled:
        training_shuffled_indices_origin.append(index)

    validation_shuffled_indices_origin = []
    for index in dataset_loader.validation_data_paths_indices_shuffled:
        validation_shuffled_indices_origin.append(index)

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
    ab_model.mean = mean
    ab_model.standard_deviation = standard_deviation

    # add hyperparameter data to model
    ab_model.add_hyperparameters_config(epochs=epochs,
                                        learning_rate=learning_rate,
                                        train_percent=train_percent,
                                        training_batch_size=training_batch_size,
                                        weight_decay=weight_decay,
                                        pooling_strategy=pooling_strategy,
                                        add_loss_penalty=add_loss_penalty,
                                        target_option=target_option,
                                        duplicate_flip_option=duplicate_flip_option,
                                        randomize_data_per_epoch=randomize_data_per_epoch,
                                        num_random_layers=num_random_layers,
                                        elm_sparsity=elm_sparsity)

    # Upload model to minio
    model_name = "{}.safetensors".format(filename)
    model_output_path = os.path.join(output_path, model_name)
    ab_model.save(dataset_loader.minio_client, bucket_name, model_output_path)

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

    graph_buffer = get_graph_report(training_loss=ab_model.training_loss,
                                    validation_loss=ab_model.validation_loss,
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
                                    learning_rate=learning_rate,
                                    training_batch_size=training_batch_size,
                                    weight_decay=weight_decay,
                                    date=date_now,
                                    network_type=network_type,
                                    input_type=input_type,
                                    input_shape=input_shape,
                                    output_type=output_type,
                                    train_sum_correct=train_sum_correct,
                                    validation_sum_correct=validation_sum_correct,
                                    loss_func=ab_model.loss_func_name,
                                    rank_name=rank_model_info["rank_model_string"],
                                    pooling_strategy=pooling_strategy,
                                    normalize_vectors=normalize_vectors,
                                    num_random_layers=num_random_layers,
                                    add_loss_penalty=add_loss_penalty,
                                    target_option=target_option,
                                    duplicate_flip_option=duplicate_flip_option,
                                    randomize_data_per_epoch=randomize_data_per_epoch,
                                    elm_sparsity=elm_sparsity,
                                    training_shuffled_indices_origin=training_shuffled_indices_origin,
                                    validation_shuffled_indices_origin=validation_shuffled_indices_origin,
                                    total_selection_datapoints=dataset_loader.total_selection_datapoints,
                                    loss_penalty_range=penalty_range,
                                    saved_model_epoch=ab_model.lowest_loss_model_epoch
                                    )
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

    # save model to mongodb if its input type is clip-h
    ranking_model_name= "{}-{}-{}".format(output_type, network_type, input_type)
    model_data= get_ranking_model_data(model_name= ranking_model_name,
                                       model_type="elm-v1",
                                       rank_id= rank_id,
                                       model_path= model_output_path,
                                       latest_model_creation_time= date_now,
                                       creation_time=date_now)
    
    if(input_type==constants.KANDINSKY_CLIP):
        request.http_add_ranking_model(model_data)

    return model_output_path, report_output_path, graph_output_path


def run_ab_ranking_elm_v1_task(training_task, minio_access_key, minio_secret_key):
    model_output_path, \
        report_output_path, \
        graph_output_path = train_ranking(rank=training_task["rank_model_infor"],
                                          minio_access_key=minio_access_key,
                                          minio_secret_key=minio_secret_key,
                                          epochs=training_task["epochs"],
                                          learning_rate=training_task["learning_rate"],
                                          train_percent=training_task["train_percent"])

    return model_output_path, report_output_path, graph_output_path


def parse_args():
    parser = argparse.ArgumentParser()

    # Add arguments for MinIO connection
    parser.add_argument('--minio-ip-addr', type=str, default=None, help='MinIO IP address')
    parser.add_argument('--minio-access-key', type=str, default=None, help='MinIO access key')
    parser.add_argument('--minio-secret-key', type=str, default=None, help='MinIO secret key')

     # Add arguments for training parameters
    parser.add_argument('--rank-id', type=int, default=None, help='Rank Id, if none, will train all ranks')
    parser.add_argument('--input-type', type=str, default='clip', help='Input type')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--train-percent', type=float, default=0.9, help='Percentage of data used for training')
    parser.add_argument('--training-batch-size', type=int, default=1, help='Training batch size')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--load-data-to-ram', action='store_true', help='Load data to RAM')
    parser.add_argument('--debug-asserts', action='store_true', help='Enable debug asserts')
    parser.add_argument('--normalize-vectors', action='store_true', help='Normalize input vectors')
    parser.add_argument('--pooling-strategy', type=str, default=constants.AVERAGE_POOLING, help='Pooling strategy')
    parser.add_argument('--add-loss-penalty', action='store_true', help='Add loss penalty')
    parser.add_argument('--target-option', type=str, default=constants.TARGET_1_AND_0, help='Target option')
    parser.add_argument('--duplicate-flip-option', type=str, default=constants.DUPLICATE_AND_FLIP_ALL, help='Duplicate and flip option')
    parser.add_argument('--randomize-data-per-epoch', action='store_true', help='Randomize data per epoch')
    parser.add_argument('--penalty-range', type=float, default=5.0, help='Penalty range')

    # more paramter for elm ranking model
    parser.add_argument('--num-random-layers', type=int, default=1, help='Number of random layers')
    parser.add_argument('--elm-sparsity', type=float, default=0.5, help='ELM sparsity')

    

    return parser.parse_args()

def main():
    args = parse_args()

    # Get all rank model infor
    rank_model_list = http_get_rank_list()

    for rank_model in rank_model_list:
        print("{} Ranking....".format(rank_model["rank_model_string"]))
        
        # if rank_id is None, will train all ranks
        if args.rank_id is not None and rank_model["rank_model_id"] != args.rank_id:
            continue
        
        train_ranking(
                rank_model_info=rank_model,
                minio_ip_addr=args.minio_ip_addr,
                minio_access_key=args.minio_access_key,
                minio_secret_key=args.minio_secret_key,
                input_type=args.input_type,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                train_percent=args.train_percent,
                training_batch_size=args.training_batch_size,
                weight_decay=args.weight_decay,
                load_data_to_ram=args.load_data_to_ram,
                debug_asserts=args.debug_asserts,
                normalize_vectors=args.normalize_vectors,
                pooling_strategy=args.pooling_strategy,
                num_random_layers=args.num_random_layers,
                add_loss_penalty=args.add_loss_penalty,
                target_option=args.target_option,
                duplicate_flip_option=args.duplicate_flip_option,
                randomize_data_per_epoch=args.randomize_data_per_epoch,
                elm_sparsity=args.elm_sparsity,
                penalty_range=args.penalty_range,
        )
    
if __name__ == '__main__':

    main()
