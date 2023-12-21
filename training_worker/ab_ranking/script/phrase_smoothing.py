import os
import torch
import sys
from datetime import datetime
from pytz import timezone

base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from utility.regression_utils import torchinfo_summary
from training_worker.ab_ranking.model.reports.ab_ranking_train_report import get_train_report
from training_worker.ab_ranking.model.reports.graph_report_phrase_smoothing import *
from training_worker.ab_ranking.model.reports.get_model_card import get_model_card_buf
from utility.minio import cmd
from training_worker.ab_ranking.model import constants
from training_worker.ab_ranking.model.reports import score_residual, sigma_score
from training_worker.ab_ranking.model.phrase_smoothing import ScorePhraseSmoothingModel
from data_loader.independent_approximation_dataset_loader import IndependentApproximationDatasetLoader
from data_loader.phrase_vector_loader import PhraseVectorLoader
from data_loader.phrase_embedding_loader import PhraseEmbeddingLoader

def train_ranking(dataset_name: str,
                  input_type="positive",
                  minio_ip_addr=None,
                  minio_access_key=None,
                  minio_secret_key=None,
                  phrases_csv_name=None,
                  epochs=10000,
                  learning_rate=0.05,
                  train_percent=0.9,
                  training_batch_size=1,
                  weight_decay=0.00,
                  debug_asserts=False,
                  normalize_vectors=True,
                  pooling_strategy=constants.AVERAGE_POOLING,
                  add_loss_penalty=True,
                  target_option=constants.TARGET_1_AND_0,
                  duplicate_flip_option=constants.DUPLICATE_AND_FLIP_ALL,
                  randomize_data_per_epoch=True
                  ):

    # load phrase
    phrase_loader = PhraseVectorLoader(dataset_name=dataset_name,
                                       minio_ip_addr=minio_ip_addr,
                                       minio_access_key=minio_access_key,
                                       minio_secret_key=minio_secret_key,)

    if phrases_csv_name != None:
        phrase_loader.load_dataset_phrases_from_csv(phrases_csv_name)
    else:
        phrase_loader.load_dataset_phrases()
        phrase_loader.upload_csv()

    date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%Y-%m-%d')
    print("Current datetime: {}".format(datetime.now(tz=timezone("Asia/Hong_Kong"))))
    bucket_name = "datasets"
    training_dataset_path = os.path.join(bucket_name, dataset_name)
    network_type = "phrase-smoothing"
    input_type_str = "{}-phrase-vector".format(input_type)
    output_type = "score"
    output_path = "{}/models/ranking".format(dataset_name)

    if input_type == "positive":
        input_shape = len(phrase_loader.index_positive_phrases_dict)
        if phrase_loader.index_positive_phrases_dict.get(-1) is not None:
            input_shape -= 1
            print("exists")
            print("val=", phrase_loader.index_positive_phrases_dict.get(-1))
    else:
        input_shape = len(phrase_loader.index_negative_phrases_dict)
        if phrase_loader.index_negative_phrases_dict.get(-1) is not None:
            input_shape -= 1

    print("input shape=", input_shape)
    # load dataset
    dataset_loader = IndependentApproximationDatasetLoader(dataset_name=dataset_name,
                                                           minio_ip_addr=minio_ip_addr,
                                                           minio_access_key=minio_access_key,
                                                           minio_secret_key=minio_secret_key,
                                                           train_percent=train_percent,
                                                           phrase_vector_loader=phrase_loader,
                                                           input_type=input_type)
    dataset_loader.load_dataset()

    # get final filename
    sequence = 0
    # if exist, increment sequence
    while True:
        filename = "{}-{:02}-{}-{}-{}".format(date_now, sequence, output_type, network_type, input_type_str)
        exists = cmd.is_object_exists(dataset_loader.minio_client, bucket_name,
                                      os.path.join(output_path, filename + ".pth"))
        if not exists:
            break

        sequence += 1

    training_total_size = dataset_loader.get_len_training_ab_data()
    validation_total_size = dataset_loader.get_len_validation_ab_data()

    # load phrase embeddings
    phrase_embedding_loader = PhraseEmbeddingLoader(dataset_name=dataset_name,
                                                    minio_ip_addr=minio_ip_addr,
                                                    minio_access_key=minio_access_key,
                                                    minio_secret_key=minio_secret_key)
    phrase_embedding_loader.load_dataset_phrases()

    # update phrase embeddings
    phrase_embedding_loader.update_dataset_phrases(phrase_loader.get_positive_phrases_arr())

    # get token length vector
    token_length_vector = phrase_loader.get_token_length_vector(input_type)
    token_length_vector = torch.tensor(token_length_vector)

    ab_model = ScorePhraseSmoothingModel(inputs_shape=input_shape,
                                         dataset_loader=dataset_loader,
                                         phrase_vector_loader=phrase_loader,
                                         phrase_embedding_loader=phrase_embedding_loader,
                                         token_length_vector=token_length_vector,
                                         input_type=input_type)

    training_predicted_score_images_x, \
        training_predicted_score_images_y, \
        training_predicted_probabilities, \
        training_target_probabilities, \
        validation_predicted_score_images_x, \
        validation_predicted_score_images_y, \
        validation_predicted_probabilities, \
        validation_target_probabilities, \
        training_loss_per_epoch, \
        validation_loss_per_epoch = ab_model.train(training_batch_size=training_batch_size,
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
                                        randomize_data_per_epoch=randomize_data_per_epoch)

    # Upload model to minio
    model_name = "{}.pth".format(filename)
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

    # Upload text report to minio
    report_name = "{}.txt".format(filename)
    report_output_path = os.path.join(output_path, report_name)

    report_buffer = BytesIO(report_str.encode(encoding='UTF-8'))

    # upload the txt report
    cmd.upload_data(dataset_loader.minio_client, bucket_name, report_output_path, report_buffer)

    # get score offset
    score_offset = -1
    for name, param in ab_model.model.named_parameters():
        if name == "score_offset":
            score_offset = param.cpu().detach().squeeze().numpy()

    # get phrase scores vector
    phrase_scores_vector = None
    # for name, param in ab_model.model.named_parameters():
    #     if name == "prompt_phrase_trainable_score":
    #         phrase_scores_vector = param.cpu().detach().squeeze().numpy()

    phrase_scores_vector = ab_model.upload_phrases_score_csv()

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
                                    phrase_scores=phrase_scores_vector,
                                    score_offset=score_offset,
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
                                    input_type=input_type_str,
                                    input_shape=input_shape,
                                    output_type=output_type,
                                    train_sum_correct=train_sum_correct,
                                    validation_sum_correct=validation_sum_correct,
                                    loss_func=ab_model.loss_func_name,
                                    dataset_name=dataset_name,
                                    pooling_strategy=pooling_strategy,
                                    normalize_vectors=normalize_vectors,
                                    num_random_layers=-1,
                                    add_loss_penalty=add_loss_penalty,
                                    target_option=target_option,
                                    duplicate_flip_option=duplicate_flip_option,
                                    randomize_data_per_epoch=randomize_data_per_epoch,
                                    elm_sparsity=-1,
                                    training_shuffled_indices_origin=training_shuffled_indices_origin,
                                    validation_shuffled_indices_origin=validation_shuffled_indices_origin,
                                    total_selection_datapoints=dataset_loader.total_selection_datapoints)

    # upload the graph report
    cmd.upload_data(dataset_loader.minio_client, bucket_name, graph_output_path, graph_buffer)

    # get model card and upload
    model_card_name = "{}.json".format(filename)
    model_card_name_output_path = os.path.join(output_path, model_card_name)
    model_card_buf, model_card = get_model_card_buf(ab_model,
                                                    training_total_size,
                                                    validation_total_size,
                                                    graph_output_path,
                                                    input_type_str,
                                                    output_type)
    cmd.upload_data(dataset_loader.minio_client, bucket_name, model_card_name_output_path, model_card_buf)





