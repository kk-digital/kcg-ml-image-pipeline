import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics
from dotenv import dotenv_values
import schedule
import sys
from pytz import timezone
from datetime import datetime

base_directory = os.getcwd()
sys.path.insert(0, base_directory)
sys.path.insert(0, os.path.join(base_directory, 'utils', 'dataset'))

from utility.regression_utils import (create_directories,
                                      ensure_zip_file,
                                      delete_all_files_in_folder,
                                      torchinfo_summary)

from ab_ranking_linear.model.ab_ranking_linear import ABRankingModel
from ab_ranking_linear.model.reports.ab_ranking_linear_train_report import get_train_report
from ab_ranking_linear.model.reports.save_graph_report_ab_ranking_linear import *
from ab_ranking_linear.model.ab_ranking_data_loader import ABRankingDatasetLoader, split_ab_data_vectors


def train_ranking():
    epochs = 10
    learning_rate = 0.001
    input_type = "clip-feature"
    train_percent = 0.7
    dataset_name = "icons"
    config = dotenv_values("./scheduled_workers/.env")
    model_output_path = "./output"
    # load dataset
    dataset_loader = ABRankingDatasetLoader(dataset_name=dataset_name,
                                            minio_access_key=config["MINIO_ACCESS_KEY"],
                                            minio_secret_key=config["MINIO_SECRET_KEY"],
                                            buffer_size=20000,
                                            train_percent=0.9)
    dataset_loader.load_dataset()
    ab_model = ABRankingModel(inputs_shape=768)
    training_loss_per_epoch, validation_loss_per_epoch = ab_model.train(dataset_loader=dataset_loader,
                                                                        training_batch_size=16,
                                                                        epochs=epochs,
                                                                        learning_rate=learning_rate)

    # # Storing the classification model to disk
    # ab_model.save(model_output_path)
    #
    # # Generate report
    # # nn_summary = torchinfo_summary(ab_model.model, training_image_x_features)
    #
    # training_predicted_score_images_x = training_predicted_score_images_x.detach().cpu().numpy()
    # training_predicted_score_images_y = training_predicted_score_images_y.detach().cpu().numpy()
    # validation_predicted_score_images_x = validation_predicted_score_images_x.detach().cpu().numpy()
    # validation_predicted_score_images_y = validation_predicted_score_images_y.detach().cpu().numpy()

    # training_pred_prob = training_pred_prob.detach().cpu().numpy()
    # training_target_probabilities = training_target_probabilities.detach().cpu().numpy()
    # validation_pred_prob = validation_pred_prob.detach().cpu().numpy()
    # validation_target_probabilities = validation_target_probabilities.detach().cpu().numpy()

    # get number of correct predictions
    # train_sum_correct = 0
    # for i in range(len(training_target_probabilities)):
    #     if training_target_probabilities[i] == [1.0]:
    #         if training_predicted_score_images_x[i] > training_predicted_score_images_y[i]:
    #             train_sum_correct += 1
    #     else:
    #         if training_predicted_score_images_x[i] < training_predicted_score_images_y[i]:
    #             train_sum_correct += 1

    # validation_sum_correct = 0
    # for i in range(len(validation_target_probabilities)):
    #     if validation_target_probabilities[i] == [1.0]:
    #         if validation_predicted_score_images_x[i] > validation_predicted_score_images_y[i]:
    #             validation_sum_correct += 1
    #     else:
    #         if validation_predicted_score_images_x[i] < validation_predicted_score_images_y[i]:
    #             validation_sum_correct += 1

    # # save report
    # report_str = get_train_report(ab_model,
    #                               training_dataset_path,
    #                               train_percent,
    #                               base_training_total_size,
    #                               training_total_size,
    #                               base_validation_total_size,
    #                               validation_total_size,
    #                               train_sum_correct,
    #                               validation_sum_correct,
    #                               nn_summary,
    #                               training_predicted_score_images_x,
    #                               training_predicted_score_images_y,
    #                               validation_predicted_score_images_x,
    #                               validation_predicted_score_images_y)
    #
    # reports_name = "ab-ranking-linear-report.txt"
    # reports_path = os.path.join(model_output_path, reports_name)
    #
    # # save report
    # with open(reports_path, "w", encoding="utf-8") as file:
    #     file.write(report_str)
    # print("Reports saved at {}".format(reports_path))
    #
    # # show and save graph
    # graph_name = "ab-ranking-linear-report.png"
    # save_graph_report(training_pred_prob,
    #                   training_target_probabilities,
    #                   validation_pred_prob,
    #                   validation_target_probabilities,
    #                   training_predicted_score_images_x,
    #                   training_predicted_score_images_y,
    #                   validation_predicted_score_images_x,
    #                   validation_predicted_score_images_y,
    #                   training_total_size,
    #                   validation_total_size,
    #                   input_type,
    #                   training_loss_per_epoch,
    #                   validation_loss_per_epoch,
    #                   epochs, learning_rate,
    #                   graph_name,
    #                   model_output_path)


def main(time_to_run: str):
    print("Current datetime: {}".format(datetime.now(tz=timezone("Asia/Hong_Kong"))))
    print("The script will run everyday at {} UTC+8:00".format(time_to_run))
    # schedule.every().day.at(time_to_run, timezone("Asia/Hong_Kong")).do(train_ranking)
    # while True:
    #     schedule.run_pending()
    train_ranking()


def parse_args():
    parser = argparse.ArgumentParser(description="Worker for ab ranking linear training")

    # Required parameters
    parser.add_argument("--time-to-run", type=str, default="00:30")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args.time_to_run)
