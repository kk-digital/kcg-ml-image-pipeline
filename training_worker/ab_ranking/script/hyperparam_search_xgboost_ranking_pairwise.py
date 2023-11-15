import sklearn.datasets
import sklearn.metrics
import os
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import sys
from ray import train, tune
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import session
import numpy as np
from datetime import datetime
from pytz import timezone
import time
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from data_loader.ab_ranking_dataset_loader import ABRankingDatasetLoader
from training_worker.ab_ranking.script.ab_ranking_xgboost_ranking_pairwise import forward_bradley_terry
from training_worker.ab_ranking.model.ab_ranking_linear import ABRankingModel
from training_worker.ab_ranking.model.reports.get_model_card import get_model_card_buf
from utility.minio import cmd
from training_worker.ab_ranking.model import constants
from training_worker.ab_ranking.model.reports import upload_score_residual
from training_worker.ab_ranking.model.reports.graph_report_ab_ranking_linear import get_graph_report

def get_performance_graph_report(results):
    # Initialize all graphs/subplots
    plt.figure(figsize=(22, 20))
    figure_shape = (2, 2)
    performance = plt.subplot2grid(figure_shape, (0, 0), rowspan=2, colspan=2)
    # ----------------------------------------------------------------------------------------------------------------#
    # performance
    loss_values = []
    for res in results:
        loss_values.append(res.metrics["validation-error"])

    x_axis_values = [i for i in range(len(loss_values))]
    performance.plot(x_axis_values,
                     loss_values,
                     label="xgboost",
                     c="#281ad9")

    performance.set_xlabel("Iterations")
    performance.set_ylabel("Loss")
    performance.set_title("Loss vs Iterations")
    performance.legend()

    performance.autoscale(enable=True, axis='both')

    # Save figure
    # graph_path = os.path.join(model_output_path, graph_name)
    # plt.subplots_adjust(left=0.15, hspace=0.5)
    # plt.savefig(graph_path)
    # plt.show()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return buf


def train_ranking(config: dict, training_data, training_targets, validation_data, validation_targets):
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

    evals = [(dtest_reg, "validation")]
    evals_result = {}

    xgb.train(config,
              dtrain_reg,
              evals=evals,
              verbose_eval=False,
              callbacks=[TuneReportCheckpointCallback(filename="model.xgb")],
              evals_result=evals_result
              )


def get_best_model_checkpoint(results):
    best_bst = xgb.Booster()
    best_result = results.get_best_result()

    with best_result.checkpoint.as_directory() as best_checkpoint_dir:
        best_bst.load_model(os.path.join(best_checkpoint_dir, "model.xgb"))
    accuracy = 1.0 - best_result.metrics["validation-error"]
    print(f"Best model parameters: {best_result.config}")
    print(f"Best model total accuracy: {accuracy:.4f}")
    return best_bst


def get_dataset(dataset_name,
                minio_ip_addr,  # will use default if none is given
                minio_access_key,
                minio_secret_key,
                input_type):
    # load dataset
    dataset_loader = ABRankingDatasetLoader(dataset_name=dataset_name,
                                            minio_ip_addr=minio_ip_addr,
                                            minio_access_key=minio_access_key,
                                            minio_secret_key=minio_secret_key,
                                            input_type=input_type)
    dataset_loader.load_dataset(pre_shuffle=False)

    training_total_size = dataset_loader.get_len_training_ab_data()
    validation_total_size = dataset_loader.get_len_validation_ab_data()

    # get training data and convert to numpy array
    training_features_x, \
        training_features_y, \
        training_targets = dataset_loader.get_next_training_feature_vectors_and_target_linear(training_total_size)

    training_features_x = training_features_x.numpy()
    training_features_y = training_features_y.numpy()
    training_targets = training_targets.numpy()

    print("training data shape=", training_features_x.shape)
    print("training target shape=", training_targets.shape)

    # get validation data
    validation_features_x, \
        validation_features_y, \
        validation_targets = dataset_loader.get_validation_feature_vectors_and_target_linear()
    validation_features_x = validation_features_x.numpy()
    validation_features_y = validation_features_y.numpy()
    validation_targets = validation_targets.numpy()

    print("validation data shape=", validation_features_x.shape)
    print("validation target shape=", validation_targets.shape)

    return training_features_x, training_targets, validation_features_x, validation_targets, training_features_y, validation_features_y, dataset_loader


def tune_xgboost(training_data,
                 training_targets,
                 validation_data,
                 validation_targets):
    search_space = {
        # You can mix constants with search space objects.
        "objective": "rank:pairwise",
        "eval_metric": "error",
        "max_depth": tune.randint(1, 9),
        "min_child_weight": tune.choice([1, 2, 3]),
        "subsample": tune.uniform(0.5, 1.0),
        "eta": tune.loguniform(1e-4, 1e-1),
    }
    # This will enable aggressive early stopping of bad trials.
    scheduler = ASHAScheduler(
        max_t=50, grace_period=1, reduction_factor=2  # 10 training iterations
    )

    hyper_opt = HyperOptSearch(metric="validation-error", mode="min")
    trainable_with_cpu = tune.with_resources(train_ranking, {"cpu": 2})
    tuner = tune.Tuner(
        tune.with_parameters(trainable_with_cpu,
                             training_data=training_data,
                             training_targets=training_targets,
                             validation_data=validation_data,
                             validation_targets=validation_targets),
        tune_config=tune.TuneConfig(
            metric="validation-error",
            mode="min",
            search_alg=hyper_opt,
            scheduler=scheduler,
            num_samples=200,
        ),
        param_space=search_space,
    )
    results = tuner.fit()

    return results

def do_hyperparameter_search(dataset_name="environmental",
                             minio_ip_addr=None,  # will use defualt if none is given
                             minio_access_key="",
                             minio_secret_key="",
                             input_type="embedding"):
    # load dataset
    (training_features_x,
     training_targets,
     validation_features_x,
     validation_targets,
     training_features_y,
     validation_features_y,
     dataset_loader) = get_dataset(dataset_name=dataset_name,
                                   input_type=input_type,
                                   minio_ip_addr=minio_ip_addr,
                                   minio_access_key=minio_access_key,
                                   minio_secret_key=minio_secret_key)

    training_total_size = dataset_loader.get_len_training_ab_data()
    validation_total_size = dataset_loader.get_len_validation_ab_data()

    date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%Y-%m-%d')
    print("Current datetime: {}".format(datetime.now(tz=timezone("Asia/Hong_Kong"))))
    bucket_name = "datasets"
    network_type = "xgboost-rank-pairwise"
    output_type = "score"
    output_path = "{}/models/ranking".format(dataset_name)

    # placeholders for the graph report
    epochs = 1000
    input_shape = 768

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

    # do hyper parameter search
    results = tune_xgboost(training_features_x,
                           training_targets,
                           validation_features_x,
                           validation_targets)

    # Load the best model checkpoint.
    best_bst = get_best_model_checkpoint(results)
    xgboost_model = best_bst

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
    training_loss_per_epoch = [0] * epochs
    validation_loss_per_epoch = [0] * epochs
    training_loss = 0.0
    validation_loss = 0.0

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
                                    )

    # upload the graph report
    cmd.upload_data(dataset_loader.minio_client, bucket_name, graph_output_path, graph_buffer)

    # performance
    performance_graph = get_performance_graph_report(results)
    graph_name = "{}_performance.png".format(filename)
    graph_output_path = os.path.join(output_path, graph_name)
    cmd.upload_data(dataset_loader.minio_client, bucket_name, graph_output_path, performance_graph)

if __name__ == "__main__":
    start_time = time.time()

    do_hyperparameter_search(dataset_name="environmental",
                             minio_ip_addr=None,  # will use default if none is given
                             minio_access_key="nkjYl5jO4QnpxQU0k0M1",
                             minio_secret_key="MYtmJ9jhdlyYx3T1McYy4Z0HB3FkxjmITXLEPKA1",
                             input_type="embedding",
                             )

    time_elapsed = time.time() - start_time
    print("Time elapsed: {0}s".format(format(time_elapsed, ".2f")))

