import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from io import BytesIO
import torch
import sys

base_directory = os.getcwd()
sys.path.insert(0, base_directory)


def separate_values_based_on_targets(training_targets, validation_targets, train_prob_predictions,
                                     validation_prob_predictions, training_pred_scores_img_x,
                                     training_pred_scores_img_y, validation_pred_scores_img_x,
                                     validation_pred_scores_img_y):
    train_prob_predictions_target_1 = []
    train_prob_predictions_target_0 = []
    training_pred_scores_img_x_target_1 = []
    training_pred_scores_img_y_target_1 = []
    training_pred_scores_img_x_target_0 = []
    training_pred_scores_img_y_target_0 = []

    validation_prob_predictions_target_1 = []
    validation_prob_predictions_target_0 = []
    validation_pred_scores_img_x_target_1 = []
    validation_pred_scores_img_y_target_1 = []
    validation_pred_scores_img_x_target_0 = []
    validation_pred_scores_img_y_target_0 = []

    # separate training values with targets 1.0 and 0.0
    for i in range(len(train_prob_predictions)):
        if training_targets[i] == [1.0]:
            train_prob_predictions_target_1.append(train_prob_predictions[i])
            training_pred_scores_img_x_target_1.append(training_pred_scores_img_x[i])
            training_pred_scores_img_y_target_1.append(training_pred_scores_img_y[i])
        else:
            train_prob_predictions_target_0.append(train_prob_predictions[i])
            training_pred_scores_img_x_target_0.append(training_pred_scores_img_x[i])
            training_pred_scores_img_y_target_0.append(training_pred_scores_img_y[i])

    # separate validation values with targets 1.0 and 0.0
    for i in range(len(validation_prob_predictions)):
        if validation_targets[i] == [1.0]:
            validation_prob_predictions_target_1.append(validation_prob_predictions[i])
            validation_pred_scores_img_x_target_1.append(validation_pred_scores_img_x[i])
            validation_pred_scores_img_y_target_1.append(validation_pred_scores_img_y[i])
        else:
            validation_prob_predictions_target_0.append(validation_prob_predictions[i])
            validation_pred_scores_img_x_target_0.append(validation_pred_scores_img_x[i])
            validation_pred_scores_img_y_target_0.append(validation_pred_scores_img_y[i])

    if len(train_prob_predictions_target_1) > 0 and torch.is_tensor(train_prob_predictions_target_1[0]):
        train_prob_predictions_target_1 = torch.stack(train_prob_predictions_target_1)
    if len(train_prob_predictions_target_0) > 0 and torch.is_tensor(train_prob_predictions_target_0[0]):
        train_prob_predictions_target_0 = torch.stack(train_prob_predictions_target_0)
    if len(validation_prob_predictions_target_1) > 0 and torch.is_tensor(validation_prob_predictions_target_1[0]):
        validation_prob_predictions_target_1 = torch.stack(validation_prob_predictions_target_1)
    if len(validation_prob_predictions_target_0) > 0 and torch.is_tensor(validation_prob_predictions_target_0[0]):
        validation_prob_predictions_target_0 = torch.stack(validation_prob_predictions_target_0)

    return train_prob_predictions_target_1, \
        train_prob_predictions_target_0, \
        validation_prob_predictions_target_1, \
        validation_prob_predictions_target_0, \
        training_pred_scores_img_x_target_1, \
        training_pred_scores_img_y_target_1, \
        training_pred_scores_img_x_target_0, \
        training_pred_scores_img_y_target_0, \
        validation_pred_scores_img_x_target_1, \
        validation_pred_scores_img_y_target_1, \
        validation_pred_scores_img_x_target_0, \
        validation_pred_scores_img_y_target_0


def get_graph_report(training_loss,
                     validation_loss,
                     train_prob_predictions,
                     training_targets,
                     validation_prob_predictions,
                     validation_targets,
                     training_pred_scores_img_x,
                     training_pred_scores_img_y,
                     validation_pred_scores_img_x,
                     validation_pred_scores_img_y,
                     training_total_size,
                     validation_total_size,
                     training_losses,
                     validation_losses,
                     x_chronological_sigma_scores=None,
                     y_chronological_sigma_scores=None,
                     epochs=-1,
                     learning_rate=-1,
                     training_batch_size=-1,
                     weight_decay=-1,
                     date=-1,
                     network_type=-1,
                     input_type=-1,
                     input_shape=-1,
                     output_type=-1,
                     train_sum_correct=-1,
                     validation_sum_correct=-1,
                     loss_func=-1,
                     dataset_name=-1,
                     pooling_strategy=-1,
                     normalize_vectors=-1,
                     num_random_layers=-1,
                     add_loss_penalty=-1,
                     target_option=-1,
                     duplicate_flip_option=-1,
                     randomize_data_per_epoch=-1,
                     elm_sparsity=-1,
                     training_shuffled_indices_origin=None,
                     validation_shuffled_indices_origin=None,
                     total_selection_datapoints=0):
    train_prob_predictions_target_1, \
        train_prob_predictions_target_0, \
        validation_prob_predictions_target_1, \
        validation_prob_predictions_target_0, \
        training_pred_scores_img_x_target_1, \
        training_pred_scores_img_y_target_1, \
        training_pred_scores_img_x_target_0, \
        training_pred_scores_img_y_target_0, \
        validation_pred_scores_img_x_target_1, \
        validation_pred_scores_img_y_target_1, \
        validation_pred_scores_img_x_target_0, \
        validation_pred_scores_img_y_target_0, \
        = separate_values_based_on_targets(training_targets,
                                           validation_targets,
                                           train_prob_predictions,
                                           validation_prob_predictions,
                                           training_pred_scores_img_x,
                                           training_pred_scores_img_y,
                                           validation_pred_scores_img_x,
                                           validation_pred_scores_img_y
                                           )
    train_x_axis_values_target_1 = [i for i in range(len(train_prob_predictions_target_1))]
    train_x_axis_values_target_0 = [i for i in range(len(train_prob_predictions_target_0))]
    validation_x_axis_values_target_1 = [i for i in range(len(validation_prob_predictions_target_1))]
    validation_x_axis_values_target_0 = [i for i in range(len(validation_prob_predictions_target_0))]

    # Initialize all graphs/subplots
    plt.figure(figsize=(22, 20))
    figure_shape = (7, 2)
    predicted_prob = plt.subplot2grid(figure_shape, (0, 0), rowspan=1, colspan=1)
    loss_per_epoch = plt.subplot2grid(figure_shape, (0, 1), rowspan=1, colspan=1)
    train_scores_histogram = plt.subplot2grid(figure_shape, (1, 0), rowspan=1, colspan=1)
    validation_scores_histogram = plt.subplot2grid(figure_shape, (1, 1), rowspan=1, colspan=1)
    train_residual_histogram = plt.subplot2grid(figure_shape, (2, 0), rowspan=1, colspan=1)
    validation_residual_histogram = plt.subplot2grid(figure_shape, (2, 1), rowspan=1, colspan=1)
    train_target_1_predicted_scores = plt.subplot2grid(figure_shape, (3, 0), rowspan=1, colspan=1)
    validation_target_1_predicted_scores = plt.subplot2grid(figure_shape, (3, 1), rowspan=1, colspan=1)
    train_target_0_predicted_scores = plt.subplot2grid(figure_shape, (4, 0), rowspan=1, colspan=1)
    validation_target_0_predicted_scores = plt.subplot2grid(figure_shape, (4, 1), rowspan=1, colspan=1)
    chronological_pred_scores_target_1_plot = plt.subplot2grid(figure_shape, (5, 0), rowspan=1, colspan=2)
    chronological_pred_sigma_scores_target_1_plot = plt.subplot2grid(figure_shape, (6, 0), rowspan=1, colspan=2)
    # ----------------------------------------------------------------------------------------------------------------#
    # predicted_prob
    predicted_prob.scatter(train_x_axis_values_target_1, train_prob_predictions_target_1,
                           label="Train samples with target 1.0 ({0})".format(len(train_prob_predictions_target_1)),
                           c="#281ad9", s=5)
    predicted_prob.scatter(train_x_axis_values_target_0, train_prob_predictions_target_0,
                           label="Train samples with target 0.0 ({0})".format(len(train_prob_predictions_target_0)),
                           c="#14e33a", s=5)
    predicted_prob.scatter(validation_x_axis_values_target_1, validation_prob_predictions_target_1,
                           label="Validation samples with target 1.0 ({0})".format(
                               len(validation_prob_predictions_target_1)),
                           c="#c9780e", s=5)
    predicted_prob.scatter(validation_x_axis_values_target_0, validation_prob_predictions_target_0,
                           label="Validation samples with target 0.0 ({0})".format(
                               len(validation_prob_predictions_target_0)),
                           c="#f4ff2b", s=5)

    predicted_prob.set_xlabel("Sample")
    predicted_prob.set_ylabel("Predicted Probability")
    predicted_prob.set_title("Sample vs Predicted Probability")
    predicted_prob.legend()

    predicted_prob.autoscale(enable=True, axis='y')

    # training loss, validation loss plot
    epochs_x_axis = [i for i in range(epochs)]
    loss_per_epoch.plot(epochs_x_axis, training_losses,
                        label="Training Loss", c="#281ad9", zorder=0)
    loss_per_epoch.plot(epochs_x_axis, validation_losses,
                        label="Validation Loss", c="#14e33a", zorder=0)

    loss_per_epoch.set_ylim(bottom=0, top=1.0)
    loss_per_epoch.set_xlabel("Epoch")
    loss_per_epoch.set_ylabel("Loss")
    loss_per_epoch.set_title("Loss Per Epoch")
    loss_per_epoch.legend()
    # loss_per_epoch.autoscale(enable=True, axis='y')
    # ----------------------------------------------------------------------------------------------------------------#
    # Training Score Histogram

    training_scores_predictions = np.append(training_pred_scores_img_x, training_pred_scores_img_y)
    train_scores_histogram.set_xlabel("Predicted Score")
    train_scores_histogram.set_ylabel("Frequency")
    train_scores_histogram.set_title("Train Scores Histogram")
    train_scores_histogram.hist(training_scores_predictions,
                                weights=np.ones(len(training_scores_predictions)) / len(training_scores_predictions))
    train_scores_histogram.yaxis.set_major_formatter(PercentFormatter(1))

    # Validation Score Histogram
    validation_scores_predictions = np.append(validation_pred_scores_img_x, validation_pred_scores_img_y)

    validation_scores_histogram.set_xlabel("Predicted Score")
    validation_scores_histogram.set_ylabel("Frequency")
    validation_scores_histogram.set_title("Validation Scores Histogram")
    validation_scores_histogram.hist(validation_scores_predictions,
                                     weights=np.ones(len(validation_scores_predictions)) / len(
                                         validation_scores_predictions))
    validation_scores_histogram.yaxis.set_major_formatter(PercentFormatter(1))
    # ----------------------------------------------------------------------------------------------------------------#

    # Training Residuals
    # Calculate train residuals
    training_residuals_target_1 = [abs(1.0 - train_prob_predictions_target_1[i].item()) for i in
                                   range(len(train_prob_predictions_target_1))]
    training_residuals_target_0 = [abs(0.0 - train_prob_predictions_target_0[i].item()) for i in
                                   range(len(train_prob_predictions_target_0))]

    # assert max(training_residuals_target_0) <= 1.0
    # assert min(training_residuals_target_0) >= 0.0
    # assert max(training_residuals_target_1) <= 1.0
    # assert min(training_residuals_target_1) >= 0.0

    # print("training pred prob target 1=", train_prob_predictions_target_1)
    # print("training pred prob target 0=", train_prob_predictions_target_0)
    # print("training residuals 1=", training_residuals_target_1)
    # print("training residuals 0=", training_residuals_target_0)
    training_residuals = np.append(training_residuals_target_1, training_residuals_target_0)

    train_residual_histogram.set_xlabel("Residual")
    train_residual_histogram.set_ylabel("Frequency")
    train_residual_histogram.set_title("Train Residual Histogram")
    train_residual_histogram.hist(training_residuals, range=(0.0, 1.0),
                                  weights=np.ones(len(training_residuals)) / len(training_residuals))
    train_residual_histogram.yaxis.set_major_formatter(PercentFormatter(1))

    # Calculate validation residuals
    validation_residuals_target_1 = [abs(1.0 - validation_prob_predictions_target_1[i].item()) for i in
                                     range(len(validation_prob_predictions_target_1))]
    validation_residuals_target_0 = [abs(0.0 - validation_prob_predictions_target_0[i].item()) for i in
                                     range(len(validation_prob_predictions_target_0))]

    # assert max(validation_residuals_target_0) <= 1.0
    # assert min(validation_residuals_target_0) >= 0.0
    # assert max(validation_residuals_target_1) <= 1.0
    # assert min(validation_residuals_target_1) >= 0.0
    validation_residuals = np.append(validation_residuals_target_1, validation_residuals_target_0)

    validation_residual_histogram.set_xlabel("Residual")
    validation_residual_histogram.set_ylabel("Frequency")
    validation_residual_histogram.set_title("Validation Residual Histogram")
    validation_residual_histogram.hist(validation_residuals, range=(0.0, 1.0),
                                       weights=np.ones(len(validation_residuals)) / len(validation_residuals))
    validation_residual_histogram.yaxis.set_major_formatter(PercentFormatter(1))
    # ----------------------------------------------------------------------------------------------------------------#

    # train_target_1_predicted_scores
    train_target_1_predicted_scores.scatter(train_x_axis_values_target_1, training_pred_scores_img_x_target_1,
                                            label="Train selected image scores with target 1.0 ({0})".format(
                                                len(training_pred_scores_img_x_target_1)),
                                            c="#281ad9", s=5)
    train_target_1_predicted_scores.scatter(train_x_axis_values_target_1, training_pred_scores_img_y_target_1,
                                            label="Train unselected image scores with target 1.0 ({0})".format(
                                                len(training_pred_scores_img_y_target_1)),
                                            c="#14e33a", s=5)

    train_target_1_predicted_scores.set_xlabel("Sample")
    train_target_1_predicted_scores.set_ylabel("Predicted Score")
    train_target_1_predicted_scores.set_title("Train Predicted Score for target 1.0")
    train_target_1_predicted_scores.legend()

    # validation_target_1_predicted_scores
    validation_target_1_predicted_scores.scatter(validation_x_axis_values_target_1,
                                                 validation_pred_scores_img_x_target_1,
                                                 label="Validation selected image scores with target 1.0 ({0})".format(
                                                     len(validation_pred_scores_img_x_target_1)),
                                                 c="#281ad9", s=5)
    validation_target_1_predicted_scores.scatter(validation_x_axis_values_target_1,
                                                 validation_pred_scores_img_y_target_1,
                                                 label="Validation unselected image scores with target 1.0 ({0})".format(
                                                     len(validation_pred_scores_img_y_target_1)),
                                                 c="#14e33a", s=5)

    validation_target_1_predicted_scores.set_xlabel("Sample")
    validation_target_1_predicted_scores.set_ylabel("Predicted Score")
    validation_target_1_predicted_scores.set_title("Validation Predicted Score for target 1.0")
    validation_target_1_predicted_scores.legend()

    # ----------------------------------------------------------------------------------------------------------------#
    # train_target_0_predicted_scores
    train_target_0_predicted_scores.scatter(train_x_axis_values_target_0, training_pred_scores_img_x_target_0,
                                            label="Train selected image scores with target 0.0 ({0})".format(
                                                len(training_pred_scores_img_x_target_0)),
                                            c="#281ad9", s=5)
    train_target_0_predicted_scores.scatter(train_x_axis_values_target_0, training_pred_scores_img_y_target_0,
                                            label="Train unselected image scores with target 0.0 ({0})".format(
                                                len(training_pred_scores_img_y_target_0)),
                                            c="#14e33a", s=5)

    train_target_0_predicted_scores.set_xlabel("Sample")
    train_target_0_predicted_scores.set_ylabel("Predicted Score")
    train_target_0_predicted_scores.set_title("Train Predicted Score for target 0.0")
    train_target_0_predicted_scores.legend()

    # validation_target_0_predicted_scores
    validation_target_0_predicted_scores.scatter(validation_x_axis_values_target_0,
                                                 validation_pred_scores_img_x_target_0,
                                                 label="Validation selected image scores with target 0.0 ({0})".format(
                                                     len(validation_pred_scores_img_x_target_0)),
                                                 c="#281ad9", s=5)
    validation_target_0_predicted_scores.scatter(validation_x_axis_values_target_0,
                                                 validation_pred_scores_img_y_target_0,
                                                 label="Validation unselected image scores with target 0.0 ({0})".format(
                                                     len(validation_pred_scores_img_y_target_0)),
                                                 c="#14e33a", s=5)

    validation_target_0_predicted_scores.set_xlabel("Sample")
    validation_target_0_predicted_scores.set_ylabel("Predicted Score")
    validation_target_0_predicted_scores.set_title("Validation Predicted Score for target 0.0")
    validation_target_0_predicted_scores.legend()
    # ----------------------------------------------------------------------------------------------------------------#
    # chronological scores graph
    chronological_pred_scores_img_x_target_1 = [None] * int(
        (len(training_pred_scores_img_x) + len(validation_pred_scores_img_x)) / 2)
    chronological_pred_scores_img_x_target_0 = [None] * int(
        (len(training_pred_scores_img_x) + len(validation_pred_scores_img_x)) / 2)

    count = 0
    for score in training_pred_scores_img_x:
        if training_targets[count] == [0.0]:
            chronological_pred_scores_img_x_target_0[training_shuffled_indices_origin[count]] = score.item()
        else:
            chronological_pred_scores_img_x_target_1[training_shuffled_indices_origin[count]] = score.item()
        count += 1
    count = 0

    for score in validation_pred_scores_img_x:
        if validation_targets[count] == [0.0]:
            chronological_pred_scores_img_x_target_0[validation_shuffled_indices_origin[count]] = score.item()
        else:
            chronological_pred_scores_img_x_target_1[validation_shuffled_indices_origin[count]] = score.item()
        count += 1

    chronological_pred_scores_img_y_target_1 = [None] * int(
        (len(training_pred_scores_img_y) + len(validation_pred_scores_img_y)) / 2)

    count = 0
    for score in training_pred_scores_img_y:
        if training_targets[count] == [1.0]:
            chronological_pred_scores_img_y_target_1[training_shuffled_indices_origin[count]] = score.item()
        count += 1
    count = 0

    for score in validation_pred_scores_img_y:
        if validation_targets[count] == [1.0]:
            chronological_pred_scores_img_y_target_1[validation_shuffled_indices_origin[count]] = score.item()
        count += 1

    chronological_pred_scores_target_1_plot.scatter([i for i in range(len(chronological_pred_scores_img_x_target_1))],
                                                    chronological_pred_scores_img_x_target_1,
                                                    label="Chronological predicted image x scores ({0})".format(
                                                        len(chronological_pred_scores_img_x_target_1)),
                                                    c="#281ad9", s=5)
    chronological_pred_scores_target_1_plot.scatter([i for i in range(len(chronological_pred_scores_img_y_target_1))],
                                                    chronological_pred_scores_img_y_target_1,
                                                    label="Chronological predicted image y scores ({0})".format(
                                                        len(chronological_pred_scores_img_y_target_1)),
                                                    c="#14e33a", s=5)

    chronological_pred_scores_target_1_plot.set_xlabel("Sample")
    chronological_pred_scores_target_1_plot.set_ylabel("Predicted Score")
    chronological_pred_scores_target_1_plot.set_title("Chronological Dataset Predicted Score")
    chronological_pred_scores_target_1_plot.legend()

    # -------------------------------------------------- Sigma Scores Graph ------------------------------------------#
    if x_chronological_sigma_scores is not None and y_chronological_sigma_scores is not None:
        chronological_pred_sigma_scores_target_1_plot.scatter([i for i in range(len(x_chronological_sigma_scores))],
                                                        x_chronological_sigma_scores,
                                                        label="Chronological image x sigma scores ({0})".format(
                                                            len(x_chronological_sigma_scores)),
                                                        c="#281ad9", s=5)
        chronological_pred_sigma_scores_target_1_plot.scatter([i for i in range(len(y_chronological_sigma_scores))],
                                                        y_chronological_sigma_scores,
                                                        label="Chronological image y sigma scores ({0})".format(
                                                            len(y_chronological_sigma_scores)),
                                                        c="#14e33a", s=5)

        chronological_pred_sigma_scores_target_1_plot.set_xlabel("Sample")
        chronological_pred_sigma_scores_target_1_plot.set_ylabel("Sigma Score")
        chronological_pred_sigma_scores_target_1_plot.set_title("Chronological Dataset Sigma Score")
        chronological_pred_sigma_scores_target_1_plot.legend()

    # ----------------------------------------------------------------------------------------------------------------#
    if learning_rate == -1:
        learning_rate = "N/A"

    if training_batch_size == -1:
        training_batch_size = "N/A"

    if weight_decay == -1:
        weight_decay = "N/A"

    if loss_func == -1:
        loss_func = "N/A"

    pooling_strategy_str = "N/A"
    if pooling_strategy == 0 and input_type != "clip":
        pooling_strategy_str = "average pooling"
    elif pooling_strategy == 1 and input_type != "clip":
        pooling_strategy_str = "max pooling"
    elif pooling_strategy == 2 and input_type != "clip":
        pooling_strategy_str = "max abs pooling"

    if normalize_vectors == -1:
        normalize_vectors = "N/A"

    if num_random_layers == -1:
        num_random_layers = "N/A"

    if add_loss_penalty == -1:
        add_loss_penalty = "N/A"

    target_option_str = "N/A"
    if target_option == 0:
        target_option_str = "target 1 and 0"
    elif target_option == 1:
        target_option_str = "target 1 only"
    elif target_option == 2:
        target_option_str = "target 0 only"

    duplicate_flip_option_str = "N/A"
    if duplicate_flip_option == 0:
        duplicate_flip_option_str = "duplicate and flip all"
    elif duplicate_flip_option == 1:
        duplicate_flip_option_str = "duplicate and flip random"

    if randomize_data_per_epoch == -1:
        randomize_data_per_epoch = "N/A"

    if elm_sparsity == -1:
        elm_sparsity = "N/A"

    # add additional info on top left side
    plt.figtext(0, 0.55, "Date = {}\n"
                         "Dataset = {}\n"
                         "Network type = {}\n"
                         "Input type = {}\n"
                         "Input shape = {}\n"
                         "Output type= {}\n\n"
                         ""
                         "Total Selection \n"
                         "Datapoints = {}\n"
                         "Training size = {}\n"
                         "Validation size = {}\n\n"
                         ""
                         "Learning rate = {}\n"
                         "Epochs = {}\n"
                         "Training batch size = {}\n"
                         "Weight decay = {}\n"
                         "Loss func = {}\n\n"
                         ""
                         "Pooling strategy = {}\n"
                         "Normalize vectors = {}\n"
                         "num_random_layers = {}\n"
                         "add_loss_penalty = {}\n"
                         "target_option = {}\n"
                         "duplicate_flip_option = {}\n"
                         "randomize_data_per_epoch = {}\n"
                         "elm_sparsity = {}\n\n"
                         ""
                         "Training loss = {:03.04}\n"
                         "Validation loss = {:03.04}\n"
                         "Train Correct Predictions \n"
                         "= {}({:02.02f}%)\n"
                         "Validation Correct \n"
                         "Predictions = {}({:02.02f}%)\n\n".format(date,
                                                                   dataset_name,
                                                                   network_type,
                                                                   input_type,
                                                                   input_shape,
                                                                   output_type,
                                                                   total_selection_datapoints,
                                                                   training_total_size,
                                                                   validation_total_size,
                                                                   learning_rate,
                                                                   epochs,
                                                                   training_batch_size,
                                                                   weight_decay,
                                                                   loss_func,
                                                                   pooling_strategy_str,
                                                                   normalize_vectors,
                                                                   num_random_layers,
                                                                   add_loss_penalty,
                                                                   target_option_str,
                                                                   duplicate_flip_option_str,
                                                                   randomize_data_per_epoch,
                                                                   elm_sparsity,
                                                                   training_loss,
                                                                   validation_loss,
                                                                   train_sum_correct,
                                                                   (train_sum_correct / training_total_size) * 100,
                                                                   validation_sum_correct,
                                                                   (
                                                                               validation_sum_correct / validation_total_size) * 100)
                )

    # Save figure
    # graph_path = os.path.join(model_output_path, graph_name)
    plt.subplots_adjust(left=0.15, hspace=0.5)
    # plt.savefig(graph_path)
    # plt.show()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return buf
