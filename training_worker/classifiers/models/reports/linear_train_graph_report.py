import os
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from matplotlib.ticker import PercentFormatter

def get_graph_report(model_class,
                     train_predictions,
                     training_targets,
                     validation_predictions,
                     validation_targets,
                     date,
                     dataset_name,
                     network_type,
                     input_type,
                     input_shape,
                     output_type,
                     tag_string,
                     positive_dataset_total_size,
                     negative_dataset_total_size,
                     training_positive_size,
                     validation_positive_size,
                     training_negative_size,
                     validation_negative_size,
                     pooling_strategy,
                     training_loss_per_epoch,
                     validation_loss_per_epoch):
    # Initialize all graphs/subplots
    plt.figure(figsize=(22, 20))
    figure_shape = (3, 2)
    predicted_score = plt.subplot2grid(figure_shape, (0, 0), rowspan=1, colspan=1)
    loss_per_epoch = plt.subplot2grid(figure_shape, (0, 1), rowspan=1, colspan=1)
    training_predicted_score_hist = plt.subplot2grid(figure_shape, (1, 0), rowspan=1, colspan=1)
    validation_predicted_score_hist = plt.subplot2grid(figure_shape, (1, 1), rowspan=1, colspan=1)
    train_residual_histogram = plt.subplot2grid(figure_shape, (2, 0), rowspan=1, colspan=1)
    validation_residual_histogram = plt.subplot2grid(figure_shape, (2, 1), rowspan=1, colspan=1)

    # --------------------------------------------------------------------------------------------------------

    train_predictions_target_1 = []
    train_predictions_target_0 = []
    for i in range(len(train_predictions)):
        if training_targets[i] == [1.0]:
            train_predictions_target_1.append(train_predictions[i])
        else:
            train_predictions_target_0.append(train_predictions[i])

    validation_predictions_target_1 = []
    validation_predictions_target_0 = []
    for i in range(len(validation_predictions)):
        if validation_targets[i] == [1.0]:
            validation_predictions_target_1.append(validation_predictions[i])
        else:
            validation_predictions_target_0.append(validation_predictions[i])

    train_x_axis_values_target_1 = [i for i in range(len(train_predictions_target_1))]
    train_x_axis_values_target_0 = [i for i in range(len(train_predictions_target_0))]
    validation_x_axis_values_target_1 = [i for i in range(len(validation_predictions_target_1))]
    validation_x_axis_values_target_0 = [i for i in range(len(validation_predictions_target_0))]

    # predicted_score
    predicted_score.scatter(train_x_axis_values_target_1, train_predictions_target_1,
                           label="Train samples with target 1.0 ({0})".format(len(train_predictions_target_1)),
                           c="#281ad9", s=5)
    predicted_score.scatter(train_x_axis_values_target_0, train_predictions_target_0,
                           label="Train samples with target 0.0 ({0})".format(len(train_predictions_target_0)),
                           c="#14e33a", s=5)
    predicted_score.scatter(validation_x_axis_values_target_1, validation_predictions_target_1,
                           label="Validation samples with target 1.0 ({0})".format(
                               len(validation_predictions_target_1)),
                           c="#c9780e", s=5)
    predicted_score.scatter(validation_x_axis_values_target_0, validation_predictions_target_0,
                           label="Validation samples with target 0.0 ({0})".format(
                               len(validation_predictions_target_0)),
                           c="#f4ff2b", s=5)

    predicted_score.set_xlabel("Sample")
    predicted_score.set_ylabel("Predicted Score")
    predicted_score.set_title("Sample vs Predicted Score")
    predicted_score.legend()

    predicted_score.autoscale(enable=True, axis='y')
    # --------------------------------------------------------------------------------------------------------
    # training loss, validation loss plot
    epochs_x_axis = [i for i in range(model_class.epochs)]
    loss_per_epoch.plot(epochs_x_axis, training_loss_per_epoch,
                        label="Training Loss", c="#281ad9", zorder=0)
    loss_per_epoch.plot(epochs_x_axis, validation_loss_per_epoch,
                        label="Validation Loss", c="#14e33a", zorder=0)

    loss_per_epoch.set_ylim(bottom=0, top=1.0)
    loss_per_epoch.set_xlabel("Epoch")
    loss_per_epoch.set_ylabel("Loss")
    loss_per_epoch.set_title("Loss Per Epoch")
    loss_per_epoch.legend()

    # --------------------------------------------------------------------------------------------------------
    # Training Score Histogram
    training_predicted_score_hist.set_xlabel("Predicted Score")
    training_predicted_score_hist.set_ylabel("Frequency")
    training_predicted_score_hist.set_title("Training Predicted Scores Histogram")
    training_predicted_score_hist.hist(train_predictions, range=(0.0, 1.0),
                                         weights=np.ones(len(train_predictions)) / len(train_predictions))
    training_predicted_score_hist.yaxis.set_major_formatter(PercentFormatter(1))

    # Validation Score Histogram
    validation_predicted_score_hist.set_xlabel("Predicted Score")
    validation_predicted_score_hist.set_ylabel("Frequency")
    validation_predicted_score_hist.set_title("Validation Predicted Scores Histogram")
    validation_predicted_score_hist.hist(validation_predictions, range=(0.0, 1.0),
                                weights=np.ones(len(validation_predictions)) / len(validation_predictions))
    validation_predicted_score_hist.yaxis.set_major_formatter(PercentFormatter(1))

    # --------------------------------------------------------------------------------------------------------
    # Training Residual Histogram
    # Calculate train residuals
    training_residuals_target_1 = [abs(1.0 - train_predictions_target_1[i].item()) for i in
                                   range(len(train_predictions_target_1))]
    training_residuals_target_0 = [abs(0.0 - train_predictions_target_0[i].item()) for i in
                                   range(len(train_predictions_target_0))]

    training_residuals = np.append(training_residuals_target_1, training_residuals_target_0)

    train_residual_histogram.set_xlabel("Residual")
    train_residual_histogram.set_ylabel("Frequency")
    train_residual_histogram.set_title("Train Residual Histogram")
    train_residual_histogram.hist(training_residuals, range=(0.0, 1.0),
                                  weights=np.ones(len(training_residuals)) / len(training_residuals))
    train_residual_histogram.yaxis.set_major_formatter(PercentFormatter(1))

    # Validation Residual Histogram
    # Calculate Validation residuals
    validation_residuals_target_1 = [abs(1.0 - validation_predictions_target_1[i].item()) for i in
                                   range(len(validation_predictions_target_1))]
    validation_residuals_target_0 = [abs(0.0 - validation_predictions_target_0[i].item()) for i in
                                   range(len(validation_predictions_target_0))]

    validation_residuals = np.append(validation_residuals_target_1, validation_residuals_target_0)

    validation_residual_histogram.set_xlabel("Residual")
    validation_residual_histogram.set_ylabel("Frequency")
    validation_residual_histogram.set_title("Validation Residual Histogram")
    validation_residual_histogram.hist(validation_residuals, range=(0.0, 1.0),
                                  weights=np.ones(len(validation_residuals)) / len(validation_residuals))
    validation_residual_histogram.yaxis.set_major_formatter(PercentFormatter(1))

    # --------------------------------------------------------------------------------------------------------
    pooling_strategy_str = "N/A"
    if pooling_strategy == 0 and input_type != "clip":
        pooling_strategy_str = "average pooling"
    elif pooling_strategy == 1 and input_type != "clip":
        pooling_strategy_str = "max pooling"
    elif pooling_strategy == 2 and input_type != "clip":
        pooling_strategy_str = "max abs pooling"

    # add additional info on top left side
    plt.figtext(0, 0.55, "Date = {}\n"
                         "Tag Name={}\n"
                         "Dataset = {}\n"
                         "Network type = {}\n"
                         "Input type = {}\n"
                         "Input shape = {}\n"
                         "Output type= {}\n"
                         ""
                         "positive-total-image-count = {}\n"
                         "negative-total-image-count = {}\n"
                         "training-positive-image-used = {}\n"
                         "training-negative-image-used = {}\n"
                         "validation-positive-image-used = {}\n"
                         "validation-negative-image-used = {}\n\n"
                         ""
                         "Pooling strategy = {}\n"
                         ""
                         "loss-func = {}\n"
                         "epochs= {}\n"
                         "learning_rate= {}\n"
                         "loss_func_name= {}\n"
                         "normalize_feature_vectors= {}\n"
                         "training-loss = {:03.04}\n"
                         "validation-loss = {:03.04}\n"
                         "\n"
                         "\n\n".format(date,
                                       tag_string,
                                       dataset_name,
                                       network_type,
                                       input_type,
                                       input_shape,
                                       output_type,
                                       positive_dataset_total_size,
                                       negative_dataset_total_size,
                                       training_positive_size,
                                       validation_positive_size,
                                       training_negative_size,
                                       validation_negative_size,
                                       pooling_strategy_str,
                                       model_class.loss_func_name,
                                       model_class.epochs,
                                       model_class.learning_rate,
                                       model_class.loss_func_name,
                                       model_class.normalize_feature_vectors,
                                       model_class.training_loss,
                                       model_class.validation_loss,
                                       ),
                )
    plt.subplots_adjust(left=0.15, hspace=0.5)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return buf

