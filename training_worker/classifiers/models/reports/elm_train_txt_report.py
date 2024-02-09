def get_train_txt_report(model_class,
                         min_prediction_vector_before_sigmoid,
                         max_prediction_vector_before_sigmoid,
                         validation_predictions,
                         mse_training_loss,
                         training_accuracy,
                         mse_validation_loss,
                         validation_accuracy,
                         bce_loss_training,
                         bce_loss_validation,
                         training_percent,
                         positive_dataset_total_size,
                         negative_dataset_total_size,
                         training_positive_size,
                         validation_positive_size,
                         training_negative_size,
                         validation_negative_size):
    report_str = "Training Report:\n\n"
    report_str += "classifier: {}\n".format(model_class.model_type)
    report_str += "transfer-function: {}\n".format(model_class.activation_func_name)
    report_str += "model-date: {}\n".format(model_class.date)
    report_str += "tag-string: {}\n".format(model_class.tag_string)

    report_str += "\n=================Dataset Info==================\n"

    report_str += "\npositive-total-image-count: {0}\n".format(positive_dataset_total_size)
    report_str += "negative-total-image-count: {0}\n".format(negative_dataset_total_size)
    report_str += "training-positive-image-used: {0} ({1:03.02f})\n".format(training_positive_size,
                                                                          round(training_percent, 2))
    report_str += "training-negative-image-used: {0} ({1:03.02f})\n".format(training_negative_size,
                                                                         round(training_percent, 2))
    report_str += "validation-positive-image-used: {0} ({1:03.02f})\n".format(validation_positive_size,
                                                                            round((1 - training_percent), 2))
    report_str += "validation-negative-image-used: {0} ({1:03.02f})\n".format(validation_negative_size,
                                                                           round((1 - training_percent), 2))

    report_str += "\n======== ELM Classifier Model Architecture ======\n"
    report_str += "Input Layer:\n"
    report_str += "  Name: input layer\n"
    report_str += "  Shape: ({})\n".format(model_class._input_size)

    report_str += "\n================  Hidden layer ================\n"
    report_str += "Hidden Layer:\n"
    report_str += "  Name: hidden layer\n"
    report_str += "  Shape: ({})\n".format(model_class._hidden_layer_neuron_count)
    report_str += "  Activation Function: {}\n".format(model_class.activation_func_name)
    report_str += "  Input to Hidden Layer Size: ({}, {})\n".format(model_class._input_size,
                                                                    model_class._hidden_layer_neuron_count)
    report_str += "  Weight Matrix Size: ({}, {})\n".format(model_class._input_size,
                                                            model_class._hidden_layer_neuron_count)
    report_str += "  Bias Matrix Size: ({})\n".format(model_class._hidden_layer_neuron_count)
    report_str += "  Input to Hidden Layer: ({}, {})\n".format(model_class._input_size,
                                                               model_class._hidden_layer_neuron_count)
    report_str += "  Output of Hidden Layer: ({}, {})\n".format(model_class._input_size,
                                                                model_class._hidden_layer_neuron_count)

    report_str += "\n================  Output layer ================\n"
    report_str += "Output Layer:\n"
    report_str += "  Name: output layer\n"
    report_str += "  Shape: ({})\n".format(model_class._output_size)
    report_str += "  Output of Output Layer: {}\n".format(model_class._output_size)
    report_str += "\n===============================================\n"

    report_str += '\nloss-func: {}\n'.format(model_class.loss_func_name)
    report_str += 'training-accuracy: {:06.04f}\n'.format(training_accuracy)
    report_str += 'validation-accuracy: {:06.04f}\n'.format(validation_accuracy)
    report_str += 'loss-mse-training: {:06.04f}\n'.format(mse_training_loss)
    report_str += 'loss-mse-validation: {:06.04f}\n'.format(mse_validation_loss)
    report_str += "output-min-before-sigmoid: {:06.04f}\n".format(min_prediction_vector_before_sigmoid)
    report_str += "output-max-before-sigmoid: {:06.04f}\n".format(max_prediction_vector_before_sigmoid)
    report_str += "output-min-after-sigmoid: {:06.04f}\n".format(min(validation_predictions).item())
    report_str += "output-max-after-sigmoid: {:06.04f}\n".format(max(validation_predictions).item())
    report_str += 'loss-bce-training: {:06.04f}\n'.format(bce_loss_training)
    report_str += 'loss-bce-validation: {:06.04f}\n'.format(bce_loss_validation)

    report_str += print_histogram_ascii(validation_predictions)

    return report_str


def print_histogram_ascii(validation_predictions):
    values_arr = []
    for data in validation_predictions:
        values_arr.append(data.item())

    bar_headers = ["0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9",
                   "0.9-1.0"]

    # calculate bar values
    bars = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for value in values_arr:
        if value < 0.1:
            bars[0] += 1
        elif 0.1 <= value < 0.2:
            bars[1] += 1
        elif 0.2 <= value < 0.3:
            bars[2] += 1
        elif 0.3 <= value < 0.4:
            bars[3] += 1
        elif 0.4 <= value < 0.5:
            bars[4] += 1
        elif 0.5 <= value < 0.6:
            bars[5] += 1
        elif 0.6 <= value < 0.7:
            bars[6] += 1
        elif 0.7 <= value < 0.8:
            bars[7] += 1
        elif 0.8 <= value < 0.9:
            bars[8] += 1
        elif value >= 0.9:
            bars[9] += 1

    # calculate percentage per bar
    percent_bars = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(bars)):
        percent_bars[i] = bars[i] / len(values_arr)
        percent_bars[i] = percent_bars[i] * 100

    # normalize bars value
    max_bars = max(bars)
    min_bars = min(bars)
    diff = max_bars - min_bars
    for i in range(len(bars)):
        bars[i] = (bars[i] - min_bars) / diff

    # scale to 50
    for i in range(len(bars)):
        bars[i] = bars[i] * 50

    report_str = "\n=================Histogram=====================\n"
    report_str += "Range     |   Percent  |   Histogram"
    for i in range(len(bars)):
        report_str += "\n{0}   |   {1:05.02f}%   | ".format(bar_headers[i], round(percent_bars[i], 2))
        for _ in range(int(round(bars[i]))):
            report_str += "* "

    return report_str
