def get_train_txt_report(model_class,
                         validation_predictions,
                         training_percent,
                         positive_dataset_total_size,
                         negative_dataset_total_size,
                         training_positive_size,
                         validation_positive_size,
                         training_negative_size,
                         validation_negative_size,
                         nn_summary):
    report_str = "Training Report:\n\n"
    report_str += "classifier: {}\n".format(model_class.model_type)
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

    report_str += '\nloss-func: {}\n'.format(model_class.loss_func_name)
    report_str += 'training loss: {}\n'.format(model_class.training_loss)
    report_str += 'validation loss: {}\n'.format(model_class.validation_loss)

    report_str += print_histogram_ascii(validation_predictions)

    report_str += "\n===================NN Info=====================\n\n"
    report_str += nn_summary

    return report_str


def print_histogram_ascii(validation_predictions):
    values_arr = []
    for data in validation_predictions:
        values_arr.append(data.item())

    report_str = "min-prediction-vector: {}\n".format(min(values_arr))
    report_str += "max-prediction-vector: {}\n".format(max(values_arr))

    bar_headers = ["0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]

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

    report_str += "\n=================Histogram=====================\n"
    report_str += "Range     |   Percent  |   Histogram"
    for i in range(len(bars)):
        report_str += "\n{0}   |   {1:05.02f}%   | ".format(bar_headers[i], round(percent_bars[i], 2))
        for _ in range(int(round(bars[i]))):
            report_str += "* "

    return report_str
