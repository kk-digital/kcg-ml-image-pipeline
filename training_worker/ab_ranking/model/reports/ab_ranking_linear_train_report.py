def get_train_report(model_class, dataset_path, training_percent, training_size, validation_size, train_sum_correct,
                     validation_sum_correct, nn_summary, training_pred_scores_img_x, training_pred_scores_img_y,
                     validation_pred_scores_img_x, validation_pred_scores_img_y, training_batch_size, learning_rate,
                     weight_decay, image_selected_index_0_count, image_selected_index_1_count, image_selected_total_count):
    report_str = "Training Report:\n\n"
    report_str += "model: {}\n".format(model_class.model_type)
    report_str += "model-date: {}\n".format(model_class.date)
    report_str += "training batch size: {}\n".format(training_batch_size)
    report_str += "learning rate: {}\n".format(learning_rate)
    report_str += "weight decay: {}\n".format(weight_decay)

    report_str += "\n=================Dataset Info==================\n"
    report_str += "dataset-path: {}\n".format(dataset_path)
    report_str += "training-data-count: {0} ({1:04}%)\n".format(training_size,
                                                                 round(training_percent * 100, 2))
    report_str += "validation-data-count: {0} ({1:04}%)\n".format(validation_size,
                                                                   round((1 - training_percent) * 100, 2))

    report_str += '\nloss-func: {}\n'.format(model_class.loss_func_name)
    report_str += 'training-loss: {:03.04}\n'.format(model_class.training_loss)
    report_str += 'validation-loss: {:03.04}\n'.format(model_class.validation_loss)

    report_str += '\nimage-selected-index-0-count: {0}({1:04}%)\n'.format(image_selected_index_0_count, round((image_selected_index_0_count/image_selected_total_count)*100, 2))
    report_str += 'image-selected-index-1-count: {0}({1:04}%)\n'.format(image_selected_index_1_count, round((image_selected_index_1_count/image_selected_total_count)*100, 2))

    report_str += "\n===================NN Info=====================\n\n"
    report_str += nn_summary

    report_str += "Correct Prediction is " \
                  "when score(x) > score(y) if x was the selected image\n" \
                  "and when score(x) < score(y) if y was the selected image\n\n"
    report_str += 'Train number of correct predictions: {}\n'.format(train_sum_correct)
    report_str += "Training-data-count: {0}\n".format(training_size)
    report_str += 'Train percent of correct predictions: {0:03.04f}\n'.format(train_sum_correct / training_size)

    report_str += '\nValidation number of correct predictions: {}\n'.format(validation_sum_correct)
    report_str += "Validation-data-count: {0}\n".format(validation_size)
    report_str += 'Validation percent of correct predictions: {0:03.04f}\n'.format(
        validation_sum_correct / validation_size)

    report_str += '\nTraining score range:\n'
    report_str += 'image x: {} to {}\n'.format(min(training_pred_scores_img_x),
                                               max(training_pred_scores_img_x))
    report_str += 'image y: {} to {}\n'.format(min(training_pred_scores_img_y),
                                               max(training_pred_scores_img_y))

    report_str += '\nValidation score range:\n'
    report_str += 'image x: {} to {}\n'.format(min(validation_pred_scores_img_x),
                                               max(validation_pred_scores_img_x))
    report_str += 'image y: {} to {}\n'.format(min(validation_pred_scores_img_y),
                                               max(validation_pred_scores_img_y))

    return report_str
