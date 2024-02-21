import sys
import os
from datetime import datetime
from pytz import timezone
import torch.nn as nn
from io import BytesIO

base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from data_loader.tagged_data_loader import TaggedDatasetLoader
from utility.minio import cmd
from training_worker.ab_ranking.model import constants
from training_worker.classifiers.models import elm_regression
from training_worker.classifiers.models.reports.elm_train_graph_report import get_graph_report
from training_worker.classifiers.models.reports.elm_train_txt_report import get_train_txt_report

def train_classifier(minio_ip_addr=None,
                     minio_access_key=None,
                     minio_secret_key=None,
                     input_type="embedding",
                     tag_name=None,
                     hidden_layer_neuron_count=3000,
                     pooling_strategy=constants.AVERAGE_POOLING,
                     train_percent=0.9,
                    ):
    date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%Y-%m-%d')
    print("Current datetime: {}".format(datetime.now(tz=timezone("Asia/Hong_Kong"))))
    bucket_name = "datasets"
    network_type = "elm-regression"
    output_type = "score"

    # check input type
    if input_type not in constants.ALLOWED_INPUT_TYPES:
        raise Exception("input type is not supported: {}".format(input_type))

    input_shape = 2 * 768
    if input_type in [constants.EMBEDDING_POSITIVE, constants.EMBEDDING_NEGATIVE, constants.CLIP]:
        input_shape = 768

    # load data
    tag_loader = TaggedDatasetLoader(minio_ip_addr=minio_ip_addr,
                                     minio_access_key=minio_access_key,
                                     minio_secret_key=minio_secret_key,
                                     tag_name=tag_name,
                                     input_type=input_type,
                                     pooling_strategy=pooling_strategy,
                                     train_percent=train_percent)
    tag_loader.load_dataset()

    # get dataset name
    dataset_name = tag_loader.dataset_name
    output_path = "{}/models/classifiers/{}".format(dataset_name, tag_name)

    # mix training positive and negative
    training_features, training_targets = tag_loader.get_shuffled_positive_and_negative_training()

    # mix validation positive and negative
    validation_features, validation_targets = tag_loader.get_shuffled_positive_and_negative_validation()

    # get final filename
    sequence = 0
    filename = "{}-{:02}-{}-{}-{}-{}".format(date_now, sequence, tag_name, output_type, network_type, input_type)

    # if exist, increment sequence
    while True:
        filename = "{}-{:02}-{}-{}-{}-{}".format(date_now, sequence, tag_name, output_type, network_type, input_type)
        exists = cmd.is_object_exists(tag_loader.minio_client, bucket_name,
                                      os.path.join(output_path, filename + ".safetensors"))
        if not exists:
            break

        sequence += 1

    # train model
    classifier_model = elm_regression.ELMRegression()
    classifier_model.set_config(tag_string=tag_name,
                                input_size=input_shape,
                                hidden_layer_neuron_count=hidden_layer_neuron_count,
                                output_size=1,
                                activation_func_name="sigmoid")

    (train_pred,
     training_loss,
     training_accuracy,
     validation_pred,
     validation_loss,
     validation_accuracy) = classifier_model.train(tag_loader=tag_loader)

    # sigmoid
    sigmoid = nn.Sigmoid()
    sigmoid_pred = sigmoid(validation_pred)
    sigmoid_train_pred = sigmoid(train_pred)

    # bce loss
    bce_loss_func = nn.BCELoss()
    bce_loss_training = bce_loss_func(sigmoid_train_pred.to('cpu'), training_targets)
    bce_loss_validation = bce_loss_func(sigmoid_pred.to('cpu'), validation_targets)

    # save model
    # Upload model to minio
    model_name = "{}.safetensors".format(filename)
    model_output_path = os.path.join(output_path, model_name)
    classifier_model.save_model(tag_loader.minio_client, bucket_name, model_output_path)

    # upload the txt report
    report_str = get_train_txt_report(model_class=classifier_model,
                                      min_prediction_vector_before_sigmoid=min(validation_pred).item(),
                                      max_prediction_vector_before_sigmoid=max(validation_pred).item(),
                                      validation_predictions=sigmoid_pred,
                                      mse_training_loss=training_loss,
                                      training_accuracy=training_accuracy,
                                      mse_validation_loss=validation_loss,
                                      validation_accuracy=validation_accuracy,
                                      bce_loss_training=bce_loss_training,
                                      bce_loss_validation=bce_loss_validation,
                                      training_percent=train_percent,
                                      positive_dataset_total_size=tag_loader.positive_dataset_total_size,
                                      negative_dataset_total_size=tag_loader.negative_dataset_total_size,
                                      training_positive_size=tag_loader.training_positive_size,
                                      validation_positive_size=tag_loader.validation_positive_size,
                                      training_negative_size=tag_loader.training_negative_size,
                                      validation_negative_size=tag_loader.validation_negative_size)

    report_name = "{}.txt".format(filename)
    report_output_path = os.path.join(output_path, report_name)
    report_buffer = BytesIO(report_str.encode(encoding='UTF-8'))
    cmd.upload_data(tag_loader.minio_client, bucket_name, report_output_path, report_buffer)

    # save graph report
    # upload the graph report
    graph_buffer = get_graph_report(model_class=classifier_model,
                                    train_predictions=sigmoid_train_pred.cpu().numpy(),
                                    training_targets=training_targets.cpu().numpy(),
                                    validation_predictions=sigmoid_pred.cpu().numpy(),
                                    validation_targets=validation_targets.cpu().numpy(),
                                    hidden_layer_neuron_count=hidden_layer_neuron_count,
                                    date=date_now,
                                    dataset_name=dataset_name,
                                    network_type=network_type,
                                    input_type=input_type,
                                    input_shape=input_shape,
                                    output_type=output_type,
                                    tag_string=tag_name,
                                    positive_dataset_total_size=tag_loader.positive_dataset_total_size,
                                    negative_dataset_total_size=tag_loader.negative_dataset_total_size,
                                    training_positive_size=tag_loader.training_positive_size,
                                    validation_positive_size=tag_loader.validation_positive_size,
                                    training_negative_size=tag_loader.training_negative_size,
                                    validation_negative_size=tag_loader.validation_negative_size,
                                    pooling_strategy=pooling_strategy,
                                    training_accuracy=training_accuracy,
                                    validation_accuracy=validation_accuracy,
                                    mse_training_loss=training_loss,
                                    mse_validation_loss=validation_loss,
                                    min_prediction_vector_before_sigmoid=min(validation_pred).item(),
                                    max_prediction_vector_before_sigmoid=max(validation_pred).item(),
                                    bce_loss_training=bce_loss_training,
                                    bce_loss_validation=bce_loss_validation,
                                    )
    graph_name = "{}.png".format(filename)
    graph_output_path = os.path.join(output_path, graph_name)
    cmd.upload_data(tag_loader.minio_client, bucket_name, graph_output_path, graph_buffer)