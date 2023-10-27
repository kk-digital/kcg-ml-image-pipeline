import os
import torch
import sys
from datetime import datetime
from pytz import timezone
import argparse
base_directory = os.getcwd()
sys.path.insert(0, base_directory)
#sys.path.insert(0, '/content/drive/MyDrive/github/new/kcg-ml-image-pipeline')
from utility.regression_utils import torchinfo_summary
from training_worker.ab_ranking.model.ab_ranking_efficient_net import ABRankingEfficientNetModel
from training_worker.ab_ranking.model.reports.ab_ranking_linear_train_report import get_train_report
from training_worker.ab_ranking.model.reports.graph_report_ab_ranking_linear import *
from training_worker.ab_ranking.model.ab_ranking_data_loader import ABRankingDatasetLoader
from utility.minio import cmd
from training_worker.ab_ranking.model.reports.get_model_card import get_model_card_buf
from training_worker.ab_ranking.model import constants


def train_ranking(dataset_name: str,
                  minio_ip_addr=None,
                  minio_access_key=None,
                  minio_secret_key=None,
                  epochs=10000,
                  learning_rate=0.05,
                  buffer_size=20000,
                  train_percent=0.9,
                  training_batch_size=1,
                  weight_decay=0.01,
                  load_data_to_ram=False,
                  debug_asserts=False,
                  normalize_vectors=False,
                  pooling_strategy=constants.AVERAGE_POOLING):
    date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%Y-%m-%d')
    print("Current datetime: {}".format(datetime.now(tz=timezone("Asia/Hong_Kong"))))
    bucket_name = "datasets"
    training_dataset_path = os.path.join(bucket_name, dataset_name)
    efficient_net_version = "b0"
    network_type = "efficient-net-{}".format(efficient_net_version)
    input_type = "embedding"
    in_channels = 1
    inputs_shape = (in_channels, 1, 768*2)
    output_type = "score"
    output_path = "{}/models/ranking/ab_ranking_efficient_net".format(dataset_name)



    # load dataset
    dataset_loader = ABRankingDatasetLoader(dataset_name=dataset_name,
                                            minio_ip_addr=minio_ip_addr,
                                            minio_access_key=minio_access_key,
                                            minio_secret_key=minio_secret_key,
                                            buffer_size=buffer_size,
                                            train_percent=train_percent,
                                            load_to_ram=load_data_to_ram,
                                            pooling_strategy=pooling_strategy,
                                            normalize_vectors=normalize_vectors)
    dataset_loader.load_dataset()

    training_total_size = dataset_loader.get_len_training_ab_data()
    validation_total_size = dataset_loader.get_len_validation_ab_data()

    ab_model = ABRankingEfficientNetModel(efficient_net_version=efficient_net_version,
                                          in_channels=in_channels,
                                          num_classes=1,
                                          inputs_shape=inputs_shape)
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
                                                   debug_asserts=debug_asserts)

    # Upload model to minio
    model_name = "{}.pth".format(date_now)
    model_output_path = os.path.join(output_path, model_name)
    ab_model.save(dataset_loader.minio_client, bucket_name, model_output_path)

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
    report_name = "{}.txt".format(date_now)
    report_output_path = os.path.join(output_path,  report_name)

    report_buffer = BytesIO(report_str.encode(encoding='UTF-8'))

    # upload the txt report
    cmd.upload_data(dataset_loader.minio_client, bucket_name, report_output_path, report_buffer)

    # show and save graph
    graph_name = "{}.png".format(date_now)
    graph_output_path = os.path.join(output_path, graph_name)

    graph_buffer = get_graph_report(ab_model,
                                    training_predicted_probabilities,
                                    training_target_probabilities,
                                    validation_predicted_probabilities,
                                    validation_target_probabilities,
                                    training_predicted_score_images_x,
                                    training_predicted_score_images_y,
                                    validation_predicted_score_images_x,
                                    validation_predicted_score_images_y,
                                    training_total_size,
                                    validation_total_size,
                                    training_loss_per_epoch,
                                    validation_loss_per_epoch,
                                    epochs,
                                    learning_rate,
                                    training_batch_size,
                                    weight_decay,
                                    date_now,
                                    network_type,
                                    input_type,
                                    inputs_shape,
                                    output_type,
                                    train_sum_correct,
                                    validation_sum_correct,
                                    ab_model.loss_func_name,
                                    dataset_name,
                                    pooling_strategy,
                                    normalize_vectors)
    # upload the graph report
    cmd.upload_data(dataset_loader.minio_client, bucket_name,graph_output_path, graph_buffer)

    # get model card and upload
    model_card_name = "{}.json".format(date_now)
    model_card_name_output_path = os.path.join(output_path, model_card_name)
    model_card_buf = get_model_card_buf(ab_model, training_total_size, validation_total_size, graph_output_path)
    cmd.upload_data(dataset_loader.minio_client, bucket_name, model_card_name_output_path, model_card_buf)

    return model_output_path, report_output_path, graph_output_path


def run_ab_ranking_efficient_net_task(training_task, minio_access_key, minio_secret_key):
    model_output_path, \
        report_output_path, \
        graph_output_path = train_ranking(dataset_name=training_task["dataset_name"],
                                          minio_access_key=minio_access_key,
                                          minio_secret_key=minio_secret_key,
                                          epochs=training_task["epochs"],
                                          learning_rate=training_task["learning_rate"],
                                          buffer_size=training_task["buffer_size"],
                                          train_percent=training_task["train_percent"])

    return model_output_path, report_output_path, graph_output_path


def test_run(minio_ip_addr,minio_access_key,minio_secret_key,batch_size,epochs,lr):
    train_ranking(minio_ip_addr=minio_ip_addr,  # will use defualt if none is given
                  minio_access_key=minio_access_key,
                  minio_secret_key=minio_secret_key,
                  dataset_name="environmental",
                  epochs=epochs,
                  learning_rate=lr,
                  buffer_size=20000,
                  train_percent=0.9,
                  training_batch_size=batch_size,
                  weight_decay=0.01,
                  load_data_to_ram=True,
                  debug_asserts=True,
                  normalize_vectors=True,
                  pooling_strategy=constants.MAX_POOLING)


if __name__ == '__main__':
    parser = argparse.ArgumentParser() # get a parser object
    parser.add_argument('--minio-addr', metavar='minio-addr', default=None, required=False,
                      help='minio server ip address')
    parser.add_argument('--minio-access-key', metavar='minio-access-key', required=True,
                      help='access key for the minio account')
    parser.add_argument('--minio-secret-key', metavar='minio-secret-key', required=True,
                      help='secret key for the minio account')  
    parser.add_argument('--batch-size', metavar='batch-size', required=True,
                      help='batch size for training')  
    parser.add_argument('--epochs', metavar='epochs', required=True,
                      help='number of epochs for training') 
    parser.add_argument('--lr', metavar='lr', required=True,
                      help='learning rate for training')                                             
                                                             
    args = parser.parse_args()
    test_run(args.minio_addr,args.minio_access_key,args.minio_secret_key,int(args.batch_size),int(args.epochs),float(args.lr))
