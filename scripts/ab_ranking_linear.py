import os
import sys
import argparse

base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from training_worker.ab_ranking.script.ab_ranking_linear import train_ranking
from worker.http import request

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train ab ranking linear model")

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--dataset-name', type=str,
                        help="The dataset name to use for training, use 'all' to train models for all datasets",
                        default='environmental')
    parser.add_argument('--input-type', type=str, default="embedding")
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--train-percent', type=float, default=0.9)
    parser.add_argument('--training-batch-size', type=int, default=1)
    parser.add_argument('--weight-decay', type=float, default=0.00)
    parser.add_argument('--load-data-to-ram', type=bool, default=True)
    parser.add_argument('--debug-asserts', type=bool, default=False)
    parser.add_argument('--normalize-vectors', type=bool, default=True)
    parser.add_argument('--pooling-strategy', type=int, default=0)
    parser.add_argument('--add-loss-penalty', type=bool, default=True)
    parser.add_argument('--target-option', type=int, default=0)
    parser.add_argument('--duplicate-flip-option', type=int, default=0)
    parser.add_argument('--randomize-data-per-epoch', type=bool, default=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    dataset_name = args.dataset_name

    if dataset_name != "all":
        train_ranking(minio_ip_addr=None,  # will use default if none is given
                      minio_access_key=args.minio_access_key,
                      minio_secret_key=args.minio_secret_key,
                      dataset_name=dataset_name,
                      input_type=args.input_type,
                      epochs=args.epochs,
                      learning_rate=args.learning_rate,
                      buffer_size=args.buffer_size,
                      train_percent=args.train_percent,
                      training_batch_size=args.training_batch_size,
                      weight_decay=args.weight_decay,
                      load_data_to_ram=args.load_data_to_ram,
                      debug_asserts=args.debug_asserts,
                      normalize_vectors=args.normalize_vectors,
                      pooling_strategy=args.pooling_strategy,
                      add_loss_penalty=args.add_loss_penalty,
                      target_option=args.target_option,
                      duplicate_flip_option=args.duplicate_flip_option,
                      randomize_data_per_epoch=args.randomize_data_per_epoch)
    else:
        # if all, train models for all existing datasets
        # get dataset name list
        dataset_names = request.http_get_dataset_names()
        print("dataset names=", dataset_names)
        for dataset in dataset_names:
            try:
                print("Training model for {}...".format(dataset))
                train_ranking(minio_ip_addr=None,  # will use default if none is given
                              minio_access_key=args.minio_access_key,
                              minio_secret_key=args.minio_secret_key,
                              dataset_name=dataset,
                              input_type=args.input_type,
                              epochs=args.epochs,
                              learning_rate=args.learning_rate,
                              buffer_size=args.buffer_size,
                              train_percent=args.train_percent,
                              training_batch_size=args.training_batch_size,
                              weight_decay=args.weight_decay,
                              load_data_to_ram=args.load_data_to_ram,
                              debug_asserts=args.debug_asserts,
                              normalize_vectors=args.normalize_vectors,
                              pooling_strategy=args.pooling_strategy,
                              add_loss_penalty=args.add_loss_penalty,
                              target_option=args.target_option,
                              duplicate_flip_option=args.duplicate_flip_option,
                              randomize_data_per_epoch=args.randomize_data_per_epoch)
            except Exception as e:
                print("Error training model for {}: {}".format(dataset, e))

