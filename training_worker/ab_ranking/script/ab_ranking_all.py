import argparse

import os
base_dir = './'
os.path.insert(0, base_dir)

from training_worker.ab_ranking.script.ab_ranking_elm_v2 import train_ranking as train_ranking_elm
from training_worker.ab_ranking.script.ab_ranking_linear_v2 import train_ranking as train_ranking_linear

# import http request service for getting rank model list
from utility.http.request import http_get_rank_model_list

# import constants for training ranking model
from training_worker.ab_ranking.model import constants


def parse_args():
    parser = argparse.ArgumentParser()
    # TODO: add argument for training ranking model
    parser.add_argument('--model-type', type=str, default='linear', help='Rank model type - linear or elm')

    # Add arguments for MinIO connection
    parser.add_argument('--minio-ip-addr', type=str, default=None, help='MinIO IP address')
    parser.add_argument('--minio-access-key', type=str, default=None, help='MinIO access key')
    parser.add_argument('--minio-secret-key', type=str, default=None, help='MinIO secret key')

     # Add arguments for training parameters
    parser.add_argument('--input-type', type=str, default='embedding', help='Input type')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--train-percent', type=float, default=0.9, help='Percentage of data used for training')
    parser.add_argument('--training-batch-size', type=int, default=1, help='Training batch size')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--load-data-to-ram', action='store_true', help='Load data to RAM')
    parser.add_argument('--debug-asserts', action='store_true', help='Enable debug asserts')
    parser.add_argument('--normalize-vectors', action='store_true', help='Normalize input vectors')
    parser.add_argument('--pooling-strategy', type=str, default=constants.AVERAGE_POOLING, help='Pooling strategy')
    parser.add_argument('--add-loss-penalty', action='store_true', help='Add loss penalty')
    parser.add_argument('--target-option', type=str, default=constants.TARGET_1_AND_0, help='Target option')
    parser.add_argument('--duplicate-flip-option', type=str, default=constants.DUPLICATE_AND_FLIP_ALL, help='Duplicate and flip option')
    parser.add_argument('--randomize-data-per-epoch', action='store_true', help='Randomize data per epoch')
    parser.add_argument('--penalty-range', type=float, default=5.0, help='Penalty range')

    # more paramter for elm ranking model
    parser.add_argument('--num-random-layers', type=int, default=1, help='Number of random layers')
    parser.add_argument('--elm-sparsity', type=float, default=0.5, help='ELM sparsity')

    

    return parser.parse_args()

def main():
    args = parse_args()

    # Get all rank model infor
    rank_model_list = http_get_rank_model_list()

    for rank_model in rank_model_list:
        print("{} Ranking....".format(rank_model["rank_model_string"]))
        if args.model_type == 'linear':
            train_ranking_linear(
                rank_model_info=rank_model,
                minio_ip_addr=args.minio_ip_addr,
                minio_access_key=args.minio_access_key,
                minio_secret_key=args.minio_secret_key,
                input_type=args.input_type,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
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
                randomize_data_per_epoch=args.randomize_data_per_epoch,
                penalty_range=args.penalty_range,
            )
        elif args.model_type == 'elm':
            train_ranking_elm(
                    rank_model_info=rank_model,
                    minio_ip_addr=args.minio_ip_addr,
                    minio_access_key=args.minio_access_key,
                    minio_secret_key=args.minio_secret_key,
                    input_type=args.input_type,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    train_percent=args.train_percent,
                    training_batch_size=args.training_batch_size,
                    weight_decay=args.weight_decay,
                    load_data_to_ram=args.load_data_to_ram,
                    debug_asserts=args.debug_asserts,
                    normalize_vectors=args.normalize_vectors,
                    pooling_strategy=args.pooling_strategy,
                    num_random_layers=args.num_random_layers,
                    add_loss_penalty=args.add_loss_penalty,
                    target_option=args.target_option,
                    duplicate_flip_option=args.duplicate_flip_option,
                    randomize_data_per_epoch=args.randomize_data_per_epoch,
                    elm_sparsity=args.elm_sparsity,
                    penalty_range=args.penalty_range,
            )
    # TODO: Add logic for training ranking model
    
if __name__ == '__main__':

    main()