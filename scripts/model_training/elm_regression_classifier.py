import os
import sys
import argparse

base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from training_worker.classifiers.scripts.elm_regression import train_classifier
from utility.http.request import http_get_tag_list

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train elm regression classifier model")

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--input-type', type=str, default="embedding")
    parser.add_argument('--tag-name', type=str)
    parser.add_argument('--hidden-layer-neuron-count', type=int, default=3000)
    parser.add_argument('--pooling-strategy', type=int, default=0)
    parser.add_argument('--train-percent', type=float, default=0.9)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    if args.tag_name != "all":
        try:
            print("Training model for tag: {}".format(args.tag_name))
            train_classifier(minio_ip_addr=None,
                             minio_access_key=args.minio_access_key,
                             minio_secret_key=args.minio_secret_key,
                             input_type=args.input_type,
                             tag_name=args.tag_name,
                             hidden_layer_neuron_count=args.hidden_layer_neuron_count,
                             pooling_strategy=args.pooling_strategy,
                             train_percent=args.train_percent)
        except Exception as e:
            print("Error training model for tag {}: {}".format(tag_name, e))
    else:
        # train for all
        tags = http_get_tag_list()
        tag_string_list = []
        for tag in tags:
            tag_string_list.append(tag["tag_string"])

        print("tags found = ", tag_string_list)
        for tag_name in tag_string_list:
            try:
                print("Training model for tag: {}".format(tag_name))
                train_classifier(minio_ip_addr=None,
                                 minio_access_key=args.minio_access_key,
                                 minio_secret_key=args.minio_secret_key,
                                 input_type=args.input_type,
                                 tag_name=tag_name,
                                 hidden_layer_neuron_count=args.hidden_layer_neuron_count,
                                 pooling_strategy=args.pooling_strategy,
                                 train_percent=args.train_percent)
            except Exception as e:
                print("Error training model for tag {}: {}".format(tag_name, e))
            print("==============================================================================")
