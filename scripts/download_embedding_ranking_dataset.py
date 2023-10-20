import os
import sys
import json
import argparse
import time
from tqdm import tqdm
base_directory = "./"
sys.path.insert(0, base_directory)

from utility.minio import cmd


DATASETS_BUCKET = "datasets"

def get_datasets(minio_client):
    datasets = cmd.get_list_of_objects(minio_client, DATASETS_BUCKET)
    return datasets


def get_aggregated_selection_datapoints(minio_client, dataset_name):
    prefix = os.path.join(dataset_name, "data/ranking/aggregate")
    datasets = cmd.get_list_of_objects_with_prefix(minio_client, DATASETS_BUCKET, prefix=prefix)

    return datasets


def get_object(client, file_path):
    response = client.get_object(DATASETS_BUCKET, file_path)
    data = response.data

    return data


class ABRankingDatasetDownloader:
    def __init__(self,
                 dataset_name,
                 minio_addr=None,
                 minio_access_key=None,
                 minio_secret_key=None,):
        self.dataset_name = dataset_name

        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.minio_client = cmd.get_minio_client(minio_access_key=self.minio_access_key,
                                                 minio_secret_key=self.minio_secret_key,
                                                 minio_addr=minio_addr)

    def get_dataset_paths(self):
        print("Loading dataset references...")

        dataset_list = get_datasets(self.minio_client)
        if self.dataset_name not in dataset_list:
            raise Exception("Dataset is not in minio server")

        # if exist then get paths for aggregated selection datapoints
        dataset_paths = get_aggregated_selection_datapoints(self.minio_client, self.dataset_name)
        print("# of dataset paths retrieved=", len(dataset_paths))
        if len(dataset_paths) == 0:
            raise Exception("No selection datapoints json found.")

        return dataset_paths

    def get_ranking_folder_name(self, output_dir):
        count = 1
        while True:
            ranking_dir = "ranking_v{}".format(count)
            output_parent_ranking_path = os.path.join(output_dir, ranking_dir)

            if os.path.exists(output_parent_ranking_path):
                # then continue, we don't want to overwrite existing
                count += 1
                continue

            # then use the current parent
            return output_parent_ranking_path

    def download_embedding_ranking_datapoint(self, dataset_path, output_path):
        # embeddings output
        embeddings_output_path = os.path.join(output_path, "embeddings")
        if not os.path.exists(embeddings_output_path):
            os.makedirs(embeddings_output_path)

        # ranking data output
        ranking_data_output_path = os.path.join(output_path, "ranking_data")
        if not os.path.exists(ranking_data_output_path):
            os.makedirs(ranking_data_output_path)

        # load json object from minio
        data = get_object(self.minio_client, dataset_path)
        decoded_data = data.decode().replace("'", '"')
        item = json.loads(decoded_data)

        # save ab ranking data json
        # Writing to sample.json
        ranking_data_output_path = os.path.join(ranking_data_output_path, os.path.basename(dataset_path))
        with open(ranking_data_output_path, "w") as outfile:
            outfile.write(decoded_data)

        file_path_img_1 = item["image_1_metadata"]["file_path"]
        file_path_img_2 = item["image_2_metadata"]["file_path"]

        # embeddings are in file_path_embedding.msgpack
        embeddings_path_img_1 = file_path_img_1.replace(".jpg", "_embedding.msgpack")
        embeddings_path_img_1 = embeddings_path_img_1.replace("datasets/", "")

        embeddings_path_img_2 = file_path_img_2.replace(".jpg", "_embedding.msgpack")
        embeddings_path_img_2 = embeddings_path_img_2.replace("datasets/", "")

        # download embeddings
        embedding_1_output_path = os.path.join(embeddings_output_path, os.path.basename(embeddings_path_img_1))
        cmd.download_from_minio(self.minio_client, DATASETS_BUCKET, embeddings_path_img_1, embedding_1_output_path)

        embedding_2_output_path = os.path.join(embeddings_output_path, os.path.basename(embeddings_path_img_2))
        cmd.download_from_minio(self.minio_client, DATASETS_BUCKET, embeddings_path_img_2, embedding_2_output_path)

    def download_all_data(self, output_dir):
        print("Downloading all embedding and ranking data...")
        start_time = time.time()

        # prepare output dir
        output_dir = os.path.join(output_dir, self.dataset_name)
        ranking_output_path = self.get_ranking_folder_name(output_dir)

        paths_list = self.get_dataset_paths()
        for path in tqdm(paths_list):
            self.download_embedding_ranking_datapoint(path, ranking_output_path)

        time_elapsed=time.time() - start_time
        print("Time elapsed: {0}s".format(format(time_elapsed, ".2f")))


def parse_args():
    parser = argparse.ArgumentParser(description="Worker for training models")

    # Required parameters
    parser.add_argument("--dataset", type=str,
                        help="The dataset to download")
    parser.add_argument("--output", type=str, default="output",
                        help="The path to save the dataset")
    parser.add_argument("--minio-addr", type=str, default=None,
                        help="The minio server ip address")
    parser.add_argument("--minio-access-key", type=str,
                        help="The minio access key to use so worker can upload files to minio server")
    parser.add_argument("--minio-secret-key", type=str,
                        help="The minio secret key to use so worker can upload files to minio server")

    return parser.parse_args()

def main():
    args = parse_args()

    ab_downloader = ABRankingDatasetDownloader(dataset_name=args.dataset,
                                               minio_addr=args.minio_addr,
                                               minio_access_key=args.minio_access_key,
                                               minio_secret_key=args.minio_secret_key)
    ab_downloader.download_all_data(args.output)


if __name__ == '__main__':
    main()