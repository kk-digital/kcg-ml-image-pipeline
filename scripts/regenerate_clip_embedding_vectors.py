import os
import sys
import argparse
from tqdm import tqdm

base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from worker.http import request
from utility.minio import cmd
from stable_diffusion.model.clip_text_embedder import CLIPTextEmbedder


def get_embeddings_file_paths(minio_client, dataset_name):
    # get embeddings list
    prefix = dataset_name
    dataset_paths = cmd.get_list_of_objects_with_prefix(minio_client, "datasets", prefix=prefix)

    # filter out non embedding files
    embedding_file_extension = "_embedding.msgpack"
    embedding_paths = []
    print("Getting only embedding file paths...")
    for path in tqdm(dataset_paths):
        if path.endswith(embedding_file_extension):
            print(path)
            embedding_paths.append(path)

    return embedding_paths


def process_embeddings(minio_client,
                       dataset_name,
                       text_embedder):
    embeddings_file_paths = get_embeddings_file_paths(minio_client, dataset_name)
    # embedding, pooled, attention_mask = text_embedder.forward_return_all(prompt)
    # attention_pooled = tensor_attention_pooling(embedding, attention_mask)



def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Regenerate embeddings: Saves new embeddings to minio")

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--dataset-name', type=str,
                        help="The dataset name to regenerate embeddings, use 'all' to regenerate embeddings for all datasets",
                        default='environmental')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    dataset_name = args.dataset_name
    # get minio client
    minio_client = cmd.get_minio_client(minio_ip_addr=None,  # will use default if none is given
                                        minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key)

    # get text embedder
    text_embedder = CLIPTextEmbedder()
    text_embedder.load_submodels()

    if dataset_name != "all":
        process_embeddings(minio_client=minio_client,
                           dataset_name=dataset_name,
                           text_embedder=text_embedder)
    else:
        # if all, run script for all existing datasets
        # get dataset name list
        dataset_names = request.http_get_dataset_names()
        print("dataset names=", dataset_names)
        for dataset in dataset_names:
            try:
                print("Running script for {}...".format(dataset))
                process_embeddings(minio_client=minio_client,
                                   dataset_name=dataset_name)
            except Exception as e:
                print("Error running script for {}: {}".format(dataset, e))

