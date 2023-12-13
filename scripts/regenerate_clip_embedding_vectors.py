import os
import sys
import argparse
from tqdm import tqdm
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from utility.http import generation_request
from utility.minio import cmd
from stable_diffusion.model.clip_text_embedder import CLIPTextEmbedder
from data_loader.prompt_embedding import PromptEmbedding
from data_loader.utils import DATASETS_BUCKET, get_object
from utility.clip.clip_text_embedder import tensor_attention_pooling, tensor_max_pooling, tensor_max_abs_pooling

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
            embedding_paths.append(path)

    return embedding_paths


def save_prompt_embedding_to_minio(minio_client, prompt_embedding, file_path):
    msgpack_string = prompt_embedding.get_msgpack_string()

    buffer = io.BytesIO()
    buffer.write(msgpack_string)
    buffer.seek(0)

    cmd.upload_data(minio_client, "datasets", file_path, buffer)


def regenerate_embeddings(minio_client, dataset_name, file_path, text_embedder):
    # prepare filenames
    # filename
    filename = os.path.split(file_path)[-1]
    filename = filename.replace("_embedding.msgpack", "")
    # get parent dir
    parent_dir = os.path.dirname(file_path)
    parent_dir = os.path.split(parent_dir)[-1]

    text_embedding_path = os.path.join(dataset_name, "embeddings/text-embedding", parent_dir, filename + "-text-embedding.msgpack")
    text_embedding_average_pooled_path = os.path.join(dataset_name, "embeddings/text-embedding", parent_dir, filename + "-text-embedding-average-pooled.msgpack")
    text_embedding_max_pooled_path = os.path.join(dataset_name, "embeddings/text-embedding", parent_dir, filename + "-text-embedding-max-pooled.msgpack")
    text_embedding_signed_max_pooled_path = os.path.join(dataset_name, "embeddings/text-embedding", parent_dir, filename + "-text-embedding-signed-max-pooled.msgpack")

    # return if exists
    if cmd.is_object_exists(minio_client, "datasets", text_embedding_path):
        return

    # read the embedding
    embedding_data = get_object(minio_client, file_path)
    prompt_embedding = PromptEmbedding.from_msgpack_bytes(embedding_data)
    positive_prompt = prompt_embedding.positive_prompt
    negative_prompt = prompt_embedding.negative_prompt

    # calculate new embeddings
    positive_embedding, _, positive_attention_mask = text_embedder.forward_return_all(positive_prompt)
    negative_embedding, _, negative_attention_mask = text_embedder.forward_return_all(negative_prompt)

    prompt_embedding.positive_attention_mask = positive_attention_mask.detach().cpu().numpy()
    prompt_embedding.negative_attention_mask = negative_attention_mask.detach().cpu().numpy()

    # normal text embedding 77*768
    prompt_embedding.positive_embedding = positive_embedding.detach().cpu().numpy()
    prompt_embedding.negative_embedding = negative_embedding.detach().cpu().numpy()
    save_prompt_embedding_to_minio(minio_client, prompt_embedding, text_embedding_path)

    # average
    positive_average_pooled = tensor_attention_pooling(positive_embedding, positive_attention_mask)
    negative_average_pooled = tensor_attention_pooling(negative_embedding, negative_attention_mask)
    prompt_embedding.positive_embedding = positive_average_pooled.detach().cpu().numpy()
    prompt_embedding.negative_embedding = negative_average_pooled.detach().cpu().numpy()
    save_prompt_embedding_to_minio(minio_client, prompt_embedding, text_embedding_average_pooled_path)


    # max
    positive_max_pooled = tensor_max_pooling(positive_embedding)
    negative_max_pooled = tensor_max_pooling(negative_embedding)
    prompt_embedding.positive_embedding = positive_max_pooled.detach().cpu().numpy()
    prompt_embedding.negative_embedding = negative_max_pooled.detach().cpu().numpy()
    save_prompt_embedding_to_minio(minio_client, prompt_embedding, text_embedding_max_pooled_path)

    # signed max
    positive_signed_max_pooled = tensor_max_abs_pooling(positive_embedding)
    negative_signed_max_pooled = tensor_max_abs_pooling(negative_embedding)
    prompt_embedding.positive_embedding = positive_signed_max_pooled.detach().cpu().numpy()
    prompt_embedding.negative_embedding = negative_signed_max_pooled.detach().cpu().numpy()
    save_prompt_embedding_to_minio(minio_client, prompt_embedding, text_embedding_signed_max_pooled_path)


def process_embeddings(minio_client,
                       dataset_name,
                       text_embedder):
    embeddings_file_paths = get_embeddings_file_paths(minio_client, dataset_name)

    print("Regenerating embeddings...")
    # use multiprocessing
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for path in embeddings_file_paths:
            futures.append(executor.submit(regenerate_embeddings,
                                           minio_client=minio_client,
                                           dataset_name=dataset_name,
                                           file_path=path,
                                           text_embedder=text_embedder))

        for _ in tqdm(as_completed(futures), total=len(futures)):
            continue


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

    text_embedder = CLIPTextEmbedder()
    text_embedder.load_submodels()

    if dataset_name != "all":
        process_embeddings(minio_client=minio_client,
                           dataset_name=dataset_name,
                           text_embedder=text_embedder)
    else:
        # if all, run script for all existing datasets
        # get dataset name list
        dataset_names = generation_request.http_get_dataset_names()
        print("dataset names=", dataset_names)
        for dataset in dataset_names:
            try:
                print("Running script for {}...".format(dataset))
                process_embeddings(minio_client=minio_client,
                                   dataset_name=dataset,
                                   text_embedder=text_embedder)
            except Exception as e:
                print("Error running script for {}: {}".format(dataset, e))

