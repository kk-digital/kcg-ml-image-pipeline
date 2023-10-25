import sys
import json
from tqdm import tqdm

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.minio import cmd
from training_worker.ab_ranking.model.ab_ranking_data_loader import get_aggregated_selection_datapoints, get_object, ABData


def get_data_dicts(minio_access_key, minio_secret_key, dataset_name):
    selection_datapoints_dict = {}
    embeddings_dict = {}

    # get minio client
    minio_client = cmd.get_minio_client(minio_access_key=minio_access_key,
                                        minio_secret_key=minio_secret_key)

    # if exist then get paths for aggregated selection datapoints
    dataset_paths = get_aggregated_selection_datapoints(minio_client, dataset_name)
    print("# of dataset paths retrieved=", len(dataset_paths))
    if len(dataset_paths) == 0:
        raise Exception("No selection datapoints json found.")

    # load json object from minio
    print("Loading objects from minio...")
    for dataset_path in tqdm(dataset_paths):
        data = get_object(minio_client, dataset_path)
        decoded_data = data.decode().replace("'", '"')
        item = json.loads(decoded_data)

        ab_data = ABData(task=item["task"],
                         username=item["username"],
                         hash_image_1=item["image_1_metadata"]["file_hash"],
                         hash_image_2=item["image_2_metadata"]["file_hash"],
                         selected_image_index=item["selected_image_index"],
                         selected_image_hash=item["selected_image_hash"],
                         image_archive="",
                         image_1_path=item["image_1_metadata"]["file_path"],
                         image_2_path=item["image_2_metadata"]["file_path"],
                         datetime=item["datetime"])
        # add to dict
        selection_datapoints_dict[dataset_path] = ab_data

        file_path_img_1 = ab_data.image_1_path
        file_path_img_2 = ab_data.image_2_path

        # embeddings are in file_path_embedding.msgpack
        embeddings_path_img_1 = file_path_img_1.replace(".jpg", "_embedding.msgpack")
        embeddings_path_img_1 = embeddings_path_img_1.replace("datasets/", "")

        embeddings_path_img_2 = file_path_img_2.replace(".jpg", "_embedding.msgpack")
        embeddings_path_img_2 = embeddings_path_img_2.replace("datasets/", "")

        embeddings_img_1_data = get_object(minio_client, embeddings_path_img_1)
        # add to dict
        embeddings_dict[embeddings_path_img_1] = embeddings_img_1_data

        embeddings_img_2_data = get_object(minio_client, embeddings_path_img_2)
        # add to dict
        embeddings_dict[embeddings_path_img_2] = embeddings_img_2_data

    return dataset_paths, selection_datapoints_dict, embeddings_dict
