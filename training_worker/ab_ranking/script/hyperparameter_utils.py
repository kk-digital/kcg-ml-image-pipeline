import sys
import json
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.minio import cmd
from training_worker.ab_ranking.model.ab_ranking_data_loader import get_object, ABData
from training_worker.ab_ranking.model import constants


def get_ab_data(minio_client, path, index):
    # load json object from minio
    data = get_object(minio_client, path)
    decoded_data = data.decode().replace("'", '"')
    item = json.loads(decoded_data)

    flagged = False
    if "flagged" in item:
        flagged = item["flagged"]

    ab_data = ABData(task=item["task"],
                     username=item["username"],
                     hash_image_1=item["image_1_metadata"]["file_hash"],
                     hash_image_2=item["image_2_metadata"]["file_hash"],
                     selected_image_index=item["selected_image_index"],
                     selected_image_hash=item["selected_image_hash"],
                     image_archive="",
                     image_1_path=item["image_1_metadata"]["file_path"],
                     image_2_path=item["image_2_metadata"]["file_path"],
                     datetime=item["datetime"],
                     flagged=flagged)

    return ab_data, flagged, path, index

def get_aggregated_selection_datapoints(minio_client, dataset_name):
    prefix = os.path.join(dataset_name, "data/ranking/aggregate")
    dataset_paths = cmd.get_list_of_objects_with_prefix(minio_client, "datasets", prefix=prefix)

    print("Get selection datapoints contents and filter out flagged datapoints...")
    dataset_path_list = [None] * len(dataset_paths)
    ab_data_list = [None] * len(dataset_paths)
    flagged_count = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        count = 0
        for path in dataset_paths:
            futures.append(executor.submit(get_ab_data, minio_client=minio_client, path=path, index=count))
            count += 1

        for future in tqdm(as_completed(futures), total=len(dataset_paths)):
            ab_data, flagged, path, index = future.result()
            if not flagged:
                ab_data_list[index] = ab_data
                dataset_path_list[index] = path
            else:
                flagged_count += 1

    unflagged_ab_data = []
    for data in tqdm(ab_data_list):
        if data is not None:
            unflagged_ab_data.append(data)

    unflagged_dataset_path_list = []
    for data in tqdm(dataset_path_list):
        if data is not None:
            unflagged_dataset_path_list.append(data)

    print("Total flagged selection datapoints = {}".format(flagged_count))
    return unflagged_ab_data, unflagged_dataset_path_list

def get_features_data(minio_client, features_path):
    features_img_data = get_object(minio_client, features_path)

    return features_img_data, features_path

def get_data_dicts(minio_client, dataset_name, input_type):
    selection_datapoints_dict = {}
    features_dict = {}

    # if exist then get paths for aggregated selection datapoints
    ab_data_list, dataset_paths = get_aggregated_selection_datapoints(minio_client, dataset_name)
    print("# of dataset paths retrieved=", len(dataset_paths))
    if len(dataset_paths) == 0:
        raise Exception("No selection datapoints json found.")

    # load json object from minio
    print("Loading {} data from minio...".format(input_type))
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(len(ab_data_list)):
            ab_data = ab_data_list[i]
            # add to dict
            selection_datapoints_dict[dataset_paths[i]] = ab_data

            input_type_extension = "_embedding.msgpack"
            if input_type == constants.CLIP:
                input_type_extension = "_clip.msgpack"

            # embeddings are in file_path_embedding.msgpack
            file_path_img_1 = ab_data.image_1_path
            features_path_img_1 = file_path_img_1.replace(".jpg", input_type_extension)
            features_path_img_1 = features_path_img_1.replace("datasets/", "")

            file_path_img_2 = ab_data.image_2_path
            features_path_img_2 = file_path_img_2.replace(".jpg", input_type_extension)
            features_path_img_2 = features_path_img_2.replace("datasets/", "")

            futures.append(executor.submit(get_features_data, minio_client=minio_client, features_path=features_path_img_1))
            futures.append(executor.submit(get_features_data, minio_client=minio_client, features_path=features_path_img_2))

        for future in tqdm(as_completed(futures), total=len(futures)):
            features_data, features_path = future.result()
            # add to dict
            features_dict[features_path] = features_data


    return dataset_paths, selection_datapoints_dict, features_dict


def get_performance_graph_report(results):
    # Initialize all graphs/subplots
    plt.figure(figsize=(22, 20))
    figure_shape = (2, 2)
    performance = plt.subplot2grid(figure_shape, (0, 0), rowspan=2, colspan=2)
    # ----------------------------------------------------------------------------------------------------------------#
    # performance
    loss_values = []
    for res in results:
        loss_values.append(res.metrics["validation-loss"])

    x_axis_values = [i for i in range(len(loss_values))]
    performance.plot(x_axis_values,
                     loss_values,
                     label="elm-v1",
                     c="#281ad9")

    performance.set_xlabel("Iterations")
    performance.set_ylabel("Loss")
    performance.set_title("Loss vs Iterations")
    performance.legend()

    performance.autoscale(enable=True, axis='both')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return buf
