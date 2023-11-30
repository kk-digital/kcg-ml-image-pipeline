import os
import sys
import json
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
base_directory = "./"
sys.path.insert(0, base_directory)

from utility.minio import cmd
from data_loader.ab_data import ABData
from utility.path import separate_bucket_and_file_path

DATASETS_BUCKET = "datasets"


def get_datasets(minio_client):
    datasets = cmd.get_list_of_objects(minio_client, "datasets")
    return datasets


def get_ab_data(minio_client, path, index):
    # load json object from minio
    data = get_object(minio_client, path)
    decoded_data = data.decode().replace("'", '"')
    item = json.loads(decoded_data)

    flagged = False
    if "flagged" in item:
        flagged = item["flagged"]

    ab_data = ABData.deserialize(item)

    return ab_data, flagged, index


def get_aggregated_selection_datapoints(minio_client, dataset_name):
    prefix = os.path.join(dataset_name, "data/ranking/aggregate")
    dataset_paths = cmd.get_list_of_objects_with_prefix(minio_client, "datasets", prefix=prefix)

    print("Get selection datapoints contents and filter out flagged datapoints...")
    ab_data_list = [None] * len(dataset_paths)
    flagged_count = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        count = 0
        for path in dataset_paths:
            futures.append(executor.submit(get_ab_data, minio_client=minio_client, path=path, index=count))
            count += 1

        for future in tqdm(as_completed(futures), total=len(dataset_paths)):
            ab_data, flagged, index = future.result()
            if not flagged:
                ab_data_list[index] = ab_data
            else:
                flagged_count += 1

    unflagged_ab_data = []
    for data in tqdm(ab_data_list):
        if data is not None:
            unflagged_ab_data.append(data)

    print("Total flagged selection datapoints = {}".format(flagged_count))
    return unflagged_ab_data


def get_object(client, file_path):
    response = client.get_object("datasets", file_path)
    data = response.data

    return data


def index_select(tensor, dim, index):
    return tensor.gather(dim, index.unsqueeze(dim)).squeeze(dim)

def split_ab_data_vectors(image_pair_data):
    image_x_feature_vector = image_pair_data[0]
    image_y_feature_vector = image_pair_data[1]
    target_probability = image_pair_data[2]

    return image_x_feature_vector, image_y_feature_vector, target_probability


def format_prompt(prompt):
    prompt = prompt.strip()

    while re.search(r'(\([\s,\.:;\|]*\))|(\<[\s,\.:;\|]*\>)|(\[[\s,\.:;\|]*\])|(\{[\s,\.:;\|]*\})', prompt):
        prompt = re.sub(r'(\([\s,\.:;\|]*\))|(\<[\s,\.:;\|]*\>)|(\[[\s,\.:;\|]*\])|(\{[\s,\.:;\|]*\})', '', prompt)

    prompt = re.sub(r'([\[\(\{\<])\s', r'\1', prompt)
    prompt = re.sub(r'\s([\]\)\}\>])', r'\1', prompt)
    prompt = re.sub(r'\s+', ' ', prompt)
    prompt = re.sub(r'(\s?[,;])+', r',', prompt)

    prompt = re.sub(r'^[\.,;\s]+', '', prompt)
    prompt = re.sub(r'[\.,;\s]+$', '', prompt)

    return prompt


def remove_weight(prompt):
    prompt = re.sub(':[-\s0-9,\.]*', ', ', prompt)
    prompt = re.sub(r'[\(\[\{\<\>\}\]\)]+', '', prompt)
    return prompt

def remove_unwanted_chars(phrase: str):
    # remove non ascii
    phrase_encode = phrase.encode("ascii", "ignore")
    phrase = phrase_encode.decode()

    # remove :n.n
    phrase = re.sub('(:\d+.\d+)|(:)', '', phrase)
    # remove parenthesis
    phrase = re.sub(r'[()]', '', phrase)
    # remove brackets
    phrase = re.sub(r'[\[\]]', '', phrase)
    # remove braces
    phrase = re.sub(r'[{}]', '', phrase)
    # remove quotations
    phrase = re.sub(r'[\"\']', '', phrase)
    # remove slashes
    phrase = re.sub(r'[\\\/]', '', phrase)
    # remove or character
    phrase = re.sub(r'[\|]', '', phrase)
    # remove dash
    phrase = re.sub(r'[\-]', '', phrase)
    # remove underscores
    phrase = re.sub(r'[\_]', '', phrase)
    # remove weird ints and floats
    phrase = re.sub('(\d+.\d+)|(\d+)', '', phrase)
    # remove tabs
    phrase = re.sub(r'[\t]', '', phrase)
    # remove newlines
    phrase = re.sub(r'[\n]', '', phrase)
    # remove carriage returns
    phrase = re.sub(r'[\r]', '', phrase)
    # remove double space
    phrase = re.sub(r'\s{2,}', ' ', phrase)

    return phrase


def get_phrases_from_prompt(prompt):
    prompt = format_prompt(prompt)
    prompt = remove_weight(prompt)
    prompt = format_prompt(prompt)

    phrases = []
    for phrase in prompt.split(','):
        phrase = phrase.strip()
        phrase = remove_unwanted_chars(phrase)

        if len(phrase) == 0:
            continue
        phrases.append(phrase)

    return phrases
