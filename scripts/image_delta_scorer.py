import os
import sys
import argparse
import io
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

base_directory = "./"
sys.path.insert(0, base_directory)

from scripts.image_scorer import ImageScorer
from utility.http import model_training_request
from utility.http import request
from utility.minio import cmd


# delta score = text embedding sigma score - clip embedding sigma score
def get_delta_score(clip_hash_sigma_score_dict,
                    embedding_hash_sigma_score_dict):
    hash_delta_score_dict = {}
    for img_hash, clip_sigma_score in clip_hash_sigma_score_dict.items():
        if img_hash not in embedding_hash_sigma_score_dict:
            continue
        delta_score = embedding_hash_sigma_score_dict[img_hash] - clip_sigma_score
        hash_delta_score_dict[img_hash] = delta_score

    return hash_delta_score_dict


def upload_scores_attributes_to_completed_jobs(clip_hash_score_dict,
                                               clip_hash_sigma_score_dict,
                                               embedding_hash_score_dict,
                                               embedding_hash_sigma_score_dict,
                                               hash_delta_score_dict,
                                               clip_hash_percentile_dict,
                                               embedding_hash_percentile_dict
                                               ):
    print("Uploading scores, sigma scores, and delta scores to mongodb...")
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = []
        for img_hash, clip_score in clip_hash_score_dict.items():
            clip_percentile = clip_hash_percentile_dict[img_hash]
            clip_sigma_score = clip_hash_sigma_score_dict[img_hash]
            if img_hash not in embedding_hash_score_dict:
                continue
            embedding_score = embedding_hash_score_dict[img_hash]
            embedding_percentile = embedding_hash_percentile_dict[img_hash]
            embedding_sigma_score = embedding_hash_sigma_score_dict[img_hash]

            if img_hash not in hash_delta_score_dict:
                continue

            delta_score = hash_delta_score_dict[img_hash]


            futures.append(executor.submit(request.http_add_score_attributes,
                                           img_hash=img_hash,
                                           image_clip_score=clip_score,
                                           image_clip_percentile=clip_percentile,
                                           image_clip_sigma_score=clip_sigma_score,
                                           text_embedding_score=embedding_score,
                                           text_embedding_percentile=embedding_percentile,
                                           text_embedding_sigma_score=embedding_sigma_score,
                                           delta_sigma_score=delta_score,))

        for _ in tqdm(as_completed(futures), total=len(futures)):
            continue


def convert_pairs_to_dict(hash_score_pairs):
    hash_score_dict = {}
    for i in range(len(hash_score_pairs)):
        hash_score_dict[hash_score_pairs[i][0]] = hash_score_pairs[i][1]

    return hash_score_dict

def get_csv_row_data(index,
                     img_hash,
                     clip_score,
                     embedding_hash_score_dict,
                     clip_hash_sigma_score_dict,
                     embedding_hash_sigma_score_dict,
                     clip_hash_percentile_dict,
                     embedding_hash_percentile_dict,
                     hash_delta_score_dict):
    # get job
    job = request.http_get_completed_job_by_image_hash(img_hash)
    if job is None:
        return None, index

    job_uuid = job["uuid"]
    prompt_generation_policy = "N/A"
    if "prompt_generation_policy" in job["task_input_dict"]:
        prompt_generation_policy = job["task_input_dict"]["prompt_generation_policy"]

    positive_prompt = "N/A"
    if "positive_prompt" in job["task_input_dict"]:
        positive_prompt = job["task_input_dict"]["positive_prompt"]

    negative_prompt = "N/A"
    if "negative_prompt" in job["task_input_dict"]:
        negative_prompt = job["task_input_dict"]["negative_prompt"]

    if img_hash not in embedding_hash_score_dict:
        return None, index
    if img_hash not in clip_hash_sigma_score_dict:
        return None, index

    embedding_score = embedding_hash_score_dict[img_hash]
    clip_sigma_score = clip_hash_sigma_score_dict[img_hash]
    embedding_sigma_score = embedding_hash_sigma_score_dict[img_hash]
    delta_score = hash_delta_score_dict[img_hash]
    clip_percentile = clip_hash_percentile_dict[img_hash]
    embedding_percentile = embedding_hash_percentile_dict[img_hash]
    row = [job_uuid, clip_score, clip_percentile, embedding_score, embedding_percentile, clip_sigma_score, embedding_sigma_score, delta_score,
           prompt_generation_policy, positive_prompt, negative_prompt]

    return row, index


def get_delta_score_csv_data(clip_hash_score_dict,
                             clip_hash_sigma_score_dict,
                             clip_hash_percentile_dict,
                             embedding_hash_score_dict,
                             embedding_hash_sigma_score_dict,
                             embedding_hash_percentile_dict,
                             hash_delta_score_dict):
    print("Processing delta score csv data...")
    csv_data = [None] * len(clip_hash_score_dict)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        count = 0
        for img_hash, clip_score in tqdm(clip_hash_score_dict.items()):
            futures.append(executor.submit(get_csv_row_data,
                                           index=count,
                                           img_hash=img_hash,
                                           clip_score=clip_score,
                                           embedding_hash_score_dict=embedding_hash_score_dict,
                                           clip_hash_sigma_score_dict=clip_hash_sigma_score_dict,
                                           embedding_hash_sigma_score_dict=embedding_hash_sigma_score_dict,
                                           clip_hash_percentile_dict=clip_hash_percentile_dict,
                                           embedding_hash_percentile_dict=embedding_hash_percentile_dict,
                                           hash_delta_score_dict=hash_delta_score_dict))
            count += 1

        for future in tqdm(as_completed(futures), total=len(futures)):
            row, index = future.result()
            if row is None:
                continue

            csv_data[index] = row

    cleaned_csv_data = []
    for data in csv_data:
        if data is not None:
            cleaned_csv_data.append(data)

    return cleaned_csv_data


def upload_delta_score_to_csv(minio_client,
                              dataset,
                              clip_model_name,
                              embedding_model_name,
                              delta_score_csv_data):
    print("Saving delta score data to csv...")
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow((["job_uuid", "clip_score", "clip_percentile", "embedding_score", "embedding_percentile","clip_sigma_score" , "embedding_sigma_score", "delta_score", "prompt_generation_policy", "positive_prompt", "negative_prompt"]))

    for row in delta_score_csv_data:
        writer.writerow(row)

    bytes_buffer = io.BytesIO(bytes(csv_buffer.getvalue(), "utf-8"))
    # upload the csv
    clip_model_name = clip_model_name.replace(".safetensors","")
    embedding_model_name = embedding_model_name.replace(".safetensors","")
    filename = "{}-{}-delta-scores.csv".format(clip_model_name, embedding_model_name)
    csv_path = os.path.join(dataset, "output/delta-scores-csv", filename)
    cmd.upload_data(minio_client, 'datasets', csv_path, bytes_buffer)

def generate_delta_scores_graph(minio_client,
                                dataset,
                                clip_model_name,
                                embedding_model_name,
                                clip_hash_score_dict,
                                clip_hash_sigma_score_dict,
                                embedding_hash_score_dict,
                                embedding_hash_sigma_score_dict,
                                hash_delta_score_dict):
    # Initialize all graphs/subplots
    plt.figure(figsize=(22, 20))
    figure_shape = (6, 2)
    clip_scores_hist = plt.subplot2grid(figure_shape, (0, 0), rowspan=1, colspan=1)
    embedding_scores_hist = plt.subplot2grid(figure_shape, (0, 1), rowspan=1, colspan=1)
    scores_graph = plt.subplot2grid(figure_shape, (1, 0), rowspan=1, colspan=2)
    clip_sigma_scores_hist = plt.subplot2grid(figure_shape, (2, 0), rowspan=1, colspan=1)
    embedding_sigma_scores_hist = plt.subplot2grid(figure_shape, (2, 1), rowspan=1, colspan=1)
    sigma_scores_graph = plt.subplot2grid(figure_shape, (3, 0), rowspan=1, colspan=2)
    delta_scores_hist = plt.subplot2grid(figure_shape, (4, 0), rowspan=1, colspan=2)
    delta_scores_graph = plt.subplot2grid(figure_shape, (5, 0), rowspan=1, colspan=2)

    clip_scores_data = []
    embedding_scores_data = []
    for img_hash, embedding_score in embedding_hash_score_dict.items():
        embedding_scores_data.append(embedding_score)
        if img_hash in clip_hash_score_dict:
            clip_scores_data.append(clip_hash_score_dict[img_hash])

    # clip scores hist
    clip_scores_hist.set_xlabel("Clip Score")
    clip_scores_hist.set_ylabel("Frequency")
    clip_scores_hist.set_title("Clip Scores Histogram")
    clip_scores_hist.hist(clip_scores_data,
                                weights=np.ones(len(clip_scores_data)) / len(
                                    clip_scores_data))
    # clip_scores_hist.yaxis.set_major_formatter(PercentFormatter(1))
    clip_scores_hist.ticklabel_format(useOffset=False)

    # embedding scores hist
    embedding_scores_hist.set_xlabel("Embedding Score")
    embedding_scores_hist.set_ylabel("Frequency")
    embedding_scores_hist.set_title("Embedding Scores Histogram")
    embedding_scores_hist.hist(embedding_scores_data,
                          weights=np.ones(len(embedding_scores_data)) / len(
                              embedding_scores_data))
    # embedding_scores_hist.yaxis.set_major_formatter(PercentFormatter(1))
    embedding_scores_hist.ticklabel_format(useOffset=False)

    # scores
    x_axis_values = [i for i in range(len(clip_scores_data))]
    scores_graph.plot(x_axis_values, clip_scores_data,
                        label="Clip Scores",
                        c="#281ad9")

    x_axis_values = [i for i in range(len(embedding_scores_data))]
    scores_graph.plot(x_axis_values, embedding_scores_data,
                          label="Embedding Scores",
                          c="#D95319")

    scores_graph.set_xlabel("Image")
    scores_graph.set_ylabel("Score")
    scores_graph.set_title("Score vs Image")
    scores_graph.legend()
    scores_graph.autoscale(enable=True, axis='y')

    clip_sigma_scores_data = []
    embedding_sigma_scores_data = []
    for img_hash, clip_sigma_score in clip_hash_sigma_score_dict.items():
        clip_sigma_scores_data.append(clip_sigma_score)
        if img_hash in embedding_hash_sigma_score_dict:
            embedding_sigma_scores_data.append(embedding_hash_sigma_score_dict[img_hash])

    # clip sigma scores hist
    clip_sigma_scores_hist.set_xlabel("Clip Sigma Score")
    clip_sigma_scores_hist.set_ylabel("Frequency")
    clip_sigma_scores_hist.set_title("Clip Sigma Scores Histogram")
    clip_sigma_scores_hist.hist(clip_sigma_scores_data,
                          weights=np.ones(len(clip_sigma_scores_data)) / len(
                              clip_sigma_scores_data))
    # clip_sigma_scores_hist.yaxis.set_major_formatter(PercentFormatter(1))
    clip_sigma_scores_hist.ticklabel_format(useOffset=False)

    # embedding sigma scores hist
    embedding_sigma_scores_hist.set_xlabel("Embedding Sigma Score")
    embedding_sigma_scores_hist.set_ylabel("Frequency")
    embedding_sigma_scores_hist.set_title("Embedding Sigma Scores Histogram")
    embedding_sigma_scores_hist.hist(embedding_sigma_scores_data,
                                weights=np.ones(len(embedding_sigma_scores_data)) / len(
                                    embedding_sigma_scores_data))
    # embedding_sigma_scores_hist.yaxis.set_major_formatter(PercentFormatter(1))
    embedding_sigma_scores_hist.ticklabel_format(useOffset=False)

    # sigma scores
    x_axis_values = [i for i in range(len(clip_sigma_scores_data))]
    sigma_scores_graph.plot(x_axis_values, clip_sigma_scores_data,
                      label="Clip Sigma Scores",
                      c="#281ad9")
    x_axis_values = [i for i in range(len(embedding_sigma_scores_data))]
    sigma_scores_graph.plot(x_axis_values, embedding_sigma_scores_data,
                      label="Embedding Sigma Scores",
                      c="#D95319")

    sigma_scores_graph.set_xlabel("Image")
    sigma_scores_graph.set_ylabel("Sigma Score")
    sigma_scores_graph.set_title("Sigma Score vs Image")
    sigma_scores_graph.legend()
    sigma_scores_graph.autoscale(enable=True, axis='y')

    # delta scores
    delta_scores_data = []
    for img_hash, delta_score in hash_delta_score_dict.items():
        delta_scores_data.append(delta_score)

    # delta scores hist
    delta_scores_hist.set_xlabel("Delta Score")
    delta_scores_hist.set_ylabel("Frequency")
    delta_scores_hist.set_title("Delta Scores Histogram")
    delta_scores_hist.hist(delta_scores_data,
                           weights=np.ones(len(delta_scores_data)) / len(
                                         delta_scores_data))
    # delta_scores_hist.yaxis.set_major_formatter(PercentFormatter(1))
    delta_scores_hist.ticklabel_format(useOffset=False)

    x_axis_values = [i for i in range(len(hash_delta_score_dict))]
    delta_scores_graph.plot(x_axis_values, delta_scores_data,
                      label="Delta Scores",
                      c="#281ad9")

    delta_scores_graph.set_xlabel("Image")
    delta_scores_graph.set_ylabel("Delta Score")
    delta_scores_graph.set_title("Delta Score vs Image")
    delta_scores_graph.legend()
    delta_scores_graph.autoscale(enable=True, axis='y')

    # Save figure
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # upload the graph report
    clip_model_name = clip_model_name.replace(".safetensors", "")
    embedding_model_name = embedding_model_name.replace(".safetensors", "")
    filename = "{}-{}-delta-scores.png".format(clip_model_name, embedding_model_name)
    graph_output = os.path.join(dataset, "output/delta-scores-graph", filename)
    cmd.upload_data(minio_client, 'datasets', graph_output, buf)

def run_image_delta_scorer(minio_client,
                           dataset_name,
                           clip_model_filename,
                           embedding_model_filename):

    start_time = time.time()

    # clip
    clip_scorer = ImageScorer(minio_client=minio_client,
                              dataset_name=dataset_name,
                              model_name=clip_model_filename)

    clip_scorer.load_model()
    clip_paths = clip_scorer.get_paths()
    clip_hash_score_pairs, image_paths = clip_scorer.get_scores(clip_paths)
    clip_hash_percentile_dict = clip_scorer.get_percentiles(clip_hash_score_pairs)
    clip_hash_sigma_score_dict = clip_scorer.get_sigma_scores(clip_hash_score_pairs)

    # embedding
    embedding_scorer = ImageScorer(minio_client=minio_client,
                                   dataset_name=dataset_name,
                                   model_name=embedding_model_filename)

    embedding_scorer.load_model()
    embedding_paths = embedding_scorer.get_paths()
    embedding_hash_score_pairs, image_paths = embedding_scorer.get_scores(embedding_paths)
    embedding_hash_percentile_dict = embedding_scorer.get_percentiles(embedding_hash_score_pairs)

    embedding_hash_sigma_score_dict = embedding_scorer.get_sigma_scores(embedding_hash_score_pairs)

    hash_delta_score_dict = get_delta_score(clip_hash_sigma_score_dict=clip_hash_sigma_score_dict,
                                            embedding_hash_sigma_score_dict=embedding_hash_sigma_score_dict)

    clip_hash_score_dict = convert_pairs_to_dict(clip_hash_score_pairs)
    embedding_hash_score_dict = convert_pairs_to_dict(embedding_hash_score_pairs)

    delta_score_csv_data = get_delta_score_csv_data(clip_hash_score_dict=clip_hash_score_dict,
                                                    clip_hash_sigma_score_dict=clip_hash_sigma_score_dict,
                                                    clip_hash_percentile_dict=clip_hash_percentile_dict,
                                                    embedding_hash_score_dict=embedding_hash_score_dict,
                                                    embedding_hash_sigma_score_dict=embedding_hash_sigma_score_dict,
                                                    embedding_hash_percentile_dict=embedding_hash_percentile_dict,
                                                    hash_delta_score_dict=hash_delta_score_dict,
                                                    )

    upload_scores_attributes_to_completed_jobs(clip_hash_score_dict=clip_hash_score_dict,
                                               clip_hash_sigma_score_dict=clip_hash_sigma_score_dict,
                                               embedding_hash_score_dict=embedding_hash_score_dict,
                                               embedding_hash_sigma_score_dict=embedding_hash_sigma_score_dict,
                                               hash_delta_score_dict=hash_delta_score_dict,
                                               clip_hash_percentile_dict=clip_hash_percentile_dict,
                                               embedding_hash_percentile_dict=embedding_hash_percentile_dict)

    upload_delta_score_to_csv(minio_client=minio_client,
                              dataset=dataset_name,
                              clip_model_name=clip_model_filename,
                              embedding_model_name=embedding_model_filename,
                              delta_score_csv_data=delta_score_csv_data)

    generate_delta_scores_graph(minio_client=minio_client,
                                dataset=dataset_name,
                                clip_model_name=clip_model_filename,
                                embedding_model_name=embedding_model_filename,
                                clip_hash_score_dict=clip_hash_score_dict,
                                clip_hash_sigma_score_dict=clip_hash_sigma_score_dict,
                                embedding_hash_score_dict=embedding_hash_score_dict,
                                embedding_hash_sigma_score_dict=embedding_hash_sigma_score_dict,
                                hash_delta_score_dict=hash_delta_score_dict)

    time_elapsed = time.time() - start_time
    print("Dataset: {}: Total Time elapsed: {}s".format(dataset_name, format(time_elapsed, ".2f")))


def parse_args():
    parser = argparse.ArgumentParser(description="Image Delta Scorer")
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--dataset-name', required=True, help='Name of the dataset for embeddings')
    parser.add_argument('--clip-model-filename', required=True, help='Filename of the clip model (e.g., "XXX.safetensors")')
    parser.add_argument('--embedding-model-filename', required=True, help='Filename of the embedding model (e.g., "XXX.safetensors")')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    dataset_name = args.dataset_name
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)
    if dataset_name != "all":
        run_image_delta_scorer(minio_client,
                               args.dataset_name,
                               args.clip_model_filename,
                               args.embedding_model_filename)
    else:
        # if all, train models for all existing datasets
        # get dataset name list
        dataset_names = request.http_get_dataset_names()
        print("dataset names=", dataset_names)
        for dataset in dataset_names:
            try:
                run_image_delta_scorer(minio_client,
                                       dataset,
                                       args.clip_model_filename,
                                       args.embedding_model_filename)
            except Exception as e:
                print("Error running image scorer for {}: {}".format(dataset, e))


if __name__ == "__main__":
    main()
