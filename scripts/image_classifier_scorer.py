import os
import sys
import io
import csv
import argparse
import numpy as np
import torch
import time
import msgpack
from io import BytesIO
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm
import math
from datetime import datetime
base_directory = "./"
sys.path.insert(0, base_directory)

from training_worker.classifiers.models.elm_regression import ELMRegression
from training_worker.classifiers.models.linear_regression import LinearRegression
from training_worker.classifiers.models.logistic_regression import LogisticRegression
from utility.http import model_training_request
from utility.http import request
from utility.minio import cmd

class ImageScorer:
    def __init__(self,
                 minio_client,
                 dataset_name="characters"):
        self.minio_client = minio_client
        self.model = None
        self.dataset = dataset_name
        self.model_name = None
        self.model_input_type = None
        self.model_input_type_list = ["embedding-negative", "embedding-positive", "embedding", "clip"]

        self.image_paths_cache = {}
        self.image_all_feature_pairs_cache = {}
        
        self.classifier_id_list = request.http_get_classifier_model_list()

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = torch.device(device)

        print("Dataset=", self.dataset)
        print("device=", self.device)

    def load_model(self, classifier_model_info):
        self.tag_id = classifier_model_info["tag_id"]
        self.classifier_id = classifier_model_info["classifier_id"]
        self.model_name = classifier_model_info["classifier_name"]

        self.model_input_type = None

        for input_type in self.model_input_type_list:
            if input_type in classifier_model_info["classifier_name"]:
                self.model_input_type = input_type
                break
        if self.model_input_type == None:
            print("Not support classifier model: {}".format(classifier_model_info["classifier_name"]))
            return False

        if "elm" in classifier_model_info["classifier_name"]:
            elm_model = ELMRegression(device=self.device)
            loaded_model, model_file_name = elm_model.load_model_with_filename(
                self.minio_client, 
                classifier_model_info["model_path"], 
                classifier_model_info["classifier_name"])
            self.model = loaded_model
        elif "linear" in classifier_model_info["classifier_name"]:
            linear_model = LinearRegression(device=self.device)
            loaded_model, model_file_name = linear_model.load_model_with_filename(
                self.minio_client, 
                classifier_model_info["model_path"], 
                classifier_model_info["classifier_name"])
            self.model = loaded_model
        elif "logistic" in classifier_model_info["classifier_name"]:
            logistic_model = LogisticRegression(device=self.device)
            loaded_model, model_file_name = logistic_model.load_model_with_filename(
                self.minio_client, 
                classifier_model_info["model_path"], 
                classifier_model_info["classifier_name"])
            self.model = loaded_model
        else:
            print("Not support classifier model: {}".format(classifier_model_info["classifier_name"]))

        if not loaded_model:
            return False
        return True

    def get_paths(self):
        print("Getting paths for dataset: {}...".format(self.dataset))
        if self.model_input_type in self.image_paths_cache:
            return self.image_paths_cache[self.model_input_type]
        all_objects = cmd.get_list_of_objects_with_prefix(self.minio_client, 'datasets', self.dataset)

        # Depending on the model type, choose the appropriate msgpack files
        file_suffix = "_clip.msgpack" if self.model_input_type == "clip" else "embedding.msgpack"

        # Filter the objects to get only those that end with the chosen suffix
        type_paths = [obj for obj in all_objects if obj.endswith(file_suffix)]

        self.image_paths_cache[self.model_input_type] = type_paths

        print("Total paths found=", len(type_paths))
        return type_paths

    def get_feature_data(self, job_uuids):
        data = request.http_get_completed_jobs_by_uuids(job_uuids)
        return data

    def get_all_feature_data(self, job_uuids_hash_dict):
        print('Getting features data...')
        # get all job uuids
        job_uuids = []
        hash_job_uuid_dict = {}
        for hash, uuid in job_uuids_hash_dict.items():
            job_uuids.append(uuid)
            hash_job_uuid_dict[uuid] = hash

        features_data_image_hash_dict = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            chunk_size = 100
            for i in range(0, len(job_uuids), chunk_size):
                batch_job_uuids = job_uuids[i:i + chunk_size]
                futures.append(executor.submit(self.get_feature_data, job_uuids=batch_job_uuids))

            for future in tqdm(as_completed(futures), total=round(len(job_uuids)/chunk_size)):
                feature_data_list = future.result()
                for features_data in feature_data_list:
                    job_uuid = features_data["uuid"]
                    generation_policy = "n/a"
                    if "prompt_generation_policy" in features_data["task_input_dict"]:
                        generation_policy = features_data["task_input_dict"]["prompt_generation_policy"]

                    completion_time = features_data["task_completion_time"]
                    positive_prompt = features_data["task_input_dict"]["positive_prompt"]
                    negative_prompt = features_data["task_input_dict"]["negative_prompt"]
                    features_data = (job_uuid, generation_policy, completion_time, positive_prompt, negative_prompt)
                    hash = hash_job_uuid_dict[job_uuid]
                    features_data_image_hash_dict[hash] = features_data

        return features_data_image_hash_dict

    def get_feature_pair(self, path, index):
        # Updated bucket name to 'datasets'
        msgpack_data = cmd.get_file_from_minio(self.minio_client, 'datasets', path)
        if not msgpack_data:
            print(f"No msgpack file found at path: {path}")
            return None, None, None, None, index

        data = msgpack.unpackb(msgpack_data.data)

        image_hash = None
        image_path = None
        first_feature = None
        second_feature = None

        if self.model_input_type == "embedding":
            positive_embedding = list(data['positive_embedding'].values())
            first_feature = torch.tensor(np.array(positive_embedding)).float()

            negative_embedding = list(data['negative_embedding'].values())
            second_feature = torch.tensor(np.array(negative_embedding)).float()

        elif self.model_input_type == "embedding-positive":
            positive_embedding = list(data['positive_embedding'].values())
            first_feature = torch.tensor(np.array(positive_embedding)).float()

        elif self.model_input_type == "embedding-negative":
            negative_embedding = list(data['negative_embedding'].values())
            first_feature = torch.tensor(np.array(negative_embedding)).float()

        elif self.model_input_type == "clip":
            clip_feature = data['clip-feature-vector']
            first_feature = torch.tensor(np.array(clip_feature)).float()

            # clip image hash isn't in clip.msgpack so get it from _data.msgpack
            data_msgpack = cmd.get_file_from_minio(self.minio_client, 'datasets',
                                                   path.replace("clip.msgpack", "data.msgpack"))
            if not data_msgpack:
                print("No msgpack file found at path: {}".format(path.replace("clip.msgpack", "data.msgpack")))
                return None

            data = msgpack.unpackb(data_msgpack.data)

        image_hash = data['file_hash']
        job_uuid = data['job_uuid']

        if self.model_input_type == "clip":
            image_path = data['file_path'].replace("_data.msgpack", ".jpg")
        else:
            image_path = data['file_path'].replace("_embedding.msgpack", ".jpg")

        return image_hash, image_path, first_feature, second_feature, index, job_uuid

    def get_all_feature_pairs(self, msgpack_paths):
        print('Getting dataset features...')
        
        if self.model_input_type in self.image_all_feature_pairs_cache:
            return self.image_all_feature_pairs_cache[self.model_input_type]

        features_data = [None] * len(msgpack_paths)
        image_paths = [None] * len(msgpack_paths)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            count = 0
            for path in msgpack_paths:
                futures.append(executor.submit(self.get_feature_pair, path=path, index=count))
                count += 1

            for future in tqdm(as_completed(futures), total=len(msgpack_paths)):
                image_hash, image_path, first_feature, second_feature, index, job_uuid= future.result()
                features_data[index] = (image_hash,
                                        first_feature,
                                        second_feature,
                                        job_uuid)
                image_paths[index] = image_path

        self.image_all_feature_pairs_cache[self.model_input_type] = (features_data, image_paths)

        return features_data, image_paths

    def get_scores(self, features_data, image_paths):
        hash_score_pairs = []
        job_uuids_hash_dict = {}

        print('Predicting dataset scores...')
        with torch.no_grad():
            count=0
            weird_count = 0
            for data in tqdm(features_data):
                try:
                    if self.model_input_type == "embedding":
                        positive_embedding_array = data[1].to(self.device)
                        negative_embedding_array = data[2].to(self.device)
                        image_hash = data[0]

                        score = self.model.classify_pooled_embeddings(positive_embedding_array, negative_embedding_array)

                    elif self.model_input_type == "embedding-positive":
                        positive_embedding_array = data[1].to(self.device)
                        image_hash = data[0]

                        score = self.model.predict_positive_or_negative_only_pooled(positive_embedding_array)

                    elif self.model_input_type == "embedding-negative":
                        negative_embedding_array = data[1].to(self.device)
                        image_hash = data[0]
                        score = self.model.predict_positive_or_negative_only_pooled(negative_embedding_array)

                    elif self.model_input_type == "clip":
                            clip_feature_vector = data[1].to(self.device)
                            image_hash = data[0]
                            score = self.model.classify(clip_feature_vector)
                except Exception as e:
                    print(f"Skipping vector due to error: {e}")
                    continue
                if score > 100000.0 or score < -100000.0:
                    print("score more than or less than 100k and -100k")
                    print("Score=", score)
                    print("image path=", image_paths[count])
                    weird_count += 1
                    continue

                hash_score_pairs.append((image_hash, score.item()))
                # add job uuids to dict
                job_uuids_hash_dict[image_hash] = data[3]

                count += 1

        print("Weird scores count = ", weird_count)

        return hash_score_pairs, image_paths, job_uuids_hash_dict

    def get_percentiles(self, hash_score_pairs):
        hash_percentile_dict = {}
        sorted_hash_score_pairs = hash_score_pairs.copy()
        sorted_hash_score_pairs.sort(key=lambda a: a[1])

        len_hash_scores = len(sorted_hash_score_pairs)
        for i in range(len(sorted_hash_score_pairs)):
            percentile = i / len_hash_scores
            hash_percentile_dict[sorted_hash_score_pairs[i][0]] = percentile

        return hash_percentile_dict

    def get_sigma_scores(self, hash_score_pairs):
        scores_arr = []
        for i in range(len(hash_score_pairs)):
            score = hash_score_pairs[i][1]
            if math.isnan(score):
                # skip nan
                continue

            scores_arr.append(score)

        scores_np_arr = np.array(scores_arr)
        print("max=", scores_np_arr.max())
        print("min=", scores_np_arr.min())

        mean = scores_np_arr.mean(dtype=np.float64)
        standard_deviation = scores_np_arr.std(dtype=np.float64)

        print("numpy arr=", scores_np_arr)
        print("mean=", mean)
        print("standard_dev=", standard_deviation)

        hash_sigma_score_dict = {}
        for i in range(len(hash_score_pairs)):
            score = hash_score_pairs[i][1]
            sigma_score = (score - mean) / standard_deviation
            hash_sigma_score_dict[hash_score_pairs[i][0]] = float(sigma_score)

        return hash_sigma_score_dict

    def upload_csv(self,
                   hash_score_pairs,
                   hash_percentile_dict,
                   hash_sigma_score_dict,
                   image_paths,
                   features_data_job_uuid_dict):
        print("Saving data to csv...")
        csv_data = []
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow((["Job UUID", "Image Hash", "Image Path", "Completed Time", "Generation Policy", "Positive Prompt", "Negative Prompt", "Score", "Percentile", "Sigma Score"]))

        count = 0
        for image_hash, score in hash_score_pairs:
            job_uuid = "n/a"
            generation_policy = "n/a"
            completed_time = "n/a"
            positive_prompt = "n/a"
            negative_prompt = "n/a"
            if image_hash in features_data_job_uuid_dict:
                features_data = features_data_job_uuid_dict[image_hash]
                job_uuid = features_data[0]
                generation_policy = features_data[1]
                completed_time = features_data[2]
                positive_prompt = features_data[3]
                negative_prompt = features_data[4]

            row = [job_uuid, image_hash, image_paths[count], completed_time, generation_policy, positive_prompt, negative_prompt, score, hash_percentile_dict[image_hash], hash_sigma_score_dict[image_hash]]
            writer.writerow(row)
            csv_data.append(row)
            count += 1

        bytes_buffer = io.BytesIO(bytes(csv_buffer.getvalue(), "utf-8"))
        # upload the csv
        csv_path = os.path.join(self.dataset, "output/scores-csv", self.model_name.replace(".safetensors", ".csv"))
        cmd.upload_data(self.minio_client, 'datasets', csv_path, bytes_buffer)

        return csv_data

    def upload_scores(self, hash_score_pairs, job_uuids_hash_dict):
        print("Uploading scores to mongodb...")
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = []
            for pair in hash_score_pairs:
                # upload score
                score_data = {
                    "job_uuid": job_uuids_hash_dict[pair[0]],
                    "classifier_id": self.classifier_id,
                    "score": pair[1],
                }
                futures.append(executor.submit(request.http_add_classifier_score, score_data=score_data))

            for _ in tqdm(as_completed(futures), total=len(hash_score_pairs)):
                continue

    def get_classifier_id_and_name(self, classifier_file_path):
        for classifier in self.classifier_id_list:
            if classifier["model_path"] == classifier_file_path:
                return classifier["classifier_id"], classifier["classifier_name"]
        return -1, ""
    
    def upload_sigma_scores(self, hash_sigma_score_dict):
        print("Uploading sigma scores to mongodb...")
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = []
            for key, val in hash_sigma_score_dict.items():
                # upload score
                sigma_score_data = {
                    "image_hash": key,
                    "sigma_score": val,
                }

                futures.append(executor.submit(request.http_add_sigma_score, sigma_score_data=sigma_score_data))

            for _ in tqdm(as_completed(futures), total=len(futures)):
                continue

    def upload_percentile(self, hash_percentile_dict):
        print("Uploading percentiles to mongodb...")
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = []
            for image_hash, percentile in hash_percentile_dict.items():
                # upload percentile
                percentile_data = {
                    "image_hash": image_hash,
                    "percentile": percentile,
                }

                futures.append(executor.submit(request.http_add_percentile, percentile_data=percentile_data))

            for _ in tqdm(as_completed(futures), total=len(hash_percentile_dict)):
                continue

    def generate_graphs(self, hash_score_pairs, hash_percentile_dict, hash_sigma_score_dict):
        # Initialize all graphs/subplots
        plt.figure(figsize=(22, 20))
        figure_shape = (4, 1)
        percentile_graph = plt.subplot2grid(figure_shape, (0, 0), rowspan=1, colspan=1)
        score_graph = plt.subplot2grid(figure_shape, (1, 0), rowspan=1, colspan=1)
        sigma_score_graph = plt.subplot2grid(figure_shape, (2, 0), rowspan=1, colspan=1)
        hist_sigma_score_graph = plt.subplot2grid(figure_shape, (3, 0), rowspan=1, colspan=1)

        # percentiles
        chronological_percentiles = []
        for pair in hash_score_pairs:
            chronological_percentiles.append(hash_percentile_dict[pair[0]])

        x_axis_values = [i for i in range(len(hash_score_pairs))]
        percentile_graph.scatter(x_axis_values, chronological_percentiles,
                            label="Image Percentiles over time",
                            c="#281ad9", s=15)

        percentile_graph.set_xlabel("Time")
        percentile_graph.set_ylabel("Percentile")
        percentile_graph.set_title("Percentile vs Time")
        percentile_graph.legend()
        percentile_graph.autoscale(enable=True, axis='y')

        # scores
        chronological_scores = []
        for pair in hash_score_pairs:
            chronological_scores.append(pair[1])

        score_graph.scatter(x_axis_values, chronological_scores,
                                 label="Image Scores over time",
                                 c="#281ad9", s=15)

        score_graph.set_xlabel("Time")
        score_graph.set_ylabel("Score")
        score_graph.set_title("Score vs Time")
        score_graph.legend()
        score_graph.autoscale(enable=True, axis='y')

        # sigma scores
        chronological_sigma_scores = []
        for pair in hash_score_pairs:
            chronological_sigma_scores.append(hash_sigma_score_dict[pair[0]])

        sigma_score_graph.scatter(x_axis_values, chronological_sigma_scores,
                            label="Image Sigma Scores over time",
                            c="#281ad9", s=15)

        sigma_score_graph.set_xlabel("Time")
        sigma_score_graph.set_ylabel("Sigma Score")
        sigma_score_graph.set_title("Sigma Score vs Time")
        sigma_score_graph.legend()
        sigma_score_graph.autoscale(enable=True, axis='y')

        # hist sigma scores
        hist_sigma_score_graph.set_xlabel("Sigma Score")
        hist_sigma_score_graph.set_ylabel("Frequency")
        hist_sigma_score_graph.set_title("Sigma Scores Histogram")
        hist_sigma_score_graph.hist(chronological_sigma_scores,
                                    weights=np.ones(len(chronological_sigma_scores)) / len(
                                        chronological_sigma_scores))
        hist_sigma_score_graph.yaxis.set_major_formatter(PercentFormatter(1))

        # Save figure
        plt.suptitle('Dataset: {}\nModel: {}'.format(self.dataset, self.model_name), fontsize=20)

        # plt.subplots_adjust(left=0.15, hspace=0.5)
        # plt.savefig("./output/{}-{}-scores.jpg".format(self.model_name.replace(".safetensors", ""), self.dataset))
        plt.show()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # upload the graph report
        graph_name = "{}-{}.png".format(self.model_name.replace(".safetensors", ""), self.dataset)
        graph_output = os.path.join(self.dataset, "output/scores-graph", graph_name)
        cmd.upload_data(self.minio_client, 'datasets', graph_output, buf)

    def generate_graphs_by_policy(self,
                                  csv_data,
                                  policy="top-k"):
        # get scores for policy
        scores = []
        percentiles = []
        sigma_scores = []
        for row in csv_data:
            row_policy = row[4]
            if  row_policy == policy:
                score = row[5]
                scores.append(score)

                percentile = row[6]
                percentiles.append(percentile)

                sigma_score = row[7]
                sigma_scores.append(sigma_score)

        # Initialize all graphs/subplots
        plt.figure(figsize=(22, 20))
        figure_shape = (3, 1)
        percentile_graph = plt.subplot2grid(figure_shape, (0, 0), rowspan=1, colspan=1)
        score_graph = plt.subplot2grid(figure_shape, (1, 0), rowspan=1, colspan=1)
        sigma_score_graph = plt.subplot2grid(figure_shape, (2, 0), rowspan=1, colspan=1)

        # percentiles
        x_axis_values = [i for i in range(len(percentiles))]
        percentile_graph.scatter(x_axis_values, percentiles,
                            label="Image Percentiles over time",
                            c="#281ad9", s=15)

        percentile_graph.set_xlabel("Time")
        percentile_graph.set_ylabel("Percentile")
        percentile_graph.set_title("Percentile vs Time")
        percentile_graph.legend()
        percentile_graph.autoscale(enable=True, axis='y')

        # scores
        score_graph.scatter(x_axis_values, scores,
                                 label="Image Scores over time",
                                 c="#281ad9", s=15)

        score_graph.set_xlabel("Time")
        score_graph.set_ylabel("Score")
        score_graph.set_title("Score vs Time")
        score_graph.legend()
        score_graph.autoscale(enable=True, axis='y')

        # sigma scores

        sigma_score_graph.scatter(x_axis_values, sigma_scores,
                            label="Image Sigma Scores over time",
                            c="#281ad9", s=15)

        sigma_score_graph.set_xlabel("Time")
        sigma_score_graph.set_ylabel("Sigma Score")
        sigma_score_graph.set_title("Sigma Score vs Time")
        sigma_score_graph.legend()
        sigma_score_graph.autoscale(enable=True, axis='y')

        # Save figure
        plt.suptitle('Dataset: {}\nModel: {}\nPolicy: {}'.format(self.dataset, self.model_name, policy), fontsize=20)

        # plt.subplots_adjust(left=0.15, hspace=0.5)
        # plt.savefig("./output/{}-{}-scores.jpg".format(self.model_name.replace(".safetensors", ""), self.dataset))
        plt.show()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # upload the graph report
        graph_name = "{}-{}.png".format(self.model_name.replace(".safetensors", ""), self.dataset)
        graph_output = os.path.join(self.dataset, "output/scores-graph", graph_name)
        cmd.upload_data(self.minio_client, 'datasets', graph_output, buf)

def parse_args():
    parser = argparse.ArgumentParser(description="Embedding Scorer")
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--dataset-name', required=True, help='Name of the dataset for embeddings')

    args = parser.parse_args()
    return args


def run_image_scorer(minio_client,
                     dataset_name):
    start_time = time.time()
    # remove
    print("run_image_scorer")

    scorer = ImageScorer(minio_client=minio_client,
                         dataset_name=dataset_name)


    classifier_model_list = request.http_get_classifier_model_list()
    
    for classifier_model in classifier_model_list:
        try:
            is_loaded = scorer.load_model(classifier_model_info=classifier_model)
        except Exception as e:
            print("Failed loading model, {}".format(classifier_model["model_path"]), e)
            continue
        if not is_loaded:
            continue

        paths = scorer.get_paths()
        print(paths)
        features_data, image_paths = scorer.get_all_feature_pairs(paths)
        
        hash_score_pairs, image_paths, job_uuids_hash_dict = scorer.get_scores(features_data, image_paths)
        print("Successfully calculated")
        scorer.upload_scores(hash_score_pairs, job_uuids_hash_dict)

    time_elapsed = time.time() - start_time
    print("Dataset: {}: Total Time elapsed: {}s".format(dataset_name, format(time_elapsed, ".2f")))   


def main():
    args = parse_args()

    dataset_name = args.dataset_name

    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)

    if dataset_name != "all":
        run_image_scorer(minio_client, dataset_name)
    else:
        # if all, train models for all existing datasets
        # get dataset name list
        dataset_names = request.http_get_dataset_names()
        print("dataset names=", dataset_names)
        for dataset in dataset_names:
            try:
                run_image_scorer(minio_client, dataset)
            except Exception as e:
                print("Error running image scorer for {}: {}".format(dataset, e))


if __name__ == "__main__":
    main()
