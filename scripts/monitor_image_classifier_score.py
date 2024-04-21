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
import random

base_directory = "./"
sys.path.insert(0, base_directory)

from training_worker.classifiers.models.elm_regression import ELMRegression
from training_worker.classifiers.models.linear_regression import LinearRegression
from training_worker.classifiers.models.logistic_regression import LogisticRegression
from utility.http import model_training_request
from utility.http import request
from utility.minio import cmd

from utility.path import separate_bucket_and_file_path

class ImageScorer:
    def __init__(self,
                 minio_client,
                 dataset_names=[],
                 batch_size=100):
        self.minio_client = minio_client
        self.model = None
        self.datasets = dataset_names
        self.model_name = None
        self.model_input_type = None
        self.model_input_type_list = ["embedding-negative", "embedding-positive", "embedding", "clip"]

        self.image_paths_cache = {}
        self.image_all_feature_pairs_cache = {}
        
        self.classifier_id_list = request.http_get_classifier_model_list()

        self.batch_size = batch_size

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = torch.device(device)

        print("Dataset=", self.datasets)
        print("device=", self.device)

    def load_model(self, classifier_model_info):
        print("loading model...")

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
            loaded_model = False
            print("Not support classifier model: {}".format(classifier_model_info["classifier_name"]))

        if not loaded_model:
            return False
        return True

    def get_paths(self):
        print("Getting paths for dataset: {}...".format(self.datasets))

        all_objects = []
        
        # Depending on the model type, choose the appropriate msgpack files
        file_suffix = "_clip.msgpack" if self.model_input_type == "clip" else "_embedding.msgpack"
        
        for dataset in self.datasets:
            print("Getting paths for dataset: {}...".format(dataset))
            all_objects.extend(cmd.get_list_of_objects_with_prefix(self.minio_client, 'datasets', dataset))

            # Filter the objects to get only those that end with the chosen suffix
            type_paths = [obj for obj in all_objects if obj.endswith(file_suffix)]
            if len(type_paths) > 100000:
                break

        print("Total paths found=", len(type_paths))
        return type_paths
    
    def get_paths_from_mongodb(self):
        print("Getting paths for dataset: {}...".format(self.datasets))
        completed_jobs = []
        file_suffix = "_clip.msgpack" if self.model_input_type == "clip" else "_embedding.msgpack"
        for dataset in self.datasets:
            completed_jobs.extend(request.http_get_completed_job_by_dataset(dataset=dataset))
            if len(completed_jobs) > 100000:
                break
        image_paths = []
        for job in completed_jobs:
            try:
                if job["task_output_file_dict"]["output_file_path"].endswith(".jpg"):
                    image_paths.append(job["task_output_file_dict"]["output_file_path"].replace(".jpg", file_suffix))
            except Exception as e:
                print("Error to get image path from mongodb: {}".format(e))

        return image_paths

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
        _, path = separate_bucket_and_file_path(path)
        msgpack_data = cmd.get_file_from_minio(self.minio_client, 'datasets', path)
        if not msgpack_data:
            print(f"No msgpack file found at path: {path}")
            return None, path, None, None, index, None

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
                if image_hash is None:
                    continue
                features_data[index] = (image_hash,
                                        first_feature,
                                        second_feature,
                                        job_uuid)
                image_paths[index] = image_path

        return features_data, image_paths

    def get_scores(self, features_data, image_paths):
        hash_score_pairs = []
        job_uuids_hash_dict = {}

        print('Predicting dataset scores...')
        with torch.no_grad():
            count=0
            weird_count = 0
            len_features_data = len(features_data)
            for start_index in range(0, len_features_data, self.batch_size):
                data = features_data[start_index:min(start_index + self.batch_size, len_features_data)]
                image_path = image_paths[start_index:min(start_index + self.batch_size, len_features_data)]
                try:
                    if self.model_input_type == "embedding":
                        positive_embedding_array = []
                        negative_embedding_array = []
                        image_hash = []

                        for i in len(data):
                            positive_embedding_array.append(data[i][1])
                            negative_embedding_array.append(data[i][2])
                            image_hash.append(data[i][0])
                        
                        positive_embedding_array = torch.cat(positive_embedding_array.squeeze(), dim=0).to(self.device)
                        negative_embedding_array = torch.cat(negative_embedding_array.squeeze(), dim=0).to(self.device)

                        score = self.model.classify_pooled_embeddings(positive_embedding_array, negative_embedding_array).squeeze()

                    elif self.model_input_type == "embedding-positive":
                        positive_embedding_array = []
                        image_hash = []

                        for i in range(0, len(data)):
                            positive_embedding_array.append(data[i][1])
                            image_hash.append(data[i][0])
                        
                        positive_embedding_array = torch.cat(positive_embedding_array, dim=0).to(self.device)

                        score = self.model.predict_positive_or_negative_only_pooled(positive_embedding_array).squeeze()
                    elif self.model_input_type == "embedding-negative":
                        negative_embedding_array = []
                        image_hash = []

                        for i in range(0, len(data)):
                            negative_embedding_array.append(data[i][2])
                            image_hash.append(data[i][0])
                        
                        negative_embedding_array = torch.cat(negative_embedding_array, dim=0).to(self.device)

                        score = self.model.predict_positive_or_negative_only_pooled(negative_embedding_array).squeeze()
                    elif self.model_input_type == "clip":
                         
                        clip_feature_vector = []
                        image_hash = []

                        for i in range(0, len(data)):
                            clip_feature_vector.append(data[i][1])
                            image_hash.append(data[i][0])
                        
                        clip_feature_vector = torch.cat(clip_feature_vector, dim=0).to(self.device)

                        score = self.model.classify(clip_feature_vector).squeeze()
                except Exception as e:
                    print(f"Skipping vector due to error: {e}")
                    continue
                
                # print the image path and score where score is too much or too small
                invalid_mask = torch.where(score > 0.5)[0].tolist()

                if len(invalid_mask) > 0:
                    print("score more than or less than 100k and -100k")
                    print("image path and score")
                    invalid_score = score[invalid_mask].tolist()
                    invalid_image_path = [image_path[i] for i in invalid_mask]
                    print(list(zip(invalid_image_path, invalid_score)))

                    weird_count += len(score[invalid_mask])
                    
                hash_score_pairs.extend(list(zip(image_hash, score.tolist())))

                # add job uuids to dict
                for j in range(len(image_hash)):
                    image_hash_str = str(image_hash[j])
                    job_uuids_hash_dict[image_hash_str] = data[j][3]

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

    def upload_scores(self, hash_score_pairs, job_uuids_hash_dict):
        print("Uploading scores to mongodb...")
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = []
            len_hash_score_pairs = len(hash_score_pairs)
            for start_index in tqdm(range(0, len_hash_score_pairs, self.batch_size)):
                pairs = hash_score_pairs[start_index:min(start_index+self.batch_size, len_hash_score_pairs)]
                score_data_list = []
                for pair in pairs:
                    # upload score
                    score_data_list.append({
                        "job_uuid": job_uuids_hash_dict[pair[0]],
                        "classifier_id": self.classifier_id,
                        "score": pair[1],
                    })
                futures.append(executor.submit(request.http_add_classifier_score, score_data=score_data_list))

            for _ in tqdm(as_completed(futures), total=len(hash_score_pairs)//self.batch_size + 1):
                continue

    def get_classifier_id_and_name(self, classifier_file_path):
        for classifier in self.classifier_id_list:
            if classifier["model_path"] == classifier_file_path:
                return classifier["classifier_id"], classifier["classifier_name"]
        return -1, ""

def run_image_scorer(minio_client,
                     dataset_names,
                     batch_size,
                     increment,
                     database):
    start_time = time.time()
    # remove
    print("run_image_scorer")

    scorer = ImageScorer(minio_client=minio_client,
                         dataset_names=dataset_names,
                         batch_size=batch_size)

    classifier_model_list = request.http_get_classifier_model_list()
    
    # load classifier_model
    is_loaded = False
    for classifier_model in classifier_model_list:
        try:
            is_loaded = scorer.load_model(classifier_model_info=classifier_model)
        except Exception as e:
            print("Failed loading model, {}".format(classifier_model["model_path"]), e)
        if is_loaded:
            break
    if not is_loaded:
        return
    
    start_time = time.time()
    if database == 'minio':
        paths = scorer.get_paths()
    elif database == 'mongodb':
        paths = scorer.get_paths_from_mongodb()
    else:
        return None
    end_time = time.time()
    time_elapsed_to_get_paths = end_time - start_time
    print("Getting paths of clip vector is done! Time elapsed = ", time_elapsed_to_get_paths, ", number of paths = ", len(paths))

    # list of clip ve
    list_load_clip_vector_count = [1024]
    for i in range(increment, 100000, increment):
        list_load_clip_vector_count.append(i)

    print("Getting paths of clip vector is done!")
    monitor_loading = []
    monitor_scoring = []
    monitor_uploading = []

    for clip_vector_count in list_load_clip_vector_count:
        paths_batch = paths[:clip_vector_count]

        # Record the runtime to load clip vectors
        print("loading clip vectors")
        start_time = time.time()
        features_data, image_paths = scorer.get_all_feature_pairs(paths_batch)
        end_time = time.time()
        time_elapsed = end_time - start_time
        monitor_loading.append(time_elapsed)

        # Record the runtime to score clip vectors
        print("scoring clip vectors")
        start_time = time.time()
        hash_score_pairs, image_paths, job_uuids_hash_dict = scorer.get_scores(features_data, image_paths)
        end_time = time.time()
        time_elapsed = end_time - start_time
        monitor_scoring.append(time_elapsed)

        # Record the runtime to upload scores
        start_time = time.time()
        scorer.upload_scores(hash_score_pairs, job_uuids_hash_dict)
        end_time = time.time()
        time_elapsed = end_time - start_time
        monitor_uploading.append(time_elapsed)


        print("{} clip vectors: Total Time elapsed: {}s".format(clip_vector_count, format(time_elapsed, ".2f")))

    # graph
    # set title of graph
    plt.title("Monitoring image classifier score")
    fig_report_text = ("Batch_size = {}\n"
                       "Elapsed time getting paths from {}:\n    {}s"
                       .format(batch_size, database, format(time_elapsed_to_get_paths, ".2f")))
    #info text about the model

    plt.figure(figsize=(10, 5))
    plt.figtext(0.02,0.7, fig_report_text, bbox=dict(facecolor="white", alpha=0.5, pad=2))
    
    # plot loading time
    plt.plot(list_load_clip_vector_count, 
             monitor_loading, 
             label="Loading time", 
             marker='o',
             markersize=3)
    
    # plot scoring time
    plt.plot(list_load_clip_vector_count, 
             monitor_scoring,
             label="Scoring time",
             marker='o',
             markersize=3)
    
    # plot uploading time
    plt.plot(list_load_clip_vector_count, 
             monitor_uploading, 
             label="Uploading time",
             marker='o',
             markersize=3)

    plt.xlabel("Count of clip vectors")
    plt.ylabel("Time(s)")
    plt.legend()

    plt.subplots_adjust(left=0.3)

    plt.savefig("output/{}_monitor_image_classifier_scorer.png".format(datetime.now()), format="png")
    print("Saved graph successfully!")
    
def parse_args():
    parser = argparse.ArgumentParser(description="Embedding Scorer")
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--batch-size', required=False, default=100, type=int, help='Name of the dataset for embeddings')
    parser.add_argument('--increment', required=False, default=10000, type=int, help='Name of the dataset for embeddings')
    parser.add_argument('--database', type=str, default='minio', help='Name of the database to get paths')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)

    dataset_names = request.http_get_dataset_names()
    print("dataset names=", dataset_names)
    # try:
    run_image_scorer(minio_client, dataset_names, args.batch_size, args.increment, args.database)
    # except Exception as e:
        # print("Error running image scorer for {}: {}".format(dataset_names, e))


if __name__ == "__main__":
    main()
