import os
import sys
import io
import argparse
import numpy as np
import torch
import time
import msgpack
from io import BytesIO
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

base_directory = "./"
sys.path.insert(0, base_directory)

from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from training_worker.ab_ranking.model.ab_ranking_linear import ABRankingModel as ABRankingLinearModel
from training_worker.http import request
from utility.minio import cmd


def determine_model_input_type_size(model_filename):
    if "embedding-positive" in model_filename:
        return "embedding-positive", 768
    elif "embedding-negative" in model_filename:
        return "embedding-negative", 768
    elif "embedding" in model_filename:
        return "embedding", 2 * 768
    elif "clip" in model_filename:
        return "clip", 768
    else:
        raise ValueError("Unknown model type in the filename.")


class ImageScorer:
    def __init__(self,
                 minio_addr=None,
                 minio_access_key=None,
                 minio_secret_key=None,
                 dataset_name="characters",
                 model_name=""):
        self.minio_client = cmd.get_minio_client(minio_access_key=minio_access_key,
                                                 minio_secret_key=minio_secret_key,
                                                 minio_ip_addr=minio_addr)
        self.model = None
        self.dataset = dataset_name
        self.model_name = model_name
        self.model_input_type, self.input_size = determine_model_input_type_size(model_name)
        self.model_id = None
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = torch.device(device)

        print("Dataset=", self.dataset)
        print("Model=", self.model_name)
        print("Input type=", self.model_input_type)
        print("device=", self.device)

    def load_model(self):
        model_path = os.path.join(self.dataset, "models", "ranking", self.model_name)

        if "elm" in self.model_name:
            model = ABRankingELMModel(self.input_size)
        else:
            model = ABRankingLinearModel(self.input_size)

        model_file_data = cmd.get_file_from_minio(self.minio_client, 'datasets', model_path)
        if not model_file_data:
            raise Exception("No .pth file found at path: ", model_path)

        byte_buffer = io.BytesIO()
        for data in model_file_data.stream(amt=8192):
            byte_buffer.write(data)
        byte_buffer.seek(0)
        model.load(byte_buffer)

        # assign
        self.model = model
        self.model.model = model.model.to(self.device)

        # get model id
        self.model_id = request.http_get_model_id(self.model.model_hash)
        print("model_hash=", self.model.model_hash)
        print("model_id=", self.model_id)

    def get_paths(self):
        all_objects = cmd.get_list_of_objects_with_prefix(self.minio_client, 'datasets', self.dataset)

        # Depending on the model type, choose the appropriate msgpack files
        file_suffix = "_clip.msgpack" if self.model_input_type == "clip" else "_embedding.msgpack"

        # Filter the objects to get only those that end with the chosen suffix
        type_paths = [obj for obj in all_objects if obj.endswith(file_suffix)]

        print("Total paths found=", len(type_paths))

        return type_paths

    def get_feature_pair(self, path, index):
        # Updated bucket name to 'datasets'
        msgpack_data = cmd.get_file_from_minio(self.minio_client, 'datasets', path)
        if not msgpack_data:
            print(f"No msgpack file found at path: {path}")
            return None, None, None, index

        data = msgpack.unpackb(msgpack_data.data)

        if self.model_input_type == "embedding":
            positive_embedding = list(data['positive_embedding'].values())
            positive_embedding_array = torch.tensor(np.array(positive_embedding)).float()

            negative_embedding = list(data['negative_embedding'].values())
            negative_embedding_array = torch.tensor(np.array(negative_embedding)).float()

            image_hash = data['file_hash']

            return image_hash, positive_embedding_array, negative_embedding_array, index

        elif self.model_input_type == "embedding-positive":
            positive_embedding = list(data['positive_embedding'].values())
            positive_embedding_array = torch.tensor(np.array(positive_embedding)).float()
            image_hash = data['file_hash']

            return image_hash, positive_embedding_array, None, index

        elif self.model_input_type == "embedding-negative":
            negative_embedding = list(data['negative_embedding'].values())
            negative_embedding_array = torch.tensor(np.array(negative_embedding)).float()
            image_hash = data['file_hash']

            return image_hash, negative_embedding_array, None, index

        elif self.model_input_type == "clip":
            clip_feature = data['clip-feature-vector']
            clip_feature_vector = torch.tensor(np.array(clip_feature)).float()

            # clip image hash isn't in clip.msgpack so get it from _data.msgpack
            data_msgpack = cmd.get_file_from_minio(self.minio_client, 'datasets',
                                                   path.replace("clip.msgpack", "data.msgpack"))
            if not data_msgpack:
                print("No msgpack file found at path: {}".format(path.replace("clip.msgpack", "data.msgpack")))
                return None

            data = msgpack.unpackb(data_msgpack.data)
            image_hash = data['file_hash']

            return image_hash, clip_feature_vector, None, index

        return None, None, None, index

    def get_all_feature_pairs(self, msgpack_paths):
        print('Getting dataset features...')

        features_data = [None] * len(msgpack_paths)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            count = 0
            for path in msgpack_paths:
                futures.append(executor.submit(self.get_feature_pair, path=path, index=count))
                count += 1

            for future in tqdm(as_completed(futures), total=len(msgpack_paths)):
                image_hash, first_feature, second_feature, index = future.result()
                features_data[index] = (image_hash, first_feature, second_feature)

        return features_data

    def get_scores(self, msgpack_paths):
        hash_score_pairs = []

        features_data = self.get_all_feature_pairs(msgpack_paths)
        print('Predicting dataset scores...')
        with torch.no_grad():
            for data in tqdm(features_data):
                if self.model_input_type == "embedding":
                    positive_embedding_array = data[1].to(self.device)
                    negative_embedding_array = data[2].to(self.device)
                    image_hash = data[0]
                    score = self.model.predict(positive_embedding_array, negative_embedding_array)

                elif self.model_input_type == "embedding-positive":
                    positive_embedding_array = data[1].to(self.device)
                    image_hash = data[0]
                    score = self.model.predict_positive_or_negative_only(positive_embedding_array)

                elif self.model_input_type == "embedding-negative":
                    negative_embedding_array = data[1].to(self.device)
                    image_hash = data[0]
                    score = self.model.predict_positive_or_negative_only(negative_embedding_array)

                elif self.model_input_type == "clip":
                    clip_feature_vector = data[1].to(self.device)
                    image_hash = data[0]
                    score = self.model.predict_clip(clip_feature_vector)

                hash_score_pairs.append((image_hash, score.item()))

        # # Merge the vectors into a list of dictionaries
        # # Save scores to CSV
        # with open('scores.csv', 'w') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(["Model Filename", "Model Hash", "Image Path", "Image Hash", "Score"])
        #     for msgpack_path, score in zip(msgpack_objects, scores):
        #         writer.writerow([msgpack_path, os.path.basename(msgpack_path), score['positive'], score['negative'], score['score']])
        # print('Scores saved to scores.csv')

        return hash_score_pairs

    def upload_scores(self, hash_score_pairs):
        print("Uploading scores to mongodb...")
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = []
            for pair in hash_score_pairs:
                # upload score
                score_data = {
                    "model_id": self.model_id,
                    "image_hash": pair[0],
                    "score": pair[1],
                }

                futures.append(executor.submit(request.http_add_score, score_data=score_data))

            for _ in tqdm(as_completed(futures), total=len(hash_score_pairs)):
                continue

    def generate_graphs(self, hash_score_pairs):
        # Initialize all graphs/subplots
        plt.figure(figsize=(22, 20))
        figure_shape = (1, 1)
        score_graph = plt.subplot2grid(figure_shape, (0, 0), rowspan=1, colspan=1)

        scores = [i[1] for i in hash_score_pairs]
        x_axis_values = [i for i in range(len(hash_score_pairs))]
        score_graph.scatter(x_axis_values, scores,
                            label="Image Scores over time",
                            c="#281ad9", s=15)

        score_graph.set_xlabel("Time")
        score_graph.set_ylabel("Score")
        score_graph.set_title("Score vs Time")
        score_graph.legend()
        score_graph.autoscale(enable=True, axis='y')

        # Save figure
        # plt.subplots_adjust(left=0.15, hspace=0.5)
        # plt.savefig("./output/{}-{}-scores.jpg".format(self.model_name.replace(".pth", ""), self.dataset))
        plt.show()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # upload the graph report
        graph_output = os.path.join(self.dataset, "models", "ranking",
                                    self.model_name.replace(".pth", "-dataset-graph.png"))
        cmd.upload_data(self.minio_client, 'datasets', graph_output, buf)


def parse_args():
    parser = argparse.ArgumentParser(description="Embedding Scorer")
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--dataset-name', required=True, help='Name of the dataset for embeddings')
    parser.add_argument('--model-filename', required=True, help='Filename of the main model (e.g., "XXX.pth")')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    scorer = ImageScorer(minio_addr=args.minio_addr,
                         minio_access_key=args.minio_access_key,
                         minio_secret_key=args.minio_secret_key,
                         dataset_name=args.dataset_name,
                         model_name=args.model_filename)
    start_time = time.time()

    scorer.load_model()
    paths = scorer.get_paths()
    hash_score_pairs = scorer.get_scores(paths)
    scorer.generate_graphs(hash_score_pairs)
    scorer.upload_scores(hash_score_pairs)

    time_elapsed = time.time() - start_time
    print("Total Time elapsed: {0}s".format(format(time_elapsed, ".2f")))

if __name__ == "__main__":
    main()
