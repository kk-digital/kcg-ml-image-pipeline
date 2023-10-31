import glob
import os
import sys
import csv
import io
import argparse
import numpy as np
import torch
import msgpack
import matplotlib.pyplot as plt  
base_directory = "./"
sys.path.insert(0, base_directory)
from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from utility.minio import cmd

class EmbeddingScorer:
    def __init__(self,
                 minio_addr=None,
                 minio_access_key=None,
                 minio_secret_key=None,
                 dataset_name="default_dataset"):
        
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.minio_client = cmd.get_minio_client(minio_access_key=self.minio_access_key,
                                                 minio_secret_key=self.minio_secret_key,
                                                 minio_ip_addr=minio_addr)
        self.embedding_score_model = None
        self.embedding_score_model_positive = None
        self.embedding_score_model_negative = None
        self.dataset=dataset_name
        self.input = os.path.join("datasets", dataset_name)  # Construct the path dynamically using the provided dataset name


    def load_model(self, model_path, input_size):
        embedding_model = ABRankingELMModel(input_size)
        model_files=cmd.get_list_of_objects_with_prefix(self.minio_client, 'datasets', model_path)
        most_recent_model = None
 
        for model_file in model_files:
            file_extension = os.path.splitext(model_file)[1]
            if file_extension == ".pth":
                most_recent_model = model_file
        if most_recent_model:
            model_file_data =cmd.get_file_from_minio(self.minio_client, 'datasets', most_recent_model)
        else:
            print("No .pth files found in the list.")
            return
        
        # Create a BytesIO object and write the downloaded content into it
        byte_buffer = io.BytesIO()
        for data in model_file_data.stream(amt=8192):
            byte_buffer.write(data)
        # Reset the buffer's position to the beginning
        byte_buffer.seek(0)
        embedding_model.load(byte_buffer)
        return embedding_model

    def load_all_models(self):
        input_path =self.dataset + "/models/ranking/"

        self.embedding_score_model = self.load_model(os.path.join(input_path, "ab_ranking_elm_v1"), 768*2)
        self.embedding_score_model_negative = self.load_model(os.path.join(input_path, "ab_ranking_elm_v1_positive_only"), 768)
        self.embedding_score_model_positive = self.load_model(os.path.join(input_path, "ab_ranking_elm_v1_negative_only"), 768)
    def get_scores(self):
        msgpack_files = glob.glob(os.path.join(self.input, "**/*_embedding.msgpack"), recursive=True)
        positive_scores = []
        negative_scores = []
        normal_scores = []
        print('making predictions..........')
        for msgpack_path in msgpack_files:
            with open(msgpack_path, 'rb') as file:
                data_bytes = file.read()
            # Load the data from the bytes using msgpack
            data = msgpack.unpackb(data_bytes, raw=False)

            positive_embedding= list(data['positive_embedding'].values())
            positive_embedding_array = torch.tensor(np.array(positive_embedding)).float()
            negative_embedding= list(data['negative_embedding'].values())
            negative_embedding_array =torch.tensor(np.array(negative_embedding)).float()

            positive_scores.append(self.embedding_score_model_positive.predict_positive_or_negative_only(positive_embedding_array))
            negative_scores.append(self.embedding_score_model_negative.predict_positive_or_negative_only(negative_embedding_array))
            normal_scores.append(self.embedding_score_model.predict(positive_embedding_array, negative_embedding_array))
        
        # Normalize the positive and negative scores
        normalized_positive_scores = normalize_scores(positive_scores)
        normalized_negative_scores = normalize_scores(negative_scores)
        normilozed_normal_score = normalize_scores(normal_scores)
        # Merge the vectors into a list of dictionaries
        scores = []
        for pos, neg, score in zip(normalized_positive_scores, normalized_negative_scores, normilozed_normal_score):
            scores.append({'positive': pos, 'negative': neg, 'score': score})
        # Save scores to CSV
        with open('scores.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(["Image Path", "Image Hash", "Positive Score", "Negative Score", "Combined Score"])
            for msgpack_path, score in zip(msgpack_files, scores):
                writer.writerow([msgpack_path, os.path.basename(msgpack_path), score['positive'], score['negative'], score['score']])
        print('Scores saved to scores.csv')
        
        return scores
    
    def generate_graphs(self):
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.hist(self.positive_scores, bins=30, color='green')
        plt.title("Positive Scores")
        plt.subplot(1, 3, 2)
        plt.hist(self.negative_scores, bins=30, color='red')
        plt.title("Negative Scores")
        plt.subplot(1, 3, 3)
        plt.hist(self.normal_scores, bins=30, color='blue')
        plt.title("Normal Scores")
        plt.tight_layout()
        plt.savefig("score_distributions.png")
        plt.show()

def normalize_scores(scores):  # fixed indentation
    # Assuming min-max normalization
    min_val = min(scores)
    max_val = max(scores)
    return [(x - min_val) / (max_val - min_val) for x in scores]

def parse_args():
    parser = argparse.ArgumentParser(description="Embedding Scorer")
    parser.add_argument('--minio-addr', required=True, help='Minio server address')
    parser.add_argument('--minio-access-key', required=True, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=True, help='Minio secret key')
    parser.add_argument('--dataset-name', required=True, help='Name of the dataset for embeddings')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    scorer = EmbeddingScorer(minio_addr=args.minio_addr,
                             minio_access_key=args.minio_access_key,
                             minio_secret_key=args.minio_secret_key,
                             dataset_name=args.dataset_name)
    
    scorer.load_all_models()
    scorer.get_scores()
    scorer.generate_graphs()