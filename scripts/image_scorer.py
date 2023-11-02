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



MINIO_ADDRESS = "192.168.3.5:9000"
access_key = "3lUCPCfLMgQoxrYaxgoz"
secret_key = "MXszqU6KFV6X95Lo5jhMeuu5Xm85R79YImgI3Xmp"

device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        self.dataset = dataset_name
        self.input = os.path.join("datasets", dataset_name)

    def load_model(self, model_filename, input_size):
        model_path = os.path.join(self.dataset, "models", "ranking", model_filename)
        embedding_model = ABRankingELMModel(input_size)

        model_file_data = cmd.get_file_from_minio(self.minio_client, 'datasets', model_path)
        if not model_file_data:
            print("No .pth file found at path: ", model_path)
            return

        byte_buffer = io.BytesIO()
        for data in model_file_data.stream(amt=8192):
            byte_buffer.write(data)
        byte_buffer.seek(0)
        embedding_model.load(byte_buffer)
        return embedding_model


    def load_all_models(self, model_filename, positive_model_filename, negative_model_filename):
        self.embedding_score_model = self.load_model(model_filename, 768 * 2)
        self.embedding_score_model_positive = self.load_model(positive_model_filename, 768)
        self.embedding_score_model_negative = self.load_model(negative_model_filename, 768)

    def get_scores(self):
        # Corrected the prefix to recursively search inside all subdirectories
        prefix = os.path.join(self.dataset, '')
        all_objects = cmd.get_list_of_objects_with_prefix(self.minio_client, 'datasets', prefix)
        
        # Filter the objects to get only those that end with '_embedding.msgpack'
        msgpack_objects = [obj for obj in all_objects if obj.endswith('_embedding.msgpack')]
        
        positive_scores = []
        negative_scores = []
        normal_scores = []
        print('making predictions..........')
        for msgpack_object in msgpack_objects:
            # Updated bucket name to 'datasets'
            msgpack_data = cmd.get_file_from_minio(self.minio_client, 'datasets', msgpack_object)
            if not msgpack_data:
                print(f"No msgpack file found at path: {msgpack_object}")
                continue

            byte_buffer = io.BytesIO()
            for data in msgpack_data.stream(amt=8192):
                byte_buffer.write(data)
            byte_buffer.seek(0)
            data_bytes = byte_buffer.read()
            data = msgpack.unpackb(data_bytes, raw=False)

            positive_embedding = list(data['positive_embedding'].values())
            positive_embedding_array = torch.tensor(np.array(positive_embedding)).float().to(device)
            negative_embedding = list(data['negative_embedding'].values())
            negative_embedding_array = torch.tensor(np.array(negative_embedding)).float().to(device)

            with torch.no_grad():
                positive_scores.append(self.embedding_score_model_positive.predict_positive_or_negative_only(positive_embedding_array))
                negative_scores.append(self.embedding_score_model_negative.predict_positive_or_negative_only(negative_embedding_array))
                normal_scores.append(self.embedding_score_model.predict(positive_embedding_array, negative_embedding_array))
        
        # Normalize the positive and negative scores
        self.normalized_positive_scores = normalize_scores([score.cpu() for score in positive_scores])
        self.normalized_negative_scores = normalize_scores([score.cpu() for score in negative_scores])
        self.normalized_score = normalize_scores([score.cpu() for score in normal_scores])


        # Merge the vectors into a list of dictionaries
        scores = []
        for pos, neg, score in zip(self.normalized_positive_scores, self.normalized_negative_scores, self.normalized_score):
            scores.append({'positive': pos, 'negative': neg, 'score': score})
        # Save scores to CSV
        with open('scores.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(["Image Path", "Image Hash", "Positive Embedding Score", "Negative Embedding Score", "Embedding Score"])
            for msgpack_path, score in zip(msgpack_objects, scores):
                writer.writerow([msgpack_path, os.path.basename(msgpack_path), score['positive'], score['negative'], score['score']])
        print('Scores saved to scores.csv')
        
        return scores

    
    def generate_graphs(self):

        # Helper function to convert possible tensors in a list to numpy
        def to_numpy(data):
            if isinstance(data, torch.Tensor):
                return data.cpu().numpy()
            return np.array(data)

        # Convert lists to numpy arrays using the helper function
        positive_scores_np = to_numpy(self.normalized_positive_scores)
        negative_scores_np = to_numpy(self.normalized_negative_scores)
        normal_scores_np = to_numpy(self.normalized_score)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.hist(positive_scores_np, bins=30, color='green')
        plt.title(f"({self.positive_model_filename.split('/')[-1]})") 
        plt.subplot(1, 3, 2)
        plt.hist(negative_scores_np, bins=30, color='red')
        plt.title(f"({self.negative_model_filename.split('/')[-1]})") 
        plt.subplot(1, 3, 3)
        plt.hist(normal_scores_np, bins=30, color='blue')
        plt.title(f"({self.model_filename.split('/')[-1]})") 
        plt.tight_layout()
        plt.savefig("score_distributions.png")




def normalize_scores(scores):  # fixed indentation
    # Assuming min-max normalization
    min_val = min(scores)
    max_val = max(scores)
    return [(x - min_val) / (max_val - min_val) for x in scores]

def parse_args():
    parser = argparse.ArgumentParser(description="Embedding Scorer")
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default=MINIO_ADDRESS)
    parser.add_argument('--minio-access-key', required=False, help='Minio access key', default=access_key)
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key', default=secret_key)
    parser.add_argument('--dataset-name', required=True, help='Name of the dataset for embeddings')
    parser.add_argument('--model-filename', required=True, help='Filename of the main model (e.g., "XXX.pth")')
    parser.add_argument('--positive-model-filename', required=True, help='Filename of the positive model')
    parser.add_argument('--negative-model-filename', required=True, help='Filename of the negative model')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    scorer = EmbeddingScorer(minio_addr=args.minio_addr,
                             minio_access_key=args.minio_access_key,
                             minio_secret_key=args.minio_secret_key,
                             dataset_name=args.dataset_name)
    
    scorer.load_all_models(args.model_filename, args.positive_model_filename, args.negative_model_filename)
    scorer.get_scores()
    scorer.generate_graphs()

if __name__ == "__main__":
    main()