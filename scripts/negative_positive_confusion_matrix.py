import glob
import io
import os
import sys
import argparse
from matplotlib import pyplot as plt
import msgpack
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sb
base_directory = "./"
sys.path.insert(0, base_directory)

from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from utility.minio import cmd
 

class EmbeddingConfusionMatrix:
    def __init__(self,
                 minio_addr=None,
                 minio_access_key=None,
                 minio_secret_key=None,
                 input="input"):

        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.minio_client = cmd.get_minio_client(minio_access_key=self.minio_access_key,
                                                 minio_secret_key=self.minio_secret_key,
                                                 minio_ip_addr=minio_addr)
        self.embedding_score_model=None
        self.embedding_score_model_positive= None
        self.embedding_score_model_negative= None
        self.input=input

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
        input_path="environmental/models/ranking/"

        self.embedding_score_model= self.load_model(input_path+ "ab_ranking_elm_v1", 768*2)
        self.embedding_score_model_negative= self.load_model(input_path + "ab_ranking_elm_v1_positive_only", 768)
        self.embedding_score_model_positive= self.load_model(input_path + "ab_ranking_elm_v1_negative_only", 768)
    
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

        # Merge the vectors into a list of dictionaries
        scores=[]
        for pos, neg, score in zip(normalized_positive_scores, normalized_negative_scores, normal_scores):
            scores.append({'positive': pos, 'negative': neg, 'score': score})
        
        return scores
        
    
    def show_confusion_matrix(self):
        print('constructing confusion matrix.......')
        # get predicted scores
        scores=self.get_scores()
        # create a confusion matrix of average scores
        cm=get_confusion_matrix(scores)

        # Create a heatmap of the confusion matrix
        plt.figure(figsize=(15, 12))
        sb.heatmap(cm, annot=True ,cbar=True)
        positions = np.linspace(0.5, 10.5, 11) 
        labels = [str(i / 10.0) for i in range(11)]
        # Set the class labels as ticks on both axes
        plt.xticks(positions, labels)
        plt.yticks(positions, labels)
        plt.title("Confusion Matrix")
        plt.xlabel('Negative')
        plt.ylabel('Positive')

        plt.gca().invert_yaxis()
        plt.show()

        print('done')
    
    
def get_confusion_matrix(scores):
    labels = [i / 10 for i in range(11)]
    cm=np.zeros((11, 11))
    for i, negative_value in enumerate(labels):
            for j, positive_val in enumerate(labels):
                cm[i][j]=average_score(negative_value ,positive_val, scores)
    return cm

def average_score(negative_value, positive_value, scores):
    avg=0
    for s in scores:
        if s['positive']== positive_value and s['negative']==negative_value:
            avg+=s['score']
    
    return int(avg/len(scores))

# Normalize the vectors
def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
    rounded_scores = [round(float(score), 1) for score in normalized_scores]
    return rounded_scores  

def parse_args():
    parser = argparse.ArgumentParser()

    #parameters
    parser.add_argument("--input", type=str, default="input",
                        help="The path to dataset directory")
    parser.add_argument("--minio-addr", type=str, default=None,
                        help="The minio server ip address")
    parser.add_argument("--minio-access-key", type=str,
                        help="The minio access key to use so worker can upload files to minio server")
    parser.add_argument("--minio-secret-key", type=str,
                        help="The minio secret key to use so worker can upload files to minio server")

    return parser.parse_args()

def main():
    args = parse_args()

    confusion_matrix = EmbeddingConfusionMatrix(minio_addr=args.minio_addr,
                                               minio_access_key=args.minio_access_key,
                                               minio_secret_key=args.minio_secret_key,
                                               input= args.input)
    
    confusion_matrix.load_all_models()

    confusion_matrix.show_confusion_matrix()



if __name__ == '__main__':
    main()