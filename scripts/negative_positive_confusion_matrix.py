import glob
import io
import os
import sys
import argparse
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import msgpack
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sb
base_directory = "./"
sys.path.insert(0, base_directory)

from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from utility.minio import cmd
 

#

class EmbeddingConfusionMatrix:
    def __init__(self,
                 minio_addr=None,
                 minio_access_key=None,
                 minio_secret_key=None,
                 input="input",
                 bins=10,
                 annot=False):

        #connect to minIO server
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.minio_client = cmd.get_minio_client(minio_access_key=self.minio_access_key,
                                                 minio_secret_key=self.minio_secret_key,
                                                 minio_ip_addr=minio_addr)
        #load embedding models
        self.load_all_models()
        #input path
        self.input=input
        #number of bins
        self.bins=bins
        #annotation for confusion matrix
        self.annot=annot
        # get predicted scores
        self.scores=self.get_scores()
        # calculate confusion matrix
        self.confusion_matrix=get_confusion_matrix(self.scores, self.bins)

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
        
        print(most_recent_model)

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

        self.embedding_score_model= self.load_model(input_path+ "ab_ranking_elm_v1/", 768*2)
        self.embedding_score_model_negative= self.load_model(input_path + "ab_ranking_elm_v1_negative_only", 768)
        self.embedding_score_model_positive= self.load_model(input_path + "ab_ranking_elm_v1_positive_only", 768)
    
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
        normalized_positive_scores = normalize_scores(positive_scores, self.bins)
        normalized_negative_scores = normalize_scores(negative_scores, self.bins)
        normalized_scores=percentile_normalization(normal_scores)

        # Merge the vectors into a list of dictionaries
        scores=[]
        for pos, neg, score in zip(normalized_positive_scores, normalized_negative_scores, normalized_scores):
            scores.append({'positive': pos, 'negative': neg, 'score': score})
        
        return scores
        
    
    def show_confusion_matrix(self):
        print('constructing confusion matrix.......')
        
        # Generate a custom colormap representing brightness
        colors = [(1, 1, 1), (1, 0, 0)]  # White to Red
        custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)

        
        # Create a heatmap of the confusion matrix
        plt.figure(figsize=(15, 12))
        sb.heatmap(self.confusion_matrix,cbar=True, annot=self.annot, cmap=custom_cmap)
        positions = np.linspace(0.5, self.bins-0.5, self.bins) 
        labels = [str(i+1) for i in range(self.bins)]
        # Set the class labels as ticks on both axes
        plt.xticks(positions,labels)
        plt.yticks(positions, labels)
        plt.title("mean quality score vs negative/positive score")
        plt.xlabel('Negative')
        plt.ylabel('Positive')

        plt.gca().invert_yaxis()
        plt.show()

        print('done')
    
    def bin_histogram(self, negative_bin, positive_bin):
        distribution=[]
        for s in self.scores:
            if s['positive']== positive_bin and s['negative']==negative_bin:
                distribution.append(s['score'])

        # Create a histogram
        plt.figure(figsize=(10, 5))
        plt.hist(distribution, bins=self.bins, alpha=0.7, color='b', label='Data')

        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Percentile Distribution Histogram for bin ({negative_bin},{positive_bin})')
        plt.legend(loc='upper right')
        plt.grid(True)

        plt.show()
    
    
def get_confusion_matrix(scores, bins):
    values = [i+1 for i in range(bins)]
    cm=np.zeros((bins, bins))
    for i, negative_value in enumerate(values):
            for j, positive_val in enumerate(values):
                cm[j][i]=average_embedding_score(negative_value ,positive_val, scores)
    return cm

# def average_score(negative_value, positive_value, scores):
#     sum=0
#     for s in scores:
#         if s['positive']== positive_value and s['negative']==negative_value:
#             sum+=1
    
#     return sum/len(scores)

def average_embedding_score(negative_value, positive_value, scores):
    sum=0
    for s in scores:
        if s['positive']== positive_value and s['negative']==negative_value:
            sum+=s['score']
    
    return sum/len(scores)

# Normalize the vectors
def normalize_scores(scores, bins):
    vector_size=len(scores)
    sorted_vector=np.argsort(scores)
    normalized_scores = [int((order * bins) / vector_size)+ 1 for order in sorted_vector]
    return normalized_scores

# percentile normalization
def percentile_normalization(scores):
    vector_size=len(scores)
    sorted_vector=np.argsort(scores)
    normalized_scores = [order / vector_size for order in sorted_vector]

    return normalized_scores

def parse_args():
    parser = argparse.ArgumentParser()

    #parameters
    parser.add_argument("--annot", type=bool, default=False,
                        help="if matrix cells are annotated or not")
    parser.add_argument("--bins", type=int, default=10,
                        help="Number of bins in matrix")
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
                                               input= args.input,
                                               bins=args.bins,
                                               annot=args.annot)

    confusion_matrix.show_confusion_matrix()

    # show histogram for a specific bin (x,y)
    #confusion_matrix.bin_histogram(negative_bin=10, positive_bin=10)




if __name__ == '__main__':
    main()