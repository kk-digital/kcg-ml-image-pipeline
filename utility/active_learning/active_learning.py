import sys
import os
import requests

root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_directory)

from tqdm.auto import tqdm

import argparse
import msgpack

import numpy as np
import pandas as pd

import torch

from sklearn.metrics.pairwise import cosine_similarity

from training_worker.ab_ranking.model.ab_ranking_linear import ABRankingModel
from utility.minio.cmd import connect_to_minio_client
from utility.active_learning.pairs import get_candidate_pairs_by_score, get_candidate_pairs_within_category

API_URL = "http://123.176.98.90:8764"

class ActiveLearningPipeline:

    def __init__(self, minio_addr: str, minio_access_key: str, minio_secret_key: str, 
                 scoring_model_path: str, pca_model_path: str, kmeans_model_path: str, 
                 bins: int, bin_type: str , cluster_type: str, pairs: int,
                 csv_path: str, min_sigma_score: float, min_variance: float):

        self.bins=bins
        self.bin_type=bin_type
        self.min_sigma_score=min_sigma_score
        self.min_variance=min_variance
        self.pairs=pairs
        
        self.image_list=self.filter_images(csv_path) 
        self.connect_to_minio_client(minio_addr, minio_access_key, minio_secret_key)
        self.load_models(scoring_model_path, pca_model_path, kmeans_model_path, cluster_type)

    def filter_images(self, csv_path):
        df= pd.read_csv(csv_path)
        
        # Assert that necessary columns are in the DataFrame
        assert 'task_uuid' in df.columns, "Column 'task_uuid' not found in csv file"
        assert 'file_path' in df.columns, "Column 'file_path' not found in csv file"
        assert 'variance' in df.columns, "Column 'variance' not found in csv file"
        assert 'sigma_score' in df.columns, "Column 'sigma_score' not found in csv file"
        
        # Filter the DataFrame based on minimum sigma_score and variance
        filtered_df = df[(df['sigma_score'] >= self.min_sigma_score) & (df['variance'] >= self.min_variance)]

        return filtered_df

    def connect_to_minio_client(self, minio_addr: str, minio_access_key: str, minio_secret_key: str):

        self.client = connect_to_minio_client(
            minio_addr, 
            minio_access_key, 
            minio_secret_key
        )

        self.bucket_name = 'datasets'
    
    def load_models(self, scoring_model_path: str, pca_model_path: str, kmeans_model_path: str, cluster_type: str):
        
        model = ABRankingModel(768)
        
        model.load_safetensors(open(scoring_model_path, 'rb'))

        self.scoring_model = model.model.linear

        self.scoring_model_mean = float(model.mean)
        self.scoring_model_standard_deviation = float(model.standard_deviation)

        npz = np.load(pca_model_path)

        self.pca_components = npz['components']
        
        self.n_pca_components = 24

        npz = np.load(kmeans_model_path)

        self.kmeans_cluster_centers = npz[f'cluster_centers_{cluster_type}']
    
    def load_vision_embs(self, file_paths):
    
        vision_embs = list()
        
        for file_path in tqdm(file_paths, leave=False):
    
            object_name = file_path.replace(f'{self.bucket_name}/', '')
            object_name = os.path.splitext(object_name.split('_')[0])[0]
            object_name = f'{object_name}_clip.msgpack'
    
            try:
                
                data = self.client.get_object(self.bucket_name, object_name).data
                decoded_data = msgpack.unpackb(data)
                vision_embs.append(np.array(decoded_data['clip-feature-vector']).astype('float32'))
    
            except KeyboardInterrupt:
                return
            except:
                vision_embs.append(np.zeros((1, 768), dtype='float32'))

        vision_embs = np.concatenate(vision_embs, axis=0)
        
        return vision_embs

    def get_scores(self, vision_embs: np.ndarray):

        with torch.no_grad():
            scores = self.scoring_model(torch.tensor(vision_embs).cuda())
        scores = scores[..., 0].detach().cpu().numpy()

        sigma_scores = (scores - self.scoring_model_mean) / self.scoring_model_standard_deviation

        return sigma_scores

    def get_cluster_ids(self, vision_embs: np.ndarray):

        z = np.dot(vision_embs, self.pca_components.T[:, :self.n_pca_components])

        m = cosine_similarity(z, self.kmeans_cluster_centers)
        cluster_ids = np.argmax(m, axis=1)

        return cluster_ids
    
    def get_image_pairs(self):
        vision_embs = self.load_vision_embs(file_paths=self.image_list['file_path'].values)
        image_clip_sigma_score = self.get_scores(vision_embs)
        cluster_ids = self.get_cluster_ids(vision_embs)
                    
        # pairing
        sigma_score_pairs = get_candidate_pairs_by_score(
            job_uuids=self.image_list['task_uuid'].values,
            scores = image_clip_sigma_score, 
            max_pairs = self.pairs, 
            n_bins = self.bins, 
            use_quantiles = (self.bin_type == 'quantile')
        )

        cluster_pairs = get_candidate_pairs_within_category(
            job_uuids=self.image_list['task_uuid'].values,
            categories = cluster_ids, 
            max_pairs = self.pairs
        )

        merged_list= sigma_score_pairs.copy()
        # merge pairs by sigma score and by cluster
        for pair in cluster_pairs:
            if pair not in merged_list:
                merged_list.append(pair)
            
        return merged_list
    
    def upload_pair_of_jsons_from_csv(pair_list, policy):
        
        for pair in tqdm(pair_list):
            job_uuid_1= pair[0]
            job_uuid_2= pair[1]

            endpoint_url = f"{API_URL}/ranking-queue/add-image-pair-to-queue?job_uuid_1={job_uuid_1}&job_uuid_2={job_uuid_2}&policy={policy}"
            response = requests.post(endpoint_url)

            if response.status_code == 200:
                print(f"Successfully processed job pair: UUID1: {job_uuid_1}, UUID2: {job_uuid_2}")
            else:
                print(f"Failed to process job pair: UUID1: {job_uuid_1}, UUID2: {job_uuid_2}. Response: {response.status_code} - {response.text}")

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--csv-path", type=str,
                        help="The path to csv file")
    parser.add_argument("--scoring-model-path", type=str,
                        help="The path to clip scoring model safetensors file")
    parser.add_argument("--pca-model-path", type=str,
                        help="The path to PCA model npz file")
    parser.add_argument("--kmeans-model-path", type=str,
                        help="The path to KMeans model npz file")
    parser.add_argument("--pairs", type=int, default=1000,
                        help="The number of pairs")
    parser.add_argument("--bins", type=int, default=10,
                        help="The number of bins")
    parser.add_argument("--bin-type", type=str, default='quantile',
                        help="The binning method: fixed-range or quantile")
    parser.add_argument("--cluster-type", type=str, 
                        help="type of cluster (48, 1024, 4096)", default="48")
    parser.add_argument("--min-sigma-score", type=float, 
                        help="minimum sigma score when filtering images", default=1)
    parser.add_argument("--min-variance", type=float, 
                        help="minimum sigma score when filtering images", default=0.1)
    
    parser.add_argument("--minio-addr", type=str, default=None,
                        help="The minio server ip address")
    parser.add_argument("--minio-access-key", type=str,
                        help="The minio access key to use so worker can upload files to minio server")
    parser.add_argument("--minio-secret-key", type=str,
                        help="The minio secret key to use so worker can upload files to minio server")

    return parser.parse_args()

def main():
    
    args = parse_args()

    pipeline = ActiveLearningPipeline(
        minio_addr=args.minio_addr,
        minio_access_key=args.minio_access_key,
        minio_secret_key=args.minio_secret_key,
        scoring_model_path=args.scoring_model_path,
        pca_model_path=args.pca_model_path,
        kmeans_model_path=args.kmeans_model_path,
        bin_type=args.bin_type,
        bins=args.bins,
        cluster_type=args.cluster_type,
        csv_path=args.csv_path
    )

    # get list of pairs
    print(pipeline.get_image_pairs())
    

if __name__ == '__main__':
    main()