from datetime import datetime
import io
import json
import sys
import os
import requests
from tqdm.auto import tqdm
import argparse
import msgpack
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from utility.minio import cmd
from utility.minio.cmd import connect_to_minio_client
from utility.active_learning.pairs import get_candidate_pairs_by_score, get_candidate_pairs_within_category

API_URL = "http://123.176.98.90:8764"

class ActiveLearningPipeline:

    def __init__(self, minio_addr: str, minio_access_key: str, minio_secret_key: str, 
                 pca_model_path: str, kmeans_model_path: str, bins: int, bin_type: str , 
                 pairs: int, min_sigma_score: float, min_variance: float):
 
        self.bins=bins
        self.bin_type=bin_type
        self.min_sigma_score=min_sigma_score
        self.min_variance=min_variance
        self.pairs=pairs

        # get device
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = torch.device(device)
        
        self.connect_to_minio_client(minio_addr, minio_access_key, minio_secret_key)
        self.load_models(pca_model_path, kmeans_model_path)

    def get_latest_jobs(self):
        # Get today's date
        today = datetime.now().strftime('%Y-%m-%d')

        print('Loading image file paths for', today, '..........')
        response = requests.get(f'{API_URL}/queue/image-generation/list-by-date?start_date=2024-01-08&end_date=2024-01-09')
        
        jobs = json.loads(response.content)

        return jobs

    def connect_to_minio_client(self, minio_addr: str, minio_access_key: str, minio_secret_key: str):

        self.client = connect_to_minio_client(
            minio_addr, 
            minio_access_key, 
            minio_secret_key
        )

        self.bucket_name = 'datasets'
    
    # load ensemble elm model for entropy calculation
    def get_ensemble_models(self, dataset):
        input_path = f"{dataset}/models/ranking/"

        model_class = ABRankingELMModel

        # Get all model files
        model_files = cmd.get_list_of_objects_with_prefix(self.client, 'datasets', input_path)

        # Filter relevant model files
        relevant_models = [
            model_file for model_file in model_files
            if model_file.endswith(f"score-elm-v1-clip.safetensors")
        ]

        # Sort the model files by timestamp (assuming the file names include a timestamp)
        relevant_models=np.flip(relevant_models)

        # Load the latest num_models models
        loaded_models = []
        for i in range(min(16, len(relevant_models))):
            most_recent_model = relevant_models[i]

            # Get the model file data
            model_file_data = cmd.get_file_from_minio(self.client, 'datasets', most_recent_model)

            # Create a BytesIO object and write the downloaded content into it
            byte_buffer = io.BytesIO()
            for data in model_file_data.stream(amt=8192):
                byte_buffer.write(data)
            # Reset the buffer's position to the beginning
            byte_buffer.seek(0)

            # Load the model
            embedding_model = model_class(768)
            embedding_model.load_safetensors(byte_buffer)
            embedding_model.model=embedding_model.model.to(self.device)

            loaded_models.append(embedding_model)

        return loaded_models

    def load_models(self, pca_model_path: str, kmeans_model_path: str):

        npz = np.load(pca_model_path)

        self.pca_components = npz['components']
        
        self.n_pca_components = 24

        npz = np.load(kmeans_model_path)
        
        self.kmeans_cluster_centers_48 = npz['cluster_centers_48']
        self.kmeans_cluster_centers_1024 = npz['cluster_centers_1024']
        self.kmeans_cluster_centers_4096 = npz['cluster_centers_4096']
    
    def filter_by_score_and_variance(self, jobs, ensemble_models):
    
        vision_embs = list()
        sigma_scores = list()
        filtered_jobs = list()
        
        for job in tqdm(jobs, leave=False):
            
            file_path= job['file_path']
            # get clip embedding file path from image file path
            object_name = file_path.replace(f'{self.bucket_name}/', '')
            object_name = os.path.splitext(object_name.split('_')[0])[0]
            object_name = f'{object_name}_clip.msgpack'
    
            # get clip embedding    
            data = self.client.get_object(self.bucket_name, object_name).data
            decoded_data = msgpack.unpackb(data)
            embedding= np.array(decoded_data['clip-feature-vector']).astype('float32')

            # calculate score and variance
            mean_score, variance= self.get_variance(embedding, ensemble_models)

            if (mean_score > self.min_sigma_score) and (variance > self.min_variance):
                vision_embs.append(embedding)
                sigma_scores.append(mean_score)
                filtered_jobs.append(job)
        
        if(len(vision_embs)>1):
            vision_embs = np.concatenate(vision_embs, axis=0)
        
        return vision_embs, sigma_scores, filtered_jobs

    def get_variance(self, vision_emb: np.ndarray, ensemble_models: list):
        sigma_scores=[]
        for model in ensemble_models:
            mean=float(model.mean)
            std=float(model.standard_deviation)
            with torch.no_grad():
                score = model.predict_clip(torch.tensor(vision_emb).cuda()).item()
            
            score=(score - mean)/std
            sigma_scores.append(score)
        
        mean_score= np.mean(sigma_scores)
        variance= np.var(sigma_scores)

        return mean_score, variance

    def get_cluster_ids(self, vision_embs: np.ndarray):

        z = np.dot(vision_embs, self.pca_components.T[:, :self.n_pca_components])

        m = cosine_similarity(z, self.kmeans_cluster_centers_48)
        cluster_ids_48 = np.argmax(m, axis=1)
        m = cosine_similarity(z, self.kmeans_cluster_centers_1024)
        cluster_ids_1024 = np.argmax(m, axis=1)
        m = cosine_similarity(z, self.kmeans_cluster_centers_4096)
        cluster_ids_4096 = np.argmax(m, axis=1)

        return cluster_ids_48, cluster_ids_1024, cluster_ids_4096
    
    def get_image_pairs(self):
        jobs_list= self.get_latest_jobs()
        merged_list= []

        for dataset in jobs_list:
            print(f"Calculating pairs for the {dataset} dataset.........")
            # get jobs dataset
            jobs= jobs_list[dataset]
            print(f"{len(jobs)} jobs")
            # get ensemble model for dataset
            ensemble_models= self.get_ensemble_models(dataset=dataset)
            
            # filter by sigma score and variance and cluster embeddings
            vision_embs, sigma_scores, filtered_jobs = self.filter_by_score_and_variance(jobs=jobs, ensemble_models=ensemble_models)
            cluster_ids_48, cluster_ids_1024, cluster_ids_4096 = self.get_cluster_ids(vision_embs)

            job_uuids = [d["job_uuid"] for d in filtered_jobs]  

            # pairing
            sigma_score_pairs = get_candidate_pairs_by_score(
                job_uuids=job_uuids,
                scores = sigma_scores, 
                max_pairs = self.pairs, 
                n_bins = self.bins, 
                use_quantiles = (self.bin_type == 'quantile')
            )

            cluster_pairs_48 = get_candidate_pairs_within_category(
                job_uuids=job_uuids,
                categories = cluster_ids_48, 
                max_pairs = self.pairs
            )
            
            cluster_pairs_1024 = get_candidate_pairs_within_category(
                job_uuids=job_uuids,
                categories = cluster_ids_1024, 
                max_pairs = self.pairs
            )
            
            cluster_pairs_4096 = get_candidate_pairs_within_category(
                job_uuids=job_uuids,
                categories = cluster_ids_4096, 
                max_pairs = self.pairs
            )

            # merge pairs by sigma score and by cluster
            for pair in sigma_score_pairs:
                merged_list.append({
                    "pair": pair,
                    "policy": f"same_sigma_score_bin_{self.bins}"
                })

            # merge pairs by sigma score and by cluster
            for pair in cluster_pairs_48:
                if pair not in merged_list:
                    merged_list.append({
                        "pair": pair,
                        "policy": f"same_embedding_cluster_48"
                    })
           
            # merge pairs by sigma score and by cluster
            for pair in cluster_pairs_1024:
                if pair not in merged_list:
                    merged_list.append({
                        "pair": pair,
                        "policy": f"same_embedding_cluster_1024"
                    })
            
            # merge pairs by sigma score and by cluster
            for pair in cluster_pairs_4096:
                if pair not in merged_list:
                    merged_list.append({
                        "pair": pair,
                        "policy": f"same_embedding_cluster_4096"
                    })
                
        return merged_list
    
    def upload_pairs_to_queue(self, pair_list):
        
        for pair in tqdm(pair_list):
            job_uuid_1= pair['pair'][0]
            job_uuid_2= pair['pair'][1]

            endpoint_url = f"{API_URL}/ranking-queue/add-image-pair-to-queue?job_uuid_1={job_uuid_1}&job_uuid_2={job_uuid_2}&policy={pair['policy']}"
            response = requests.post(endpoint_url)

            if response.status_code == 200:
                print(f"Successfully processed job pair: UUID1: {job_uuid_1}, UUID2: {job_uuid_2}")
            else:
                print(f"Failed to process job pair: UUID1: {job_uuid_1}, UUID2: {job_uuid_2}. Response: {response.status_code} - {response.text}")

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--pca-model-path", type=str,
                        help="The path to PCA model npz file", default="input/model/active_learning/pca.npz")
    parser.add_argument("--kmeans-model-path", type=str,
                        help="The path to KMeans model npz file", default="input/model/active_learning/kmeans.npz")
    parser.add_argument("--pairs", type=int, default=1000,
                        help="The number of pairs")
    parser.add_argument("--bins", type=int, default=10,
                        help="The number of bins")
    parser.add_argument("--bin-type", type=str, default='quantile',
                        help="The binning method: fixed-range or quantile")
    parser.add_argument("--min-sigma-score", type=float, 
                        help="minimum sigma score when filtering images", default=1)
    parser.add_argument("--min-variance", type=float, 
                        help="minimum sigma score when filtering images", default=0.5)
    
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
        pca_model_path=args.pca_model_path,
        kmeans_model_path=args.kmeans_model_path,
        bin_type=args.bin_type,
        bins=args.bins,
        pairs=args.pairs,
        min_sigma_score=args.min_sigma_score,
        min_variance=args.min_variance
    )

    # get list of pairs
    pair_list=pipeline.get_image_pairs()
    print(pair_list)
    print(f"{len(pair_list)} pairs were created")

    # send list to active learning
    # pipeline.upload_pairs_to_queue(pair_list)
    

if __name__ == '__main__':
    main()