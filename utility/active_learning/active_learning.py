import io
import json
import sys
import os
import requests
from tqdm.auto import tqdm
import argparse
import msgpack
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from training_worker.ab_ranking.model.ab_ranking_linear import ABRankingModel
from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from utility.minio import cmd
from utility.minio.cmd import connect_to_minio_client
from utility.active_learning.pairs import get_candidate_pairs_by_score, get_candidate_pairs_within_category

API_URL = "http://123.176.98.90:8764"

class ActiveLearningPipeline:

    def __init__(self, minio_addr: str, minio_access_key: str, minio_secret_key: str, 
                 policy: str, dataset: str, scoring_model_type: str, pca_model_path: str, 
                 kmeans_model_path: str, bins: int, bin_type: str , cluster_type: str, 
                 pairs: int, csv_path: str, min_sigma_score: float, min_variance: float):
 
        self.policy=policy
        self.dataset=dataset
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
        
        self.image_list=self.filter_images(csv_path) 
        self.connect_to_minio_client(minio_addr, minio_access_key, minio_secret_key)
        self.load_models(scoring_model_type, pca_model_path, kmeans_model_path, cluster_type)

    def filter_images(self, csv_path):
        df= pd.read_csv(csv_path)
        
        # Assert that necessary columns are in the DataFrame
        assert 'task_uuid' in df.columns, "Column 'task_uuid' not found in csv file"
        assert 'variance' in df.columns, "Column 'variance' not found in csv file"
        assert 'sigma_score' in df.columns, "Column 'sigma_score' not found in csv file"
        
        # Filter the DataFrame based on minimum sigma_score and variance
        filtered_df = df[(df['sigma_score'] >= self.min_sigma_score) & (df['variance'] >= self.min_variance)]

        print('Loading image file paths..........')
        for i, job_uuid in enumerate(tqdm(filtered_df['task_uuid'])):
            try:
                info = self.get_info(job_uuid)
        
                if 'task_output_file_dict' in info:
                    output_info = info['task_output_file_dict']
                    if 'output_file_path' in output_info:
                        filtered_df.loc[i, 'file_path'] = output_info['output_file_path']
                    if 'output_file_hash' in output_info:
                        filtered_df.loc[i, 'file_hash'] = output_info['output_file_hash']
            
            except:
                continue

        return filtered_df
    
    def get_info(self, job_uuid: str):
        
        response = requests.get(f'http://{API_URL}/job/get-job/{job_uuid}')
        
        info = json.loads(response.content)

        return info

    def connect_to_minio_client(self, minio_addr: str, minio_access_key: str, minio_secret_key: str):

        self.client = connect_to_minio_client(
            minio_addr, 
            minio_access_key, 
            minio_secret_key
        )

        self.bucket_name = 'datasets'
    
    # load elm or linear scoring models
    def load_scoring_model(self, scoring_model_type: str):
        input_path=f"{self.dataset}/models/ranking/"

        if(scoring_model_type=="elm"):
            scoring_model = ABRankingELMModel(768)
            file_name=f"score-elm-v1-clip.safetensors"
        else:
            scoring_model= ABRankingModel(768)
            file_name=f"score-linear-clip.safetensors"

        model_files=cmd.get_list_of_objects_with_prefix(self.client, 'datasets', input_path)
        most_recent_model = None

        for model_file in model_files:
            if model_file.endswith(file_name):
                most_recent_model = model_file

        if most_recent_model:
            model_file_data =cmd.get_file_from_minio(self.client, 'datasets', most_recent_model)
        else:
            print("No .safetensors files found in the list.")
            return
        
        print(most_recent_model)

        # Create a BytesIO object and write the downloaded content into it
        byte_buffer = io.BytesIO()
        for data in model_file_data.stream(amt=8192):
            byte_buffer.write(data)
        # Reset the buffer's position to the beginning
        byte_buffer.seek(0)

        scoring_model.load_safetensors(byte_buffer)
        scoring_model.model=scoring_model.model.to(self.device)

        return scoring_model

    def load_models(self, scoring_model_type: str,  pca_model_path: str, kmeans_model_path: str, cluster_type: str):

        self.scoring_model = self.load_scoring_model(scoring_model_type) 

        self.scoring_model_mean = float(self.scoring_model.mean)
        self.scoring_model_standard_deviation = float(self.scoring_model.standard_deviation)

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
    
    def upload_pairs_to_queue(self, pair_list):
        
        for pair in tqdm(pair_list):
            job_uuid_1= pair[0]
            job_uuid_2= pair[1]

            endpoint_url = f"{API_URL}/ranking-queue/add-image-pair-to-queue?job_uuid_1={job_uuid_1}&job_uuid_2={job_uuid_2}&policy={self.policy}"
            response = requests.post(endpoint_url)

            if response.status_code == 200:
                print(f"Successfully processed job pair: UUID1: {job_uuid_1}, UUID2: {job_uuid_2}")
            else:
                print(f"Failed to process job pair: UUID1: {job_uuid_1}, UUID2: {job_uuid_2}. Response: {response.status_code} - {response.text}")

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--csv-path", type=str, required=True,
                        help="The path to csv file")
    parser.add_argument("--policy-string", type=str, required=True,
                        help="name of policy")
    parser.add_argument("--dataset", type=str, required=True,
                        help="name of dataset")
    parser.add_argument("--scoring-model-type", type=str,
                        help="elm or linear", default="linear")
    parser.add_argument("--pca-model-path", type=str,
                        help="The path to PCA model npz file", default="input/model/active_learning/kmeans.npz")
    parser.add_argument("--kmeans-model-path", type=str,
                        help="The path to KMeans model npz file", default="input/model/active_learning/pca.npz")
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
        policy=args.policy_string,
        dataset=args.dataset,
        scoring_model_type=args.scoring_model_type,
        pca_model_path=args.pca_model_path,
        kmeans_model_path=args.kmeans_model_path,
        bin_type=args.bin_type,
        bins=args.bins,
        cluster_type=args.cluster_type,
        pairs=args.pairs,
        csv_path=args.csv_path,
        min_sigma_score=args.min_sigma_score,
        min_variance=args.min_variance
    )

    # get list of pairs
    pair_list=pipeline.get_image_pairs()
    print(pair_list)

    # send list to active learning
    # pipeline.upload_pairs_to_queue(pair_list, args.policy_string)
    

if __name__ == '__main__':
    main()