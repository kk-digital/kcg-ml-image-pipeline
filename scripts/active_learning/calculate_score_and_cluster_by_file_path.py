import sys
import os
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


class ScoreAndClusterCalculator:

    def __init__(self, minio_addr: str, minio_access_key: str, minio_secret_key: str, scoring_model_path: str, pca_model_path: str, kmeans_model_path: str):

        self.connect_to_minio_client(minio_addr, minio_access_key, minio_secret_key)
        
        self.load_models(scoring_model_path, pca_model_path, kmeans_model_path)

    def connect_to_minio_client(self, minio_addr: str, minio_access_key: str, minio_secret_key: str):

        self.client = connect_to_minio_client(
            minio_addr, 
            minio_access_key, 
            minio_secret_key
        )

        self.bucket_name = 'datasets'
    
    def load_models(self, scoring_model_path: str, pca_model_path: str, kmeans_model_path: str):
        
        model = ABRankingModel(768)
        
        model.load_safetensors(open(scoring_model_path, 'rb'))

        self.scoring_model = model.model.linear

        self.scoring_model_mean = float(model.mean)
        self.scoring_model_standard_deviation = float(model.standard_deviation)

        npz = np.load(pca_model_path)

        self.pca_components = npz['components']
        
        self.n_pca_components = 24

        npz = np.load(kmeans_model_path)

        self.kmeans_cluster_centers_48 = npz['cluster_centers_48']
        self.kmeans_cluster_centers_1024 = npz['cluster_centers_1024']
        self.kmeans_cluster_centers_4096 = npz['cluster_centers_4096']
    
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

        return scores, sigma_scores

    def get_cluster_ids(self, vision_embs: np.ndarray):

        z = np.dot(vision_embs, self.pca_components.T[:, :self.n_pca_components])

        m = cosine_similarity(z, self.kmeans_cluster_centers_48)
        cluster_ids_48 = np.argmax(m, axis=1)
        m = cosine_similarity(z, self.kmeans_cluster_centers_1024)
        cluster_ids_1024 = np.argmax(m, axis=1)
        m = cosine_similarity(z, self.kmeans_cluster_centers_4096)
        cluster_ids_4096 = np.argmax(m, axis=1)

        return cluster_ids_48, cluster_ids_1024, cluster_ids_4096

def parse_args():
    parser = argparse.ArgumentParser(description="download job uuid info by file_path")

    # Required parameters
    
    parser.add_argument("--csv-path", type=str,
                        help="The path to csv file")
    
    parser.add_argument("--scoring-model-path", type=str,
                        help="The path to clip scoring model safetensors file")
    parser.add_argument("--pca-model-path", type=str,
                        help="The path to PCA model npz file")
    parser.add_argument("--kmeans-model-path", type=str,
                        help="The path to KMeans model npz file")
    
    parser.add_argument("--minio-addr", type=str, default=None,
                        help="The minio server ip address")
    parser.add_argument("--minio-access-key", type=str,
                        help="The minio access key to use so worker can upload files to minio server")
    parser.add_argument("--minio-secret-key", type=str,
                        help="The minio secret key to use so worker can upload files to minio server")

    return parser.parse_args()

def main():
    
    args = parse_args()

    calculator = ScoreAndClusterCalculator(
        minio_addr=args.minio_addr,
        minio_access_key=args.minio_access_key,
        minio_secret_key=args.minio_secret_key,
        scoring_model_path=args.scoring_model_path,
        pca_model_path=args.pca_model_path,
        kmeans_model_path=args.kmeans_model_path
    )
    # read

    df = pd.read_csv(args.csv_path)

    assert 'file_path' in df.columns

    # if 'image_clip_score' not in df.columns:
    #     df['image_clip_score'] = None
    # if 'image_clip_sigma_score' not in df.columns:
    #     df['image_clip_sigma_score'] = None

    # if 'cluster_id_48' not in df.columns:
    #     df['cluster_id_48'] = None
    # if 'cluster_id_1024' not in df.columns:
    #     df['cluster_id_1024'] = None
    # if 'cluster_id_4096' not in df.columns:
    #     df['cluster_id_4096'] = None
        
    # calculate

    vision_embs = calculator.load_vision_embs(df['file_path'])

    image_clip_score, image_clip_sigma_score = calculator.get_scores(vision_embs)

    cluster_ids_48, cluster_ids_1024, cluster_ids_4096 = calculator.get_cluster_ids(vision_embs)

    df['image_clip_score'] = image_clip_score
    df['image_clip_sigma_score'] = image_clip_sigma_score
    df['cluster_id_48'] = cluster_ids_48
    df['cluster_id_1024'] = cluster_ids_1024
    df['cluster_id_4096'] = cluster_ids_4096
                
    # save

    df.to_csv(args.csv_path, index=False)
    

if __name__ == '__main__':
    main()
