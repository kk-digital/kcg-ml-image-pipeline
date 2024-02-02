import csv
from datetime import datetime, timedelta
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


base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from utility.minio import cmd
from utility.minio.cmd import connect_to_minio_client
from utility.clip.clip import ClipModel

API_URL = "http://192.168.3.1:8111"
OUTPUT_PATH="environmental/output/forest_vs_non_forest_pairs"

class ForestActiveLearningPipeline:

    def __init__(self, minio_addr: str, minio_access_key: str, minio_secret_key: str,
                 high_cosine_threshold: float, low_cosine_threshold: float, quality_threshold: float):

        # get device
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = torch.device(device)

        # Define thresholds for high and low cosine similarity and quality
        self.high_cosine_threshold = high_cosine_threshold  
        self.low_cosine_threshold = low_cosine_threshold
        self.quality_threshold = quality_threshold 
        
        # connect to minio
        self.connect_to_minio_client(minio_addr, minio_access_key, minio_secret_key)
        # load clip scoring model
        self.load_scoring_model()
        # load clip model
        self.clip_model = ClipModel(device=device)
        self.clip_model.load_clip()
        self.clip_model.load_tokenizer()
        # calculate clip embedding for forest
        self.text_embs= self.get_forest_text_embs()

    def get_forest_text_embs(self):
        topic_phrase_list=[
            "jungle", "forest", "topical", "tree", "grass"
        ]
        text_embs= [self.compute_clip_vector(phrase) for phrase in topic_phrase_list]

        return text_embs

    def get_jobs(self):
        print('Loading image file paths for environmntal dataset.........')

        response = requests.get(f'{API_URL}/queue/image-generation/list-completed-by-dataset?dataset=environmental')
        
        jobs = json.loads(response.content)

        return jobs

    def connect_to_minio_client(self, minio_addr: str, minio_access_key: str, minio_secret_key: str):

        self.client = connect_to_minio_client(
            minio_addr, 
            minio_access_key, 
            minio_secret_key
        )

        self.bucket_name = 'datasets'
    
    # compute text clip vector
    def compute_clip_vector(self, text):
        clip_vector_gpu = self.clip_model.get_text_features(text)
        clip_vector_cpu = clip_vector_gpu.cpu()

        del clip_vector_gpu

        clip_vector = clip_vector_cpu.tolist()
        return clip_vector
    
    # compute clip similarity
    def compute_cosine_value(self, image_clip, phrase_emb):

        # convert numpy array to tensors
        phrase_clip_vector = torch.tensor(phrase_emb, dtype=torch.float32, device=self.device)
        image_clip_vector = torch.tensor(image_clip, dtype=torch.float32, device=self.device)

        #check the vector size
        assert phrase_clip_vector.size() == (1, 768), f"Expected size (1, 768), but got {phrase_clip_vector.size()}"
        assert image_clip_vector.size() == (1, 768), f"Expected size (1, 768), but got {image_clip_vector.size()}"

        # removing the extra dimension
        # from shape (1, 768) => (768)
        phrase_clip_vector = phrase_clip_vector.squeeze(0)
        image_clip_vector = image_clip_vector.squeeze(0)

        # Normalizing the tensor
        normalized_phrase_clip_vector = torch.nn.functional.normalize(phrase_clip_vector.unsqueeze(0), p=2, dim=1)
        normalized_image_clip_vector = torch.nn.functional.normalize(image_clip_vector.unsqueeze(0), p=2, dim=1)

        # removing the extra dimension
        # from shape (1, 768) => (768)
        normalized_phrase_clip_vector = normalized_phrase_clip_vector.squeeze(0)
        normalized_image_clip_vector = normalized_image_clip_vector.squeeze(0)

        # cosine similarity
        similarity = torch.dot(normalized_phrase_clip_vector, normalized_image_clip_vector)

        # cleanup
        del phrase_clip_vector
        del image_clip_vector
        del normalized_phrase_clip_vector
        del normalized_image_clip_vector

        return similarity.item()
    
    # calculate max cosine similarity for a list of phrases
    def max_cosine_similarity(self, image_clip):
        all_similarities= [self.compute_cosine_value(image_clip, phrase_emb) for phrase_emb in self.text_embs]
        max_similarity= max(all_similarities)

        return max_similarity
    
    # load elm scoring model
    def load_scoring_model(self):
        input_path=f"environmental/models/ranking/"
        
        self.ranking_model = ABRankingELMModel(768)

        # Get all model files
        model_files = cmd.get_list_of_objects_with_prefix(self.client, 'datasets', input_path)

        for model_file in model_files:
            if model_file.endswith("score-elm-v1-clip.safetensors"):
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

        self.ranking_model.load_safetensors(byte_buffer)
        self.ranking_model.model=self.ranking_model.model.to(self.device)

        self.mean=float(self.ranking_model.mean)
        self.std=float(self.ranking_model.standard_deviation)

    def get_score(self, vision_emb):
        with torch.no_grad():
            score = self.ranking_model.predict_clip(torch.tensor(vision_emb).cuda()).item()
        
        score=(score - self.mean)/self.std

        return score

    def get_image_pairs(self):
        jobs= self.get_jobs()
        job_data=[]

        print(f"Calculating clip scores and forest relatedness.........")
        # get jobs dataset
        print(f"{len(jobs)} jobs")

        for job in tqdm(jobs, leave=False):
            
            file_path= job['task_output_file_dict']['output_file_path']
            # get clip embedding file path from image file path
            object_name = file_path.replace(f'{self.bucket_name}/', '')
            object_name = os.path.splitext(object_name.split('_')[0])[0]
            object_name = f'{object_name}_clip.msgpack'
    
            try:
                # get clip embedding    
                data = self.client.get_object(self.bucket_name, object_name).data
                decoded_data = msgpack.unpackb(data)
                embedding= np.array(decoded_data['clip-feature-vector']).astype('float32')
                
                
            except Exception as e:
                # Handle the exception (e.g., log the error message)
                print(f"Error loading embedding for {object_name}: {e}")
                continue  # Continue the loop in case of failure

            # calculate score and clip similarity
            image_score = self.get_score(embedding)
            image_similarity = self.max_cosine_similarity(image_clip=embedding)

            # store job data
            job_data.append({
                "job_uuid": job["uuid"],
                "quality_score": image_score,
                "cosine_similarity": image_similarity
            })
        

        pairs = self.pair_images(job_data)
        self.write_pairs_to_csv(pairs)

        return pairs
    

    def write_pairs_to_csv(self, pairs):
        local_file_path = "forest_vs_non_forest_pairs.csv"
        # Open the file in write mode
        with open(local_file_path, mode='w', newline='') as file:
            # Create a CSV writer
            writer = csv.writer(file)
            
            # Write the header
            writer.writerow(['Image1_UUID', 'Image2_UUID'])
            
            # Write each pair
            for pair in pairs:
                writer.writerow([pair[0], pair[1]])
        
        # Read the contents of the .npz file
        with open(local_file_path, 'rb') as file:
            content = file.read()

        # Upload the local file to MinIO
        buffer = io.BytesIO(content)
        buffer.seek(0)

        minio_path=OUTPUT_PATH + f"/{local_file_path}"
        cmd.upload_data(self.client, 'datasets', minio_path, buffer)

        # Remove the temporary file
        os.remove(local_file_path)

    def pair_images(self, image_jobs):
        # Separate images into two lists based on cosine similarity
        high_cosine_images = [img for img in image_jobs if img['cosine_similarity'] >= self.high_cosine_threshold ]
        low_cosine_high_score_images = [img for img in image_jobs if img['cosine_similarity'] < self.low_cosine_threshold]
        
        # Sort the lists
        # High cosine images sorted by cosine similarity descending
        high_cosine_images.sort(key=lambda x: x['cosine_similarity'], reverse=True)

        # print percentage of forest related images
        print(len(high_cosine_images)/len(image_jobs))
        
        # Low cosine, high score images sorted by quality score descending
        low_cosine_high_score_images.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # Initialize pairs list
        pairs = []
        num_pairs=0
        
        # Pairing logic
        for high_cosine_img in high_cosine_images:
            for low_cosine_high_score_img in low_cosine_high_score_images:
                # Ensure the low_cosine image has a higher quality score than the high_cosine image
                if low_cosine_high_score_img['quality_score'] > high_cosine_img['quality_score'] + self.quality_threshold:
                    pairs.append((high_cosine_img['job_uuid'], low_cosine_high_score_img['job_uuid']))
                    # Once paired, remove the low_cosine_high_score_img to avoid reusing it
                    low_cosine_high_score_images.remove(low_cosine_high_score_img)

                    num_pairs+=1
                    break  # move to the next high_cosine_img
            if(num_pairs==10000):
                break
    
        return pairs
    
    def upload_pairs_to_queue(self, pair_list):
        
        for pair in tqdm(pair_list):
            job_uuid_1= pair[0]
            job_uuid_2= pair[1]

            endpoint_url = f"{API_URL}/ranking-queue/add-image-pair-to-queue?job_uuid_1={job_uuid_1}&job_uuid_2={job_uuid_2}&policy=forest_vs_non_forest_related"
            response = requests.post(endpoint_url)

            if response.status_code == 200:
                print(f"Successfully processed job pair: UUID1: {job_uuid_1}, UUID2: {job_uuid_2}")
            else:
                print(f"Failed to process job pair: UUID1: {job_uuid_1}, UUID2: {job_uuid_2}. Response: {response.status_code} - {response.text}")

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--quality-threshold", type=float,
                        help="threshold of difference in quality between pairs", default=0.5)
    parser.add_argument("--high-cosine-threshold", type=float,
                        help="Min cosine similarity for forest related images", default=0.19)
    parser.add_argument("--low-cosine-threshold", type=float,
                        help="Max cosine similaritiy for non-forest related images", default=0.15)
    
    parser.add_argument("--minio-addr", type=str, default=None,
                        help="The minio server ip address")
    parser.add_argument("--minio-access-key", type=str,
                        help="The minio access key to use so worker can upload files to minio server")
    parser.add_argument("--minio-secret-key", type=str,
                        help="The minio secret key to use so worker can upload files to minio server")

    return parser.parse_args()

def main():
    
    args = parse_args()

    pipeline = ForestActiveLearningPipeline(
        minio_addr=args.minio_addr,
        minio_access_key=args.minio_access_key,
        minio_secret_key=args.minio_secret_key,
        high_cosine_threshold=args.high_cosine_threshold,
        low_cosine_threshold=args.low_cosine_threshold,
        quality_threshold=args.quality_threshold,
    )

    # get list of pairs
    pair_list=pipeline.get_image_pairs()

    print(f"created {len(pair_list)} pairs")

    # send list to active learning
    pipeline.upload_pairs_to_queue(pair_list)
    

if __name__ == '__main__':
    main()