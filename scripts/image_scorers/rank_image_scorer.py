import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import threading
import time
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from multiprocessing import Value
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

base_directory = "./"
sys.path.insert(0, base_directory)
from scripts.image_scorers.dataloader.image_dataset_loader import ImageDatasetLoader
from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from training_worker.ab_ranking.model.ab_ranking_linear import ABRankingModel
from utility.http import request
from utility.http.external_images_request import http_get_extract_dataset_list
from utility.minio import cmd

def parse_args():
    parser = argparse.ArgumentParser(description="Embedding Scorer")
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--bucket', required=True, help='name of bucket')
    parser.add_argument('--dataset', required=True, help='name of dataset')
    parser.add_argument('--model-type', required=True, help='type of model elm-v1 or linear', default="elm-v1")
    parser.add_argument('--batch-size', required=False, default=256, type=int, help='batch size of the classifier models')

    args = parser.parse_args()
    return args

def get_rank():
    rank= dist.get_rank()
    return rank

def print_in_rank(msg: str):
    rank= get_rank()
    print(f"gpu {rank}: {msg}")

def get_dataset_list(bucket: str):
    datasets=[]
    
    if bucket == "external" or bucket == "extracts":
        datasets= http_get_extract_dataset_list()
    else:
        datasets= request.http_get_dataset_names()

    dataset_list = [dataset['dataset_name'] for dataset in datasets]
    
    return dataset_list

# Initialize the distributed environment
def initialize_dist_env(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Cleanup the distributed environment
def cleanup():
    dist.destroy_process_group()

class ClipDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'clip_vector': torch.tensor(self.data[idx]["clip_vector"]).squeeze(),
            'uuid': self.data[idx]["uuid"],
            'image_hash': self.data[idx]["image_hash"]
        }
    
def collate_fn(batch):
    clip_vectors = torch.stack([item['clip_vector'] for item in batch])
    uuids = [item['uuid'] for item in batch]
    image_hashes = [item['image_hash'] for item in batch]
    return {'uuids': uuids, 'image_hashes':image_hashes , 'clip_vectors': clip_vectors}

def load_model(minio_client, rank_id, model_type, model_path, device):

    model_file_data =cmd.get_file_from_minio(minio_client, 'datasets', model_path)
    
    if model_file_data is None:
        print(f"No ranking model was found for rank {rank_id}.")
        return None

    if model_type == "elm-v1":
        scoring_model = ABRankingELMModel(1280, device=device)
    elif model_type == "linear":
        scoring_model = ABRankingModel(1280, device=device)
    else:
        print(f"No ranking models are available for this model type.")
        return None

    # Create a BytesIO object and write the downloaded content into it
    byte_buffer = io.BytesIO()
    for data in model_file_data.stream(amt=8192):
        byte_buffer.write(data)
    # Reset the buffer's position to the beginning
    byte_buffer.seek(0)

    scoring_model.load_safetensors(byte_buffer)

    print(f"model {model_path} loaded")

    return scoring_model

def calculate_and_upload_scores(rank, world_size, image_dataset, image_source, ranking_models, batch_size):
    initialize_dist_env(rank, world_size)
    rank_device = torch.device(f'cuda:{rank}')

    dataset = ClipDataset(image_dataset)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)

    start_time = time.time()
    total_uploaded = 0
    futures = []
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        for model_id, ranking_model_data in ranking_models.items():
            rank_id = ranking_model_data["rank_id"]
            ranking_model = ranking_model_data["model"]
            ranking_model.model= ranking_model.model.to(device=rank_device)
            score_mean= float(ranking_model.mean)
            score_std= float(ranking_model.standard_deviation)

            print_in_rank(f"calculating scores for rank id {rank_id}")

            try:
                for batch_idx, image_data in enumerate(tqdm(dataloader)):

                    clip_vectors = image_data["clip_vectors"]
                    uuids = image_data["uuids"]
                    image_hashes = image_data["image_hashes"]

                    clip_vectors = clip_vectors.to(rank_device)
                    
                    print_in_rank("running scorer")
                    with torch.no_grad():
                        scores = ranking_model.predict_clip(clip_vectors)
                        print(scores[0])
                        sigma_scores= (scores - score_mean) / score_std
                    
                    print_in_rank("getting batch")
                    scores_batch= {}
                    scores_batch["scores"]= []
                    for score, sigma_score, uuid, image_hash in zip(scores, sigma_scores, uuids, image_hashes):
                        score_data = {
                            "rank_model_id": model_id,
                            "rank_id": rank_id,
                            "image_hash": image_hash,
                            "uuid": uuid,
                            "score": score.item(),
                            "sigma_score": sigma_score.item(),
                            "image_source": image_source
                        }
                        scores_batch["scores"].append(score_data)
                    
                    print_in_rank("sending job")
                    futures.append(executor.submit(request.http_add_rank_score_batch, scores_batch=scores_batch))

            except Exception as e:
                print_in_rank(f"exception occurred when uploading scores {e}")

    # Periodically check and report progress
    last_report_time = time.time()
    while futures:
        for future in as_completed(futures):
            try:
                future.result()  # Ensure any exceptions are raised
                total_uploaded += batch_size 
            except Exception as e:
                print_in_rank(f"Exception in future: {e}")
            futures.remove(future)

        current_time = time.time()
        if current_time - last_report_time >= 10:
            last_report_time = current_time
            total_uploaded_tensor = torch.tensor(total_uploaded, device=rank_device)
            dist.all_reduce(total_uploaded_tensor, op=dist.ReduceOp.SUM)
            total_uploaded_all_ranks = total_uploaded_tensor.item()

            elapsed_time = time.time() - start_time
            speed = total_uploaded_all_ranks / elapsed_time
            print(f"Uploaded {total_uploaded_all_ranks} scores at {speed:.2f} scores/sec")
    
    dist.barrier()
    
    cleanup()

def main():
    args = parse_args()

    bucket_name = args.bucket
    dataset_name = args.dataset
    model_type = args.model_type
    batch_size = args.batch_size

    # set image source
    if args.bucket=="external":
        image_source= "external_image"
    elif args.bucket=="extracts":
        image_source= "extract_image"
    else:
        image_source= "generated_image"

    minio_client = cmd.get_minio_client(
        minio_access_key=args.minio_access_key,
        minio_secret_key=args.minio_secret_key,
        minio_ip_addr=args.minio_addr
    )

    print(f"Load all rank models")
    rank_model_list = request.http_get_ranking_model_list()
    rank_models = {}
    for rank_info in rank_model_list:
        ranking_model_type = rank_info["model_type"]

        if model_type != ranking_model_type:
            continue

        rank_id = rank_info["rank_id"]
        model_id = rank_info["ranking_model_id"]
        model_path = rank_info["model_path"]
        rank_model= None

        print(f"Loading the ranking model for rank id: {rank_id}...")
        rank_model = load_model(minio_client, rank_id, model_type, model_path, torch.device('cpu'))

        if rank_model is not None:
            rank_models[model_id] = { "model": rank_model, "rank_id": rank_id}

    if dataset_name != "all":
        print(f"Load the {bucket_name}/{dataset_name} dataset")
        dataset_loader = ImageDatasetLoader(minio_client, bucket_name, dataset_name)
        image_dataset = dataset_loader.load_dataset()

        world_size = torch.cuda.device_count()
        mp.spawn(calculate_and_upload_scores, args=(world_size, image_dataset, image_source, rank_models, batch_size), nprocs=world_size, join=True)
    else:
        dataset_names = get_dataset_list(bucket_name)
        print("Dataset names:", dataset_names)
        for dataset in dataset_names:
            try:
                dataset_loader = ImageDatasetLoader(minio_client, bucket_name, dataset)
                image_dataset = dataset_loader.load_dataset()
                mp.spawn(calculate_and_upload_scores, args=(world_size, image_dataset, image_source, rank_models, batch_size), nprocs=world_size, join=True)
            except Exception as e:
                print(f"Error running image scorer for {dataset}: {e}")

if __name__ == "__main__":
    main()