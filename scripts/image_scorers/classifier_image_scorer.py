import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from training_worker.classifiers.models.elm_regression import ELMRegression
from training_worker.classifiers.models.linear_regression import LinearRegression
from training_worker.classifiers.models.logistic_regression import LogisticRegression
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
    parser.add_argument('--model-type', required=True, help='type of model elm, linear or logistic', default="all")
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
    os.environ['MASTER_PORT'] = '12356'
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

def load_model(minio_client, classifier_model_info, device):
    classifier_name = classifier_model_info["classifier_name"]
    model_path = classifier_model_info["model_path"]

    print(f"Loading classifier model {classifier_name}...")

    if 'clip-h' not in classifier_name:
        print(f"Not support for classifier model: {classifier_name}")
        return None

    if "elm" in classifier_name:
        elm_model = ELMRegression(device=device)
        loaded_model, model_file_name = elm_model.load_model_with_filename(
            minio_client, 
            model_path, 
            classifier_name)
    elif "linear" in classifier_name:
        linear_model = LinearRegression(device=device)
        loaded_model, model_file_name = linear_model.load_model_with_filename(
            minio_client, 
            model_path, 
            classifier_name)
    elif "logistic" in classifier_name:
        logistic_model = LogisticRegression(device=device)
        loaded_model, model_file_name = logistic_model.load_model_with_filename(
            minio_client, 
            model_path, 
            classifier_name)
    else:
        print(f"Not support for classifier model: {classifier_name}")
        return None
    
    return loaded_model

def calculate_and_upload_scores(rank, world_size, image_dataset, image_source, classifier_models, batch_size):
    initialize_dist_env(rank, world_size)
    rank_device = torch.device(f'cuda:{rank}')

    dataset = ClipDataset(image_dataset)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)

    start_time = time.time()
    total_uploaded = 0
    futures = []
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        for classifier_id, classifier_data in classifier_models.items():
            tag_id = classifier_data["tag_id"]
            classifier_model = classifier_data["model"]
            classifier_model.set_device(rank_device)

            print_in_rank(f"calculating scores for classifier id {classifier_id}")

            try:
                for batch_idx, image_data in enumerate(tqdm(dataloader)):

                    clip_vectors = image_data["clip_vectors"]
                    uuids = image_data["uuids"]
                    image_hashes = image_data["image_hashes"]

                    clip_vectors = clip_vectors.to(rank_device)
                    
                    with torch.no_grad():
                        scores = classifier_model.classify(clip_vectors)
                    
                    scores_batch= {}
                    scores_batch["scores"]= []
                    for score, uuid, image_hash in zip(scores, uuids, image_hashes):
                        score_data = {
                            "job_uuid": uuid,
                            "image_hash": image_hash,
                            "classifier_id": classifier_id,
                            "tag_id": tag_id,
                            "score": score.item(),
                            "image_source": image_source
                        }
                        scores_batch["scores"].append(score_data)
                    
                    futures.append(executor.submit(request.http_add_classifier_score_batch, scores_batch=scores_batch))
                
                # remove model from gpu
                classifier_model.set_device(torch.device('cpu'))

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

    print(f"Load all classifier models")
    classifier_model_list = request.http_get_classifier_model_list()
    classifier_models = {}
    for classifier_info in classifier_model_list:
        classifier_id = classifier_info["classifier_id"]
        classifier_name = classifier_info["classifier_name"]
        tag_id = classifier_info["tag_id"]
        classifier_model= None

        if model_type in classifier_name or model_type=="all":
            classifier_model = load_model(minio_client, classifier_info, torch.device('cpu'))

        if classifier_model is not None:
            classifier_models[classifier_id] = { "model": classifier_model, "tag_id": tag_id}

    if dataset_name != "all":
        print(f"Load the {bucket_name}/{dataset_name} dataset")
        dataset_loader = ImageDatasetLoader(minio_client, bucket_name, dataset_name)
        image_dataset = dataset_loader.load_dataset()

        world_size = torch.cuda.device_count()
        mp.spawn(calculate_and_upload_scores, args=(world_size, image_dataset, image_source, classifier_models, batch_size), nprocs=world_size, join=True)
    else:
        dataset_names = get_dataset_list(bucket_name)
        print("Dataset names:", dataset_names)
        for dataset in dataset_names:
            try:
                dataset_loader = ImageDatasetLoader(minio_client, bucket_name, dataset)
                image_dataset = dataset_loader.load_dataset()
                mp.spawn(calculate_and_upload_scores, args=(world_size, image_dataset, image_source, classifier_models, batch_size), nprocs=world_size, join=True)
            except Exception as e:
                print(f"Error running image scorer for {dataset}: {e}")

if __name__ == "__main__":
    main()
