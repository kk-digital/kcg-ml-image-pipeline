import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
from minio import Minio
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
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
    parser.add_argument('--batch-size', required=False, default=256, type=int, help='batch size of the classifier models')

    args = parser.parse_args()
    return args

def get_dataset_list(bucket: str):
    datasets=[]
    
    if bucket == "external" or bucket == "extracts":
        datasets= http_get_extract_dataset_list()
    else:
        datasets= request.http_get_dataset_names()

    dataset_list = [dataset['dataset_name'] for dataset in datasets]
    
    return dataset_list

def initialize_dist_env(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ClipDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip_vector = torch.tensor(self.data[idx]["clip_vector"])
        uuid = self.data[idx]["uuid"]
        return clip_vector, uuid

class ImageClassifierScorer:
    def __init__(self,
                 minio_client: Minio,
                 bucket: str,
                 dataset: str,
                 batch_size=256):
        
        self.minio_client= minio_client
        self.bucket= bucket
        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size= torch.cuda.device_count()  # Automatically get the world size

        self.classifier_models={}
        self.dataloader=None

    def load_model(self, classifier_model_info):
        tag_name = classifier_model_info["tag_string"]
        classifier_name = classifier_model_info["classifier_name"]
        model_path = classifier_model_info["model_path"]

        print(f"loading classifier model for the {tag_name} tag...")

        if not classifier_name.endswith('clip-h'):
            print("Not support for classifier model: {}".format(classifier_model_info["classifier_name"]))
            return None

        if "elm" in classifier_name:
            elm_model = ELMRegression(device=self.device)
            loaded_model, model_file_name = elm_model.load_model_with_filename(
                self.minio_client, 
                model_path, 
                classifier_name)
        elif "linear" in classifier_name:
            linear_model = LinearRegression(device=self.device)
            loaded_model, model_file_name = linear_model.load_model_with_filename(
                self.minio_client, 
                model_path, 
                classifier_name)
        elif "logistic" in classifier_name:
            logistic_model = LogisticRegression(device=self.device)
            loaded_model, model_file_name = logistic_model.load_model_with_filename(
                self.minio_client, 
                model_path, 
                classifier_name)
        else:
            print("Not support for classifier model: {}".format(classifier_name))
            return None
        
        return loaded_model

    def calculate_and_upload_scores(self, rank, image_dataset):

        initialize_dist_env(rank, self.world_size)

        dataset = ClipDataset(image_dataset)
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)

        # get classifier model list
        classifier_model_list = request.http_get_classifier_model_list()

        for classifier_info in classifier_model_list:
            classifier_id= classifier_info["classifier_id"]
            classifier_model= self.load_model(classifier_model_info=classifier_info)

            # Move classifier to the current device
            classifier_model = classifier_model.to(rank)
            classifier_model = DDP(classifier_model, device_ids=[rank])

            for clip_vectors, uuids in dataloader:
                clip_vectors = clip_vectors.to(rank)
                with torch.no_grad():
                    scores = classifier_model.module.classify(clip_vectors)
                
                with ThreadPoolExecutor(max_workers=50) as executor:
                    futures = []
                    for score, uuid in zip(scores, uuids):
                        # upload score
                        score_data= {
                            "job_uuid": uuid,
                            "classifier_id": classifier_id,
                            "score": score,
                        }

                        print(score_data)
                        # futures.append(executor.submit(request.http_add_classifier_score, score_data=score_data))

                    # for _ in tqdm(as_completed(futures), total=len(self.batch_size)):
                    #     continue
    
def main():
    args = parse_args()
    bucket_name = args.bucket
    dataset_name = args.dataset
    batch_size= args.batch_size
    
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)

    if dataset_name != "all":
        scorer= ImageClassifierScorer(minio_client, bucket_name, dataset_name, batch_size)
        print(f"Load the {bucket_name}/{dataset_name} dataset")
        dataset_loader= ImageDatasetLoader(minio_client, bucket_name, dataset_name)
        image_dataset= dataset_loader.load_dataset()

        mp.spawn(scorer.calculate_and_upload_scores(), args=(image_dataset,), nprocs=scorer.world_size, join=True)

    else:
        # if all, train models for all existing datasets
        # get dataset name list
        dataset_names = get_dataset_list(bucket_name)
        print("dataset names=", dataset_names)
        for dataset in dataset_names:
            try:
                scorer= ImageClassifierScorer(minio_client, bucket_name, dataset_name, batch_size)
                print(f"Load the {bucket_name}/{dataset_name} dataset")
                dataset_loader= ImageDatasetLoader(minio_client, bucket_name, dataset_name)
                image_dataset= dataset_loader.load_dataset()
                mp.spawn(scorer.calculate_and_upload_scores, args=(image_dataset,), nprocs=scorer.world_size, join=True)

            except Exception as e:
                print("Error running image scorer for {}: {}".format(dataset, e))


if __name__ == "__main__":
    main()
        