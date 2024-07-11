import argparse
from io import BytesIO
import os
import sys
import msgpack
import torch
from tqdm import tqdm

from data_loader.utils import get_object
from utility.minio import cmd
from utility.path import separate_bucket_and_file_path

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())
from kandinsky.models.clip_image_encoder.clip_image_encoder import KandinskyCLIPImageEncoder
from utility.http.request import http_get_completed_job_by_dataset, http_get_dataset_names
from utility.http.external_images_request import http_get_external_image_list, http_get_extract_image_list, http_get_external_dataset_list, http_get_extract_dataset_list
API_URL="http://192.168.3.1:8111"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minio-access-key', help='Minio access key')
    parser.add_argument('--minio-secret-key', help='Minio secret key')
    parser.add_argument('--bucket', type=str, help='bucket containing the dataset')
    parser.add_argument('--dataset', type=str, help='Dataset to calculate clip vectors for')
    parser.add_argument('--batch-size', type=int, default=10000, help='batch size for storing clip files')

    return parser.parse_args()

def save_data_to_minio(minio_client, bucket, data_path, data):
    data_buffer = BytesIO()
    packed_data = msgpack.packb(data, use_bin_type=True)
    data_buffer.write(packed_data)
    data_buffer.seek(0)
    minio_client.put_object(bucket, data_path, data_buffer, len(data_buffer.getvalue()))

def get_dataset_list(bucket: str):
    datasets=[]
    
    if bucket == "external":
        datasets= http_get_external_dataset_list()
    elif bucket == "extracts":
        datasets= http_get_extract_dataset_list()
    else:
        datasets= http_get_dataset_names()
    
    return datasets


class ClipBatchCaculation:
    def __init__(self,
                 minio_client,
                 bucket,
                 dataset,
                 device,
                 batch_size=10000):
        
        # parameters
        self.minio_client= minio_client
        self.bucket= bucket
        self.dataset = dataset
        self.device= device
        self.batch_size= batch_size

        # clip model
        self.clip = KandinskyCLIPImageEncoder(device= self.device)
        self.clip.load_submodels()

    def load_file_paths(self):
        print("Loading paths for each clip vector")

        if self.bucket == "datasets":
            image_data = http_get_completed_job_by_dataset(dataset= self.dataset)
        elif self.bucket == "external":
            image_data = http_get_external_image_list(dataset= self.dataset)
        elif self.bucket == "extracts":
            image_data = http_get_extract_image_list(dataset= self.dataset)

        file_paths= [image['file_path'] for image in image_data]
        uuids= [image['uuid'] for image in image_data]

        return file_paths, uuids
    
    def load_clip_vectors(self):
        # loading file paths and uuids of all jobs in the dataset
        file_paths, uuids= self.load_file_paths()

        print(f"Loading clip vectors for the {self.bucket}/{self.dataset} dataset")
        clip_batch= []
        batch_num= 1
        for file_path, uuid in tqdm(zip(file_paths, uuids)):
            try:
                bucket_name, input_file_path = separate_bucket_and_file_path(file_path)
                file_path = os.path.splitext(input_file_path)[0]

                if bucket_name == "extracts":
                    output_clip_path = file_path + "_clip-h.msgpack"
                else:
                    output_clip_path = file_path + "_clip_kandinsky.msgpack"
                    
                features_data = cmd.get_file_from_minio(self.minio_client, self.bucket, output_clip_path)
                features_vector = msgpack.unpackb(features_data)["clip-feature-vector"]

                clip_batch.append({"uuid": uuid, "clip_vector": features_vector})

                if len(clip_batch) == self.batch_size:
                    print(f"Storing a batch")

                    output_folder = f"{self.dataset}/clip_vectors/{str(batch_num).zfill(4)}"
                    data_path = output_folder + "_clip_data.msgpack"
                    # Save the new data directly as the start of a new batch
                    save_data_to_minio(self.minio_client, self.bucket, data_path, clip_batch)
                    batch_num+=1
                    clip_batch = []

            except Exception as e:
                print(f"An error occured {e}")
    

def main():
    args= parse_args()

    # get minio client
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                            minio_secret_key=args.minio_secret_key)
    
    # get device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    device = torch.device(device)

    if args.dataset == "all":

        dataset_names= get_dataset_list()

        for dataset in dataset_names:
            # initialize image extraction pipeline
            pipeline= ClipBatchCaculation(minio_client= minio_client,
                                        device=device,
                                        bucket= args.bucket,
                                        dataset=dataset,
                                        batch_size= args.batch_size)

            pipeline.load_clip_vectors()
    else:
        # initialize image extraction pipeline
        pipeline= ClipBatchCaculation(minio_client= minio_client,
                                    device=device,
                                    bucket= args.bucket,
                                    dataset=args.dataset,
                                    batch_size= args.batch_size)

        pipeline.load_clip_vectors()

if __name__ == "__main__":
    main()