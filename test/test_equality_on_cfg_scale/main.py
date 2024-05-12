import sys
import msgpack
import torch
import argparse

base_dir = './'
sys.path.insert(0, base_dir)

from data_loader.utils import get_object
from kandinsky_worker.image_generation.img2img_generator import generate_img2img_generation_jobs_with_kandinsky
from utility.minio import cmd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--dataset', type=str, default='environmental')
    parser.add_argument('--cfg-scale', type=int, default=0)

    return parser.parse_args()

def get_clip_distribution(minio_client, dataset):
    

    data = get_object(minio_client, f"{dataset}/output/stats/clip_stats.msgpack")
    data_dict = msgpack.unpackb(data)

    # Convert to PyTorch tensors
    mean_vector = torch.tensor(data_dict["mean"][0], dtype=torch.float32)
    std_vector = torch.tensor(data_dict["std"][0], dtype=torch.float32)
    max_vector = torch.tensor(data_dict["max"][0], dtype=torch.float32)
    min_vector = torch.tensor(data_dict["min"][0], dtype=torch.float32)

    return mean_vector, std_vector, max_vector, min_vector

def main():

    args = parse_args()
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key, 
                                        minio_secret_key=args.minio_secret_key)

    mean_vector, _, _, _ = get_clip_distribution(minio_client=minio_client, dataset=args.dataset)

    response= generate_img2img_generation_jobs_with_kandinsky(
        image_embedding=mean_vector,
        negative_image_embedding=None,
        dataset_name="test-generations",
        prompt_generation_policy='test-equality-on-different-cfg-scales',
        decoder_guidance_scale=0,
        self_training=True
    )

    print('Successfully generated jobs')

if __name__ == '__main__':
    main()