import sys
import msgpack
import torch

base_dir = './'
sys.path.insert(base_dir)

from data_loader.utils import get_object
from kandinsky_worker.image_generation.img2img_generator import generate_img2img_generation_jobs_with_kandinsky


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
    mean_vector = get_clip_distribution()

    response= generate_img2img_generation_jobs_with_kandinsky(
        image_embedding=mean_vector.unsqueeze(0),
        negative_image_embedding=None,
        dataset_name="test-generations",
        prompt_generation_policy='test-equality-on-different-cfg-scales',
        decoder_guidance_scale=0,
        self_training=True
    )

if __name__ == '__main__':
    main()