import argparse
import sys
import msgpack
import torch
import faiss

base_directory = './'
sys.path.insert(0, base_directory)

from training_worker.sampling.models.uniform_sampling_regression_fc import SamplingFCRegressionNetwork
from utility.minio import cmd
from data_loader.utils import get_object
import numpy as np

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--minio-addr', type=str, help='Minio address')
    parser.add_argument('--n-spheres', type=int, help='number of spheres')
    parser.add_argument('--dataset', type=str, default='', help='Dataset used to generate images')

    return parser.parse_args()


def get_top_k(features, centers, k, d, nlist = 1):
    quantizer = faiss.IndexFlatL2(nlist)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.train(features)
    index.add(features)

    distances, indices = index.search(centers, k)

    return distances, indices


def get_distribution_info(minio_client, dataset, device):
        data = get_object(minio_client, f"{dataset}/output/stats/clip_stats.msgpack")
        data_dict = msgpack.unpackb(data)

        mean_vector = torch.tensor(data_dict["mean"]).to(device=device, dtype=torch.float32)
        std_vector = torch.tensor(data_dict["std"]).to(device=device, dtype=torch.float32)
        max_vector = torch.tensor(data_dict["max"]).to(device=device, dtype=torch.float32)
        min_vector = torch.tensor(data_dict["min"]).to(device=device, dtype=torch.float32)

        return mean_vector, std_vector, max_vector, min_vector


def main():
    args = parse_args()

    minio_client = cmd.get_minio_client(args.minio_access_key, args.minio_secret_key, args.minio_addr)
    print("before model loading...")
    # get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load model
    model = SamplingFCRegressionNetwork(minio_client)
    model.load_model()

    print("successfully loaded model")
    # get distribution information of given dataset
    mean_vector, std_vector, max_vector, min_vector = get_distribution_info(minio_client, args.dataset, device)
    max_vector= np.array(max_vector, dtype='float32')
    min_vector= np.array(min_vector, dtype='float32')
    # Generate random values between 0 and 1, then scale and shift them into the [min, max] range for each feature
    print("generating the initial spheres-------------")
    
    sphere_centers = np.random.rand(args.n_spheres, len(max_vector)) * (max_vector - min_vector) + min_vector
    # Convert sphere_centers to float32
    sphere_centers = sphere_centers.astype('float32')

    print("sphere centers dim", sphere_centers.shape)

    max_radius = torch.norm(max_vector - min_vector).item()
    radius_vector = np.random.rand(args.n_spheres) * max_radius
    radius_vector = radius_vector.reshape(-1, 1)

    print("sphere radius dim", radius_vector.shape)

    feature_vector_list = torch.concat((sphere_centers, radius_vector), dim=0)
    print("model", )
    predictions = model.predict(feature_vector_list)

    center = [0]

    distances, indices = get_top_k(predictions, center, 10, predictions.shape[1])    
    print(distances, indices)


if __name__ == "__main__":
    main()