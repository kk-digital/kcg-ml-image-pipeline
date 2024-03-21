
import argparse
import sys

base_directory = "./"
sys.path.insert(0, base_directory)
from training_worker.sampling.models.sampling_fc import SamplingFCNetwork, SamplingType
from utility.minio import cmd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--minio-addr', type=str, help="Minio Address")
    parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")
    parser.add_argument('--target-avg-points', type=int, help='Target average of datapoints per sphere', 
                        default=5)
    parser.add_argument('--n-spheres', type=int, help='Number of spheres', default=100000)
    parser.add_argument('--training-batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--output-size', type=int, default=8)
    parser.add_argument('--bin-size', type=int, default=1)

    return parser.parse_args()

def main():
    args= parse_args()

    # get minio client
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)

    gaussian_sampling_model= SamplingFCNetwork(minio_client=minio_client, 
                                              dataset=args.dataset,
                                              output_size= args.output_size,
                                              bin_size= args.bin_size,
                                              type=SamplingType.UNIFORM_SAMPLING)

    gaussian_sampling_model.train(num_epochs=args.epochs,
                                 batch_size=args.training_batch_size,
                                 learning_rate= args.learning_rate,
                                 n_spheres=args.n_spheres, 
                                 target_avg_points=args.target_avg_points)
    
    gaussian_sampling_model.save_model()
    
if __name__ == "__main__":
    main()