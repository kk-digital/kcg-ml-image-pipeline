
import argparse
import sys

base_directory = "./"
sys.path.insert(0, base_directory)
from training_worker.sampling.models.uniform_sampling_fc import SamplingFCNetwork
from training_worker.sampling.models.uniform_sampling_regression_fc import SamplingFCRegressionNetwork
from training_worker.sampling.models.directional_uniform_sampling_regression_fc import DirectionalSamplingFCRegressionNetwork
from utility.minio import cmd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--dataset', type=str, help='Name of the dataset', default="environmental")
    parser.add_argument('--target-avg-points', type=int, help='Target average of datapoints per sphere', 
                        default=16)
    parser.add_argument('--sphere-type', type=str, help='Type of spheres, uniform or directional', default="uniform")
    parser.add_argument('--n-spheres', type=int, help='Number of spheres', default=100000)
    parser.add_argument('--generate-every-epoch', action="store_true", help='Generate spheres each epoch', default=False)
    parser.add_argument('--train-residual-model', action="store_true", help='Train residual models', default=False)
    parser.add_argument('--training-batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--output-type', type=str, help="variance, mean_sigma_score or score_distribution", default="mean_sigma_score")
    parser.add_argument('--output-size', type=int, default=8)
    parser.add_argument('--bin-size', type=int, default=1)

    return parser.parse_args()

def main():
    args= parse_args()

    # get minio client
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key)

    if args.output_type in ["score_distribution"]:
        uniform_sampling_model= SamplingFCNetwork(minio_client=minio_client, 
                                                dataset=args.dataset,
                                                output_size= args.output_size,
                                                output_type= args.output_type,
                                                bin_size= args.bin_size)
    elif args.output_type in ["variance", "mean_sigma_score"]:
        if args.sphere_type=="uniform":
            uniform_sampling_model= SamplingFCRegressionNetwork(minio_client=minio_client, 
                                                dataset=args.dataset,
                                                output_type= args.output_type)
        elif args.sphere_type=="directional":
            uniform_sampling_model= DirectionalSamplingFCRegressionNetwork(minio_client=minio_client, 
                                                dataset=args.dataset,
                                                output_type= args.output_type)
        
    uniform_sampling_model.train(num_epochs=args.epochs,
                                 batch_size=args.training_batch_size,
                                 learning_rate= args.learning_rate,
                                 n_spheres=args.n_spheres, 
                                 target_avg_points=args.target_avg_points,
                                 generate_every_epoch= args.generate_every_epoch,
                                 train_residual_model= args.train_residual_model)
    
    uniform_sampling_model.save_model()
    
if __name__ == "__main__":
    main()