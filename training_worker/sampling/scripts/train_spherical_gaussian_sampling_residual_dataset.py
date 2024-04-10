
import argparse
import sys

base_directory = './'
sys.path.insert(0, base_directory)
from training_worker.sampling.models.gaussian_sampling_fc import SamplingFCNetwork
from training_worker.sampling.models.gaussian_sampling_regression_fc import SamplingFCRegressionNetwork
from training_worker.sampling.models.directional_gaussian_sampling_regression_fc import DirectionalGuassianResidualFCNetwork, DirectionalSamplingResidualXGBoost
from utility.minio import cmd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--minio-addr', type=None, help='Minio Address')
    parser.add_argument('--dataset', type=str, help='Name of the dataset', default='environmental')
    parser.add_argument('--target-avg-points', type=int, help='Target average of datapoints per sphere', 
                        default=15)
    parser.add_argument('--percentile', type=int, default=75, help='Percentile for spherical gaussian')
    parser.add_argument('--std', type=float, default=1.0, help='Standard deviation for spherical gaussian')
    parser.add_argument('--n-spheres', type=int, help='Number of spheres', default=100000)
    parser.add_argument('--training-batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--output-size', type=int, default=8)
    parser.add_argument('--output-type', type=str, default='mean_sigma_score', help='mean_sigma_score')
    parser.add_argument('--input-type', type=str, default='gaussian_sphere_variance', help='Input type - gaussian_sphere_variance, gaussian_sphere_sigma, gaussian_sphere_fall_off')
    parser.add_argument('--bin-size', type=int, default=1)
    parser.add_argument('--per-epoch', action='store_true', default=False, help='Generate the random spheres per epoch')
    parser.add_argument('--sphere-type', type=str, default='directional', help='directional')
    parser.add_argument('--model-type', type=str, default='FCNetwork', help='FCNetwork, XGBoost')
    return parser.parse_args()

def main():
    args= parse_args()

    # get minio client
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)


    if args.output_type in ['score_distribution']:
        pass
    elif args.output_type in ['variance', 'mean_sigma_score']:
        if args.sphere_type == 'spherical':
            pass
        elif args.sphere_type == 'directional':
            if args.model_type == 'FCNetwork':
                gaussian_sampling_model = DirectionalGuassianResidualFCNetwork(minio_client=minio_client, 
                                                dataset=args.dataset,
                                                input_type=args.input_type,
                                                output_type= args.output_type)
            elif args.model_type == 'XGBoost':
                gaussian_sampling_model = DirectionalSamplingResidualXGBoost(minio_client=minio_client, 
                                                dataset=args.dataset,
                                                input_type=args.input_type,
                                                output_type= args.output_type)
                
    gaussian_sampling_model.set_config(sampling_parameter={'percentile': args.percentile, 'std': args.std})

    gaussian_sampling_model.train(num_epochs=args.epochs,
                                 batch_size=args.training_batch_size,
                                 learning_rate= args.learning_rate,
                                 n_spheres=args.n_spheres, 
                                 target_avg_points=args.target_avg_points,
                                 is_per_epoch=args.per_epoch, 
                                 model_name="Residual prediction model on directional gaussian sampling")
    
    gaussian_sampling_model.save_model()
    
if __name__ == '__main__':
    main()