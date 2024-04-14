import argparse
import os
import sys
import torch

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from data_loader.kandinsky_dataset_loader import KandinskyDatasetLoader
from training_worker.scoring.models.classifier_fc import ClassifierFCNetwork
from utility.http import request
from utility.minio import cmd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--tag-name', type=str, help='Name of the tag to generate for', default="topic-forest")
    parser.add_argument('--classifier-batch-size', type=int, default=1000)
    parser.add_argument('--training-batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--num-samples', type=int, default=10000)

    return parser.parse_args()

class ABRankingFcTrainingPipeline:
    def __init__(self,
                    minio_access_key,
                    minio_secret_key,
                    tag_name,
                    classifier_batch_size=1000,
                    training_batch_size=256,
                    learning_rate=0.001,
                    epochs=10):
        
        # get minio client
        self.minio_client = cmd.get_minio_client(minio_access_key=minio_access_key,
                                            minio_secret_key=minio_secret_key)
        
        # get device
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = torch.device(device)
        
        self.tag_name= tag_name
        self.training_batch_size= training_batch_size
        self.classifier_batch_size= classifier_batch_size
        self.learning_rate= learning_rate
        self.epochs= epochs

        self.dataloader= KandinskyDatasetLoader(minio_client=self.minio_client,
                                                dataset="environmental")

    def train(self):
        # load the training dataset
        inputs, outputs= self.dataloader.load_classifier_scores(self.tag_name, batch_size=self.classifier_batch_size)
        
        # training and saving the model
        print(f"training an fc model for the {self.tag_name} tag")
        model= ClassifierFCNetwork(minio_client=self.minio_client, tag_name=self.tag_name)
        loss=model.train(inputs, outputs, num_epochs= self.epochs, batch_size=self.training_batch_size, learning_rate=self.learning_rate)
        model.save_model()

def main():
    args = parse_args()

    if args.tag_name != "all":
        training_pipeline=ABRankingFcTrainingPipeline(minio_access_key=args.minio_access_key,
                                    minio_secret_key=args.minio_secret_key,
                                    tag_name= args.tag_name,
                                    classifier_batch_size=args.classifier_batch_size,
                                    training_batch_size=args.training_batch_size,
                                    epochs= args.epochs,
                                    learning_rate= args.learning_rate)
        
        # do self training
        training_pipeline.train()
    
    else:
        # if all, train models for all existing tags
        # get tag name list
        tag_names = request.http_get_tag_list()
        print("tag names=", tag_names)
        for tag_name in tag_names:  
            try:
                # initialize training pipeline
                training_pipeline=ABRankingFcTrainingPipeline(minio_access_key=args.minio_access_key,
                                    minio_secret_key=args.minio_secret_key,
                                    tag_name= tag_name,
                                    classifier_batch_size=args.classifier_batch_size,
                                    training_batch_size=args.training_batch_size,
                                    epochs= args.epochs,
                                    learning_rate= args.learning_rate)
                
                # Train the model
                training_pipeline.train()

            except Exception as e:
                print("Error training model for {}: {}".format(tag_name, e))

if __name__ == "__main__":
    main()
