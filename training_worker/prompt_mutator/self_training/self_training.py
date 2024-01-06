import argparse
import io
import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import msgpack

base_directory = "./"
sys.path.insert(0, base_directory)

from training_worker.prompt_mutator.models.substitution_ranking_xgboost import XgboostSubstitutionModel
from training_worker.prompt_mutator.models.substitution_classification_xgboost import XgboostSubstitutionClassifier
from training_worker.prompt_mutator.models.substitution_ranking_linear import LinearSubstitutionModel
from utility.minio import cmd

DATA_MINIO_DIRECTORY="environmental/data/prompt-generator/"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--model-type', help="Model type (xgboost ,linear etc..)", default="xgboost")
    parser.add_argument('--scoring-model', help="scoring model to do self training on (elm,linear etc..)", default="linear")
    parser.add_argument('--operation', help="operation to do self training for (substitution, permutation etc...)", default="substitution")
    parser.add_argument('--embedding-type', help='type of embedding, positive or negative', default='positive')
    parser.add_argument('--output-type', help='type of output for the prompt mutator model', default="sigma_score")
    parser.add_argument('--scaling-graph', action='store_true', default=False)
    args = parser.parse_args()
    return args

class SelfTrainingPromptMutator:
    def __init__(
        self,
        minio_access_key,
        minio_secret_key,
        minio_ip_addr,
        model_type,
        scoring_model,
        operation,
        embedding_type,
        output_type,
        scaling_graph
    ):
        # get minio client
        self.minio_client = cmd.get_minio_client(minio_access_key,
                                            minio_secret_key,
                                            minio_ip_addr)
        
        self.scoring_model=scoring_model
        self.operation=operation
        self.embedding_type=embedding_type
        self.output_type= output_type
        self.model_type= model_type
        self.scaling_graph=scaling_graph

        if(self.model_type=="xgboost"):
            if(self.output_type=="binary"):
                self.model= XgboostSubstitutionClassifier(minio_client=self.minio_client, ranking_model=self.scoring_model, operation=self.operation, prompt_type=self.embedding_type)
            else:
                self.model= XgboostSubstitutionModel(minio_client=self.minio_client, output_type=self.output_type, ranking_model=self.scoring_model, operation=self.operation, prompt_type=self.embedding_type)
        elif(self.model_type=="linear"):
            self.model= LinearSubstitutionModel(minio_client=self.minio_client, input_size=2306, output_type=self.output_type, ranking_model=self.scoring_model, operation=self.operation, prompt_type=self.embedding_type)


    def get_training_data(self):
        dataset_path=DATA_MINIO_DIRECTORY + f"{self.operation}/{self.embedding_type}_prompts/"
        dataset_files=self.minio_client.list_objects('datasets', prefix=dataset_path, recursive=True)
        dataset_files= [file.object_name for file in dataset_files]
        
        inputs=[]
        outputs=[]

        for file in dataset_files:
            print(file)
            # get prompt embedding
            data = self.minio_client.get_object('datasets', file)
            # Read the content of the msgpack file
            content = data.read()

            # Deserialize the content using msgpack
            msgpack_data = msgpack.loads(content)

            # get input
            if self.operation=="substitution":
                input=np.concatenate([msgpack_data['input'], [msgpack_data['position_encoding']], [msgpack_data[f'{self.scoring_model}_score_encoding']]]).tolist()
            elif self.operation=="permutation":
                input=np.concatenate([msgpack_data['input'],
                                                [msgpack_data['first_position']],
                                                [msgpack_data['second_position']],
                                        [msgpack_data[f'{self.scoring_model}_score_encoding']]]).tolist()
                
            inputs.append(input)
            
            if(self.output_type=="binary"):
                # get binary output
                if msgpack_data[f'{self.scoring_model}_score_encoding']> msgpack_data[f'{self.scoring_model}_output'] :
                    binary_linear_output="decrease"
                else:
                    binary_linear_output="increase"

                outputs.append(binary_linear_output)
            
            elif(self.output_type=="sigma_score"):
                # get sigma output
                sigma_score=msgpack_data[f'{self.scoring_model}_output']
                outputs.append(sigma_score)
            elif(self.output_type=="delta_score"):
                # get delta output
                delta_score= msgpack_data[f'{self.scoring_model}_output'] - msgpack_data[f'{self.scoring_model}_score_encoding']
                outputs.append(delta_score)
            
        return inputs, outputs
    
    def self_training(self):
        # get training data
        inputs, outputs= self.get_training_data()

        # get self training data
        self_training_path = DATA_MINIO_DIRECTORY + f"{self.operation}/self_training/{self.scoring_model}/"
        self_training_files = self.minio_client.list_objects('datasets', prefix=self_training_path, recursive=True)
        self_training_files = [file.object_name for file in self_training_files]

        # save loss
        loss_by_data=[]

        for file in self_training_files:
            print(file)
            # save loss to track performance by data
            if(self.scaling_graph):
                loss_by_data.append(self.model.train(inputs, outputs))

            # get prompt embedding
            data = self.minio_client.get_object('datasets', file)
            # Read the content of the msgpack file
            content = data.read()

            # Deserialize the content using msgpack
            self_training_data = msgpack.loads(content)
            
            # append the self training data to list of data
            self_training_inputs, self_training_outputs= self.load_self_training_data(self_training_data)
            inputs.extend(self_training_inputs)
            outputs.extend(self_training_outputs)
        
        # training and saving the model
        loss=self.model.train(inputs, outputs)
        self.model.save_model()

        # out put a graph for scaling law
        if(self.scaling_graph):
            loss_by_data.append(loss)
            self.save_scaling_graph(loss_by_data=loss_by_data, num_datapoints=len(self_training_files))
    
    def load_self_training_data(self, data):
        inputs=[]
        outputs=[]
        for d in data:
            input=np.concatenate([d['input'], [d['position_encoding']], [d['score_encoding']]]).tolist()
            inputs.append(input)
            outputs.append(d['output'])
        
        return inputs, outputs
    
    def save_scaling_graph(self, loss_by_data, num_datapoints):
        plt.plot([i * 10000 for i in range(num_datapoints + 1)], loss_by_data)
        plt.xlabel('Self Training Data', fontsize=14)
        plt.ylabel('L1 Loss', fontsize=14)
        plt.title('L1 Loss by Number of Self Training Datapoints', fontsize=16)

        # Set y-axis limits to start from 0 and end at the max value in loss_by_data
        plt.ylim(0, max(loss_by_data))

        # Save the figure to a file
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # upload the graph report
        minio_path= DATA_MINIO_DIRECTORY + f"{self.operation}/scaling_graphs/{self.scoring_model}/scaling_graph.png"
        cmd.upload_data(self.minio_client, 'datasets', minio_path, buf)
        # Remove the temporary file
        os.remove("output/scaling_graph.png")
        # Clear the current figure
        plt.clf()
    
def main():
    args = parse_args()

    model_trainer=SelfTrainingPromptMutator(minio_access_key=args.minio_access_key,
                                minio_secret_key=args.minio_secret_key,
                                minio_ip_addr=args.minio_addr,
                                model_type=args.model_type,
                                scoring_model=args.scoring_model,
                                operation=args.operation,
                                embedding_type=args.embedding_type,
                                output_type=args.output_type,
                                scaling_graph=args.scaling_graph)
    
    # do self training
    model_trainer.self_training()

if __name__ == "__main__":
    main()
