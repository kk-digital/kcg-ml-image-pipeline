import argparse
from datetime import datetime
import io
import os
import random
import sys
import time
import traceback
from xmlrpc.client import ResponseError
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import msgpack

base_directory = "./"
sys.path.insert(0, base_directory)

from training_worker.prompt_mutator.prompt_mutator_model import PromptMutator
from training_worker.prompt_mutator.binary_prompt_mutator import BinaryPromptMutator
from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from training_worker.ab_ranking.model.ab_ranking_linear import ABRankingModel
from stable_diffusion.model.clip_text_embedder.clip_text_embedder import CLIPTextEmbedder
from utility.minio import cmd
from worker.prompt_generation.prompt_generator import load_base_prompts, generate_image_generation_jobs

GENERATION_POLICY="prompt-tree-search-v1"
DATA_MINIO_DIRECTORY="environmental/data/prompt-generator/addition"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--csv-phrase', help='CSV containing phrases, must have "phrase str" column', default='input/civitai_phrases_database_v7_no_nsfw.csv')
    parser.add_argument('--n-data', type=int, help='Number of data samples to generate', default=20)
    parser.add_argument('--send-job', action='store_true', default=False)
    parser.add_argument('--dataset-name', default='test-generations')
    parser.add_argument('--scoring-model', help="elm or linear", default="linear")
    parser.add_argument('--sigma-threshold', type=float, help="threshold of rejection policy for increase of sigma score", default=0.1)
    parser.add_argument('--max-iterations', type=int, help="number of mutation iterations", default=80)
    parser.add_argument('--self-training', action='store_true', default=False)
    parser.add_argument('--store-embeddings', action='store_true', default=False)
    parser.add_argument('--save-csv', action='store_true', default=False)
    parser.add_argument(
        '--csv_base_prompts', help='CSV containing base prompts', 
        default='input/dataset-config/environmental/base-prompts-environmental.csv'
    )

    return parser.parse_args()

class PromptTreeSearchGenerator:
    def __init__(
        self,
        minio_access_key,
        minio_secret_key,
        minio_ip_addr,
        csv_phrase,
        csv_base_prompts,
        scoring_model,
        max_iterations,
        sigma_threshold,
        dataset_name,
        store_embeddings,
        self_training,
        send_job,
        save_csv
    ):
        start=time.time()

        # parameters
        # csv file containing civitai phrases
        self.csv_phrase=csv_phrase
        # the scoring model used for prompt generation (elm or linear)
        self.scoring_model= scoring_model
        # number of iterations for prompt generation
        self.max_iterations= max_iterations
        # average score by iteration to track score improvement
        self.average_score_by_iteration=np.zeros(self.max_iterations)
        # rejection threshold for increase in sigma score
        self.sigma_threshold= sigma_threshold
        # name of dataset
        self.dataset_name=dataset_name
        # wheher to self training or not
        self.self_training=self_training
        # whether to send jobs to server or not
        self.send_job=send_job
        # whether to save csv of prompts or not
        self.save_csv=save_csv
        # addition model (binary of sigma score)
        self.addition_model= None
        # number of choices per iteration
        self.choices_per_iteration= 64
        # get list of base prompts
        self.csv_base_prompts=csv_base_prompts

        # get minio client
        self.minio_client = cmd.get_minio_client(minio_access_key,
                                            minio_secret_key,
                                            minio_ip_addr)
        
        # get device
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = torch.device(device)

        # Load the clip embedder model
        self.embedder=CLIPTextEmbedder(device=device)
        self.embedder.load_submodels()

        # load the scoring models (for positive prompts and for both)
        self.positive_scorer= self.load_model(embedding_type='positive', scoring_model=self.scoring_model)
        self.scorer= self.load_model(embedding_type='combined', scoring_model=self.scoring_model, input_size=768*2)

        # get mean and std values
        self.mean, self.std= self.scorer.mean, self.scorer.standard_deviation
        self.positive_mean, self.positive_std= self.positive_scorer.mean, self.positive_scorer.standard_deviation
        
        # load the xgboost model
        self.addition_model= PromptMutator(minio_client=self.minio_client, ranking_model=self.scoring_model, operation="addition")
        self.addition_model.load_model()

        # store phrase embeddings in a file in minio 
        if(store_embeddings):
            self.store_phrase_embeddings()
        
        # get list of phrases and their embeddings
        phrase_df=pd.read_csv(csv_phrase).sort_values(by="index")
        self.phrase_list=phrase_df['phrase str'].tolist()
        self.phrase_embeddings= self.load_phrase_embeddings()

        end=time.time()
        # log the loading time
        self.loading_time= end-start
        
    # load elm or linear scoring models
    def load_model(self, embedding_type, scoring_model="linear", input_size=768):
        input_path="environmental/models/ranking/"

        if(scoring_model=="elm"):
            embedding_model = ABRankingELMModel(input_size)
            file_name=f"score-elm-v1-embedding"
        else:
            embedding_model= ABRankingModel(input_size)
            file_name=f"score-linear-embedding"
        
        if(embedding_type=="positive" or embedding_type=="negative"):
            file_name+=f"-{embedding_type}.pth"
        else:
            file_name+=".pth"

        model_files=cmd.get_list_of_objects_with_prefix(self.minio_client, 'datasets', input_path)
        most_recent_model = None

        for model_file in model_files:
            if model_file.endswith(file_name):
                most_recent_model = model_file

        if most_recent_model:
            model_file_data =cmd.get_file_from_minio(self.minio_client, 'datasets', most_recent_model)
        else:
            print("No .pth files found in the list.")
            return
        
        print(most_recent_model)

        # Create a BytesIO object and write the downloaded content into it
        byte_buffer = io.BytesIO()
        for data in model_file_data.stream(amt=8192):
            byte_buffer.write(data)
        # Reset the buffer's position to the beginning
        byte_buffer.seek(0)

        embedding_model.load(byte_buffer)
        embedding_model.model=embedding_model.model.to(self.device)

        return embedding_model

    # get the clip text embedding of a prompt or a phrase
    def get_prompt_embedding(self, prompt):
        with torch.no_grad():
            embedding= self.embedder(prompt)

        embedding= embedding.unsqueeze(0)
        embedding=embedding.to(self.device)

        return embedding

    # get linear or elm score of an embedding
    def get_prompt_score(self, embedding):
        with torch.no_grad():
            prompt_score=self.positive_scorer.predict_positive_or_negative_only(embedding)
        
        return prompt_score.item()

    # get the mean pool of an embedding
    def get_mean_pooled_embedding(self, embedding):
        embedding=torch.mean(embedding, dim=2)
        embedding = embedding.reshape(len(embedding), -1).squeeze(0)

        return embedding.detach().cpu().numpy()

    # get paths for embeddings of all prompts in a dataset
    def get_embedding_paths(self, dataset):
            objects=self.minio_client.list_objects('datasets', dataset, recursive=True)
            embedding_files = []
            for obj in objects: 
                if obj.object_name.endswith("_embedding.msgpack"):
                    embedding_files.append(obj.object_name)
                    
            return embedding_files

    # store self training data
    def store_self_training_data(self, training_data):
        batch_size = 10000
        dataset_path = DATA_MINIO_DIRECTORY + "/self_training/"
        dataset_files = self.minio_client.list_objects('datasets', prefix=dataset_path, recursive=True)
        dataset_files = [file.object_name for file in dataset_files]

        batch = []  # Accumulate training data points until the batch size is reached

        if(len(dataset_files)==0):
            index=1
        else:
            last_file_path=dataset_files[len(dataset_files)-1]
            # Read the content of the last unfinished file
            if last_file_path.endswith("_incomplete.msgpack"):
                data = self.minio_client.get_object('datasets', last_file_path)
                content = data.read()
                batch = msgpack.loads(content)
                index = len(dataset_files)
                self.minio_client.remove_object('datasets', last_file_path)
            else:
                index= len(dataset_files) + 1

        for data in training_data:
            batch.append(data)

            if len(batch) == batch_size:
                self.store_batch_in_msgpack_file(batch, index)
                index += 1
                batch = []  # Reset the batch for the next file

        # If there are remaining data points not reaching the batch size, store them
        if batch:
            self.store_batch_in_msgpack_file(batch, index, incomplete=True)

    # function for storing self training data in a msgpack file
    def store_batch_in_msgpack_file(self, batch, index, incomplete=False):
        if incomplete:
            file_path=f"{self.scoring_model}/{str(index).zfill(4)}_substitution_incomplete.msgpack"
        else:
            file_path=f"{self.scoring_model}/{str(index).zfill(4)}_substitution.msgpack"
        packed_data = msgpack.packb(batch, use_single_float=True)

        local_file_path = f"output/temporary_file.msgpack"
        with open(local_file_path, 'wb') as local_file:
            local_file.write(packed_data)

        with open(local_file_path, 'rb') as file:
            content = file.read()

        buffer = io.BytesIO(content)
        buffer.seek(0)

        minio_path = DATA_MINIO_DIRECTORY + f"/self_training/{file_path}"
        cmd.upload_data(self.minio_client, 'datasets', minio_path, buffer)

        os.remove(local_file_path)

    # store embeddings of all phrases in civitai in a file in minIO
    def store_phrase_embeddings(self):
        phrase_list=pd.read_csv(self.csv_phrase)
        phrase_list= phrase_list.sort_values(by="index")
        phrase_embeddings_list=[]
        
        for index, row in phrase_list.iterrows():
            print(f"storing phrase {row['index']}")
            embedding= self.get_prompt_embedding(row['phrase str'])
            mean_pooled_embedding= self.get_mean_pooled_embedding(embedding)
            phrase_embeddings_list.append(mean_pooled_embedding)
        
        # Convert the list of numpy arrays to a 2D numpy array
        phrase_embeddings = np.array(phrase_embeddings_list)

        # Save the numpy array to an .npz file
        local_file_path='phrase_embeddings.npz'
        np.savez_compressed(local_file_path, phrase_embeddings)

        # Read the contents of the .npz file
        with open(local_file_path, 'rb') as file:
            content = file.read()

        # Upload the local file to MinIO
        buffer = io.BytesIO(content)
        buffer.seek(0)

        minio_path=DATA_MINIO_DIRECTORY + f"/input/phrase_embeddings.npz"
        cmd.upload_data(self.minio_client, 'datasets',minio_path, buffer)

        # Remove the temporary file
        os.remove(local_file_path)

    # get civitai phrase embeddings from minIO
    def load_phrase_embeddings(self):
        # Get the file data from MinIO
        minio_path = "environmental/data/prompt-generator/substitution/input/phrase_embeddings.npz"
        file_data = cmd.get_file_from_minio(self.minio_client, 'datasets', minio_path)

        # Create a BytesIO object and write the downloaded content into it
        byte_buffer = io.BytesIO()
        for data in file_data.stream(amt=8192):
            byte_buffer.write(data)
        # Reset the buffer's position to the beginning
        byte_buffer.seek(0)

        # Load the compressed numpy array from the BytesIO object
        with np.load(byte_buffer) as data:
            phrase_embeddings = data['arr_0']

        return phrase_embeddings

    # function for rejection sampling with score
    def rejection_sampling_by_score(self, 
                                    prompt_str, 
                                    prompt_score, 
                                    prompt_embedding
                                    ):

        # get list of phrases
        prompt_list = prompt_str.split(', ')
        token_number= len(prompt_list)
        # list of potential addition choices for current iteration
        addition_choices=[]
        
        # Create a batch of addition inputs
        batch_addition_inputs = []
        sampled_phrases = []
        sampled_embeddings = []
        positions=[]

        # create a substitution for each position in the prompt
        for i in range(self.choices_per_iteration):
            # get a random phrase from civitai to add
            random_index=random.randrange(0, len(self.phrase_list))
            # get phrase string
            added_phrase = self.phrase_list[random_index]
            # get phrase embedding by its index
            added_embedding = self.phrase_embeddings[random_index]
            # choose a random position to add into
            addition_position=random.randrange(0,token_number+1)
            position_encoding=addition_position / token_number
            # concatenate input in one array to use for inference
            addition_input = np.concatenate([prompt_embedding, added_embedding, [position_encoding]])
            # save data in an array to use for inference and rejection sampling
            batch_addition_inputs.append(addition_input)
            sampled_phrases.append(added_phrase)
            sampled_embeddings.append(added_embedding)
            positions.append(addition_position)
        

        # Predict probabilities of increase and decrease for every addition
        batch_preds = self.addition_model.predict(batch_addition_inputs)

        print(prompt_score)
        print(f"predicted scores: {batch_preds}")

        # filter with rejection sampling
        for choice, sigma_score in enumerate(batch_preds):
            # only take substitutions that have more than 66% chance to increase score
            if sigma_score > prompt_score:
                print(positions[choice])
                addition_data={
                    'position':positions[choice],
                    'added_phrase':sampled_phrases[choice],
                    'added_embedding':sampled_embeddings[choice],
                    'score':sigma_score
                }
                addition_choices.append(addition_data)

        # additions are sorted from highest predicted score to lowest
        addition_choices= sorted(addition_choices, key=lambda s: s['score'], reverse=True) 
        
        return addition_choices

    # function mutating a prompt
    def generate_prompt(self):
        # get base prompt
        base_prompt_population = load_base_prompts(self.csv_base_prompts)
        prompt_str=random.choice(base_prompt_population)
        prompt_embedding=self.get_prompt_embedding(prompt_str)
        prompt_score=self.get_prompt_score(prompt_embedding)
        prompt_score= (prompt_score - self.positive_mean)/self.std

        # self training datapoints
        self_training_data=[]
        rejection_policy_time=0
        addition_time=0
        num_attempts=0
        num_success=0
        
        # run mutation process for a set number of iterations
        for i in range(self.max_iterations):
            # get pooled embedding of the prompt
            pooled_prompt_embedding=self.get_mean_pooled_embedding(prompt_embedding)
            
            start= time.time()
            # return a list of potential substitution choices, filtered by the rejection policy
            addition_choices=self.rejection_sampling_by_score(
                                                prompt_str,
                                                prompt_score,
                                                pooled_prompt_embedding)
            end= time.time()

            rejection_policy_time+= end - start
            
            start= time.time()
            # test every choice and take the first choice that increases score
            for addition in addition_choices:
                # get addition data
                position=addition['position']
                added_phrase=addition['added_phrase']
                added_embedding=addition['added_embedding']
                predicted_score=addition['score']

                #Create a modified prompt with the addition
                prompt_list = prompt_str.split(', ')
                prompt_list.insert(position, added_phrase)
                modified_prompt_str = ", ".join(prompt_list)

                #calculate modified prompt embedding and sigma score
                modified_prompt_embedding=self.get_prompt_embedding(modified_prompt_str)
                modified_prompt_score= self.get_prompt_score(modified_prompt_embedding)
                modified_prompt_score= (modified_prompt_score - self.positive_mean) / self.positive_std

                # collect self training data
                data=np.concatenate((pooled_prompt_embedding, added_embedding)).tolist(),
                prompt_data={
                    'input': data[0],
                    'position_encoding': position/len(prompt_str),
                    'output': modified_prompt_score,
                    'delta': abs(modified_prompt_score - predicted_score)
                }
                self_training_data.append(prompt_data)

                num_attempts+=1
                # check if score improves
                if(prompt_score < modified_prompt_score):
                    # if it does improve, the new prompt is saved and it jumps to the next iteration
                    prompt_str= modified_prompt_str
                    prompt_embedding= modified_prompt_embedding
                    prompt_score= modified_prompt_score
                    num_success+=1
                    break
            
            self.average_score_by_iteration[i]+=prompt_score
            end= time.time()
            addition_time+= end - start
        
        print(f"time for rejection policy {rejection_policy_time}")
        print(f"time for additions {addition_time}")
        print(f"success rate: {num_success}/{num_attempts}")

        # taking top 10 training datapoints with highest delta
        self_training_data = sorted(self_training_data, key=lambda d: d['delta'], reverse=True)[:10]  
        return prompt_str, prompt_score

def main():
    args = parse_args()
    prompt_generator= PromptTreeSearchGenerator(minio_access_key=args.minio_access_key,
                                  minio_secret_key=args.minio_secret_key,
                                  minio_ip_addr=args.minio_addr,
                                  csv_phrase=args.csv_phrase,
                                  csv_base_prompts=args.csv_base_prompts,
                                  scoring_model=args.scoring_model,
                                  max_iterations=args.max_iterations,
                                  sigma_threshold=args.sigma_threshold,
                                  dataset_name=args.dataset_name,
                                  store_embeddings=args.store_embeddings,
                                  self_training=args.self_training,
                                  send_job=args.send_job,
                                  save_csv=args.save_csv)
    
    
    generated_prompt_str, score=prompt_generator.generate_prompt()
    
    print(f"mutated_prompt:{generated_prompt_str}")
    print(f"mutated_prompt:{score}")
    
    
if __name__ == "__main__":
    main()