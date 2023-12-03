import argparse
from datetime import datetime
import io
import os
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

from worker.prompt_generation.prompt_generator import generate_image_generation_jobs

GENERATION_POLICY="greedy-substitution-search-v1"
DATA_MINIO_DIRECTORY="environmental/data/prompt-generator/substitution"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--csv-phrase', help='CSV containing phrases, must have "phrase str" column', default='input/civitai_phrases_database_v7_no_nsfw.csv')
    parser.add_argument('--n-data', type=int, help='Number of data samples to generate', default=20)
    parser.add_argument('--send-job', action='store_true', default=False)
    parser.add_argument('--update-prompts', action='store_true', default=False)
    parser.add_argument('--dataset-name', default='test-generations')
    parser.add_argument('--scoring-model', help="elm or linear", default="linear")
    parser.add_argument('--rejection-policy', help="by probability or sigma_score", default="sigma_score")
    parser.add_argument('--self-training', action='store_true', default=False)
    parser.add_argument('--store-embeddings', action='store_true', default=False)

    return parser.parse_args()

class PromptSubstitutionGenerator:
    def __init__(
        self,
        minio_access_key,
        minio_secret_key,
        minio_ip_addr,
        csv_phrase,
        scoring_model,
        rejection_policy,
        dataset_name,
        update_prompts,
        store_embeddings,
        self_training,
        send_job
    ):
        start=time.time()

        # parameters
        # csv file containing civitai phrases
        self.csv_phrase=csv_phrase
        # the scoring model used for prompt mutation (elm or linear)
        self.scoring_model= scoring_model
        # rejection policy (by probability of increasing score or sigma score)
        self.rejection_policy= rejection_policy
        # name of dataset
        self.dataset_name=dataset_name
        # wheher to self training or not
        self.self_training=self_training
        # whether to send jobs to server or not
        self.send_job=send_job
        # substitution model (binary of sigma score)
        self.substitution_model= None

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
        
        # load the xgboost model depending on what rejection policy is being used
        if(self.rejection_policy=="sigma_score"):
            self.substitution_model= PromptMutator(minio_client=self.minio_client, ranking_model=self.scoring_model)
        else:
            self.substitution_model= BinaryPromptMutator(minio_client=self.minio_client, ranking_model=self.scoring_model)

        self.substitution_model.load_model()

        # update list of initial prompts if necessary
        if(update_prompts):
            self.update_prompt_list()

        # store phrase embeddings in a file in minio 
        if(store_embeddings):
            self.store_phrase_embeddings()
        
        # get phrase list and embeddings
        self.phrase_list=pd.read_csv(csv_phrase)
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

        return embedding.cpu().numpy()

    # function for rejection sampling with sigma scores
    def rejection_sampling_by_sigma_score(self,
                                    prompt_str, 
                                    prompt_score, 
                                    prompt_embedding, 
                                    phrase_embeddings,
                                    threshold=0.2):

        # get number of tokens
        prompt_list = prompt_str.split(', ')
        token_number= len(prompt_list)
        # list of original and substitute phrases, embeddings ,position and sigma score for each substitution
        sub_phrases=[]
        sub_embeddings=[]
        original_embeddings=[]
        tokens=[]
        sigma_scores=[]

        # looping through each phrase in prompt and predicting score increase in each position
        for token in range(token_number):
            # Get substituted phrase embedding
            substituted_embedding=phrase_embeddings[token]
            # choose a random substitute phrase
            substitute_phrase=self.phrase_list.sample(1).iloc[0]
            substitute_phrase_str=str(substitute_phrase['phrase str'])
            # get substitute phrase embedding
            substitute_embedding= self.phrase_embeddings[substitute_phrase['index']]

            # make inference of sigma score with the substitution xgboost model
            substitution_input= np.concatenate([prompt_embedding, substituted_embedding, substitute_embedding, [token], [prompt_score]])
            sigma_score=self.substitution_model.predict([substitution_input])[0]
            # only take substitutions that increase score by a certain threshold
            if sigma_score > prompt_score + threshold:
                sigma_scores.append(-sigma_score)
                tokens.append(token)
                sub_phrases.append(substitute_phrase_str)
                sub_embeddings.append(substitute_embedding)
                original_embeddings.append(substituted_embedding)
            
        # substitutions are sorted from highest sigma score to lowest
        token_order= np.argsort(sigma_scores)
        tokens=[tokens[token_pos] for token_pos in token_order]
        sub_phrases=[sub_phrases[token_pos] for token_pos in token_order]
        sub_embeddings=[sub_embeddings[token_pos] for token_pos in token_order]
        original_embeddings=[original_embeddings[token_pos] for token_pos in token_order]
        
        return tokens, sub_phrases, original_embeddings, sub_embeddings

    # function for rejection sampling with score increase probability
    def rejection_sampling_by_probability(self, 
                                    prompt_str, 
                                    prompt_score, 
                                    prompt_embedding, 
                                    phrase_embeddings,
                                    ):

        # get number of tokens
        prompt_list = prompt_str.split(', ')
        token_number= len(prompt_list)
        # list of original and substitute phrases, embeddings ,position and increase probability for each substitution
        sub_phrases=[]
        sub_embeddings=[]
        original_embeddings=[]
        tokens=[]
        increase_probs=[]

        # looping through each phrase in prompt and predicting probability of score increase in each position
        for token in range(token_number):
            # Get substituted phrase embedding
            substituted_embedding=phrase_embeddings[token]
            # choose a random substitute phrase
            substitute_phrase=self.phrase_list.sample(1).iloc[0]
            substitute_phrase_str=str(substitute_phrase['phrase str'])
            # get substitute phrase embedding
            substitute_embedding= self.phrase_embeddings[substitute_phrase['index']]

            # make inference of probability of increase with the substitution xgboost model
            substitution_input= np.concatenate([prompt_embedding, substituted_embedding, substitute_embedding, [token], [prompt_score]])
            pred=self.substitution_model.predict_probs([substitution_input])[0]
            # only take substitutions that have more then 66% chance to increase score
            if pred["increase"]>0.66:
                increase_probs.append(-pred["increase"])
                tokens.append(token)
                sub_phrases.append(substitute_phrase_str)
                sub_embeddings.append(substitute_embedding)
                original_embeddings.append(substituted_embedding)
        
        # substitutions are sorted from highest increase probability to lowest
        token_order= np.argsort(increase_probs)
        tokens=[tokens[token_pos] for token_pos in token_order]
        sub_phrases=[sub_phrases[token_pos] for token_pos in token_order]
        sub_embeddings=[sub_embeddings[token_pos] for token_pos in token_order]
        original_embeddings=[original_embeddings[token_pos] for token_pos in token_order]
        
        return tokens, sub_phrases, original_embeddings, sub_embeddings

    # function mutating a prompt
    def mutate_prompt(self,
                    prompt_str,
                    prompt_embedding, 
                    prompt_score,
                    max_iterations=50):

        # calculate mean pooled embedding of each phrase in the prompt 
        phrase_embeddings= [self.get_mean_pooled_embedding(self.get_prompt_embedding(phrase)) for phrase in prompt_str.split(', ')]

        # get rejection policy function
        if(self.rejection_policy=="sigma_score"):
            rejection_func=self.rejection_sampling_by_sigma_score
        else:
            rejection_func=self.rejection_sampling_by_probability

        # self training datapoints
        self_training_data=[]

        # run mutation process for a set number of iterations
        for i in range(max_iterations):
            # get pooled embedding of the prompt
            pooled_prompt_embedding=self.get_mean_pooled_embedding(prompt_embedding)

            # return a list of potential substitution choices, filtered by the rejection policy
            tokens, sub_phrases, original_embeddings, sub_embeddings=rejection_func(
                                                prompt_str,
                                                prompt_score,
                                                pooled_prompt_embedding, 
                                                phrase_embeddings)
            
            # test every choice and take the first choice that increases score
            for token, sub_phrase, original_embedding, sub_embedding in zip(tokens,sub_phrases, original_embeddings, sub_embeddings):
                #Create a modified prompt with the substitution
                prompt_list = prompt_str.split(', ')
                prompt_list[token] = sub_phrase
                modified_prompt_str = ", ".join(prompt_list)

                #calculate modified prompt embedding and sigma score
                modified_prompt_embedding=self.get_prompt_embedding(modified_prompt_str)
                modified_prompt_score= self.get_prompt_score(modified_prompt_embedding)
                modified_prompt_score= (modified_prompt_score - self.positive_mean) / self.positive_std

                # collect self training data, only in the first 5 iterations
                if(i<5):
                    # keeping data for self training
                    data=np.concatenate((pooled_prompt_embedding, original_embedding, sub_embedding)).tolist(),
                    prompt_data={
                        'input': data[0],
                        'position_encoding': token,
                        'linear_score_encoding': prompt_score,
                        'linear_output': modified_prompt_score
                    }
                    self_training_data.append(prompt_data)

                # check if score improves
                if(prompt_score < modified_prompt_score):
                    # if it does improve, the new prompt is saved and it jumps to the next iteration
                    prompt_str= modified_prompt_str
                    prompt_embedding= modified_prompt_embedding
                    phrase_embeddings[token]= sub_embedding
                    prompt_score= modified_prompt_score
                    break
        
        return prompt_str, prompt_embedding, self_training_data

    # function to generate n images
    def generate_images(self, num_images):
        # dataframe for saving csv of generated prompts
        df_data=[]
        # scores before mutation
        original_scores=[]
        # scores after mutation
        mutated_scores=[]
        # collected self training data
        training_data=[]
        index=0

        # get initial prompts
        prompt_list = self.get_initial_prompts(num_prompts=num_images)
    
        start=time.time()
        # mutate prompts one by one
        for i, prompt in prompt_list.iterrows():

            #getting negative and positive prompts
            positive_prompt=prompt['positive_prompt'] 
            negative_prompt=prompt['negative_prompt']
            
            # get prompt embedding
            data = self.minio_client.get_object('datasets', prompt['file_path'])
            # Read the content of the msgpack file
            content = data.read()

            # Deserialize the content using msgpack to get positive and negative embedding
            msgpack_data = msgpack.loads(content)
            positive_embedding= list(msgpack_data['positive_embedding'].values())
            positive_embedding = torch.tensor(np.array(positive_embedding)).float()
            positive_embedding=positive_embedding.to(self.device)
            
            negative_embedding= list(msgpack_data['negative_embedding'].values())
            negative_embedding = torch.tensor(np.array(negative_embedding)).float()
            negative_embedding=negative_embedding.to(self.device)

            # calculating combined score and positive score of prompt before mutation
            seed_score=self.scorer.predict(positive_embedding, negative_embedding).item()
            positive_score=self.positive_scorer.predict_positive_or_negative_only(positive_embedding).item()
            # substract mean and divide by std to get sigma scores
            positive_score= (positive_score - self.positive_mean) / self.positive_std
            seed_sigma_score=(seed_score - self.mean) / self.std
            # append to the list of scores before mutation
            original_scores.append(seed_sigma_score)

            #mutate positive prompt
            mutated_positive_prompt, mutated_positive_embedding, collected_data= self.mutate_prompt(
                            prompt_str=positive_prompt, 
                            prompt_embedding=positive_embedding,
                            prompt_score=positive_score)
            
            # store the collected self training data for this prompt
            training_data.extend(collected_data)

            # calculating new score with the mutated positive prompt
            score=self.scorer.predict(mutated_positive_embedding, negative_embedding).item()
            sigma_score=(score - self.mean) / self.std
            # append to list of scores after mutation
            mutated_scores.append(sigma_score)

            print(f"prompt {index} mutated.")
            print(f"----initial score: {seed_score}.")
            print(f"----final score: {score}.")

            # sending a job to generate an image with the mutated prompt
            if self.send_job:
                try:
                    response = generate_image_generation_jobs(
                        positive_prompt=mutated_positive_prompt,
                        negative_prompt=negative_prompt,
                        prompt_scoring_model=f'image-pair-ranking-{self.scoring_model}',
                        prompt_score=score,
                        prompt_generation_policy=GENERATION_POLICY,
                        top_k='',
                        dataset_name=self.dataset_name
                    )
                    task_uuid = response['uuid']
                    task_time = response['creation_time']
                except:
                    print('Error occured:')
                    print(traceback.format_exc())
                    task_uuid = -1
                    task_time = -1
                
                # storing job data to put in csv file later
                df_data.append({
                    'seed_score': seed_score,
                    'seed_sigma_score': seed_sigma_score,
                    'score': score,
                    'sigma_score': sigma_score,
                    'positive_prompt': mutated_positive_prompt,
                    'negative_prompt': negative_prompt,
                    'seed_prompt': positive_prompt,
                    'generation_policy_string': GENERATION_POLICY,
                    'task_uuid': task_uuid,
                    'time': task_time
                })
            
            index+=1

        end=time.time()

        print(f"time taken for {num_images} prompts is {end - start:.2f} seconds")

        # logging speed of generation
        generation_speed= num_images/(end - start)

        # creating path to save generation data
        current_date=datetime.now().strftime("%Y-%m-%d-%H:%M")
        generation_path=DATA_MINIO_DIRECTORY + f"/generated-images/{current_date}-generated-data"
        # save generated prompts in csv
        if self.send_job:
            self.store_prompts_in_csv_file(df_data, generation_path)
        
        # save a histogram of score distribution before and after mutation for comparison
        self.compare_distributions(generation_path, original_scores, mutated_scores)
        # save a txt file containing generation stats
        self.generation_stats(minio_path=generation_path,
                        generation_speed=generation_speed,
                        num_prompts=num_images,
                        avg_score_before_mutation= np.mean(original_scores),
                        avg_score_after_mutation= np.mean(mutated_scores),
                        )
        
        # save self training data
        if self.self_training:
            self.store_self_training_data(training_data)

    # get paths for embeddings of all prompts in a dataset
    def get_embedding_paths(self, dataset):
            objects=self.minio_client.list_objects('datasets', dataset, recursive=True)
            embedding_files = []
            for obj in objects: 
                if obj.object_name.endswith("_embedding.msgpack"):
                    embedding_files.append(obj.object_name)
                    
            return embedding_files

    # store list of initial prompts in a csv to use for prompt mutation
    def store_prompts_in_csv_file(self, data, minio_path):
        local_path="output/generated_prompts.csv"
        pd.DataFrame(data).to_csv(local_path, index=False)
        # Read the contents of the CSV file
        with open(local_path, 'rb') as file:
            csv_content = file.read()

        #Upload the CSV file to Minio
        buffer = io.BytesIO(csv_content)
        buffer.seek(0)

        minio_path= minio_path + "/generated_prompts.csv"
        cmd.upload_data(self.minio_client, 'datasets', minio_path, buffer)
        # Remove the temporary file
        os.remove(local_path)

    # outputs two histograms for scores before and after mutation for comparison 
    def compare_distributions(self, minio_path, original_scores, mutated_scores):

        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        min_val= min(original_scores)
        max_val= max(mutated_scores)

        # plot histogram of original scores
        axs[0].hist(original_scores, bins=10, range=[min_val,max_val], color='blue', alpha=0.7)
        axs[0].set_xlabel('Scores')
        axs[0].set_ylabel('Frequency')
        axs[0].set_title('Scores Before Mutation')

        # plot histogram of mutated scores
        axs[1].hist(mutated_scores, bins=10, range=[min_val,max_val], color='blue', alpha=0.7)
        axs[1].set_xlabel('Scores')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title('Scores After mutation')

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0.3)

        plt.savefig("output/mutated_scores.png")

        # Save the figure to a file
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # upload the graph report
        minio_path= minio_path + "/mutated_scores.png"
        cmd.upload_data(self.minio_client, 'datasets', minio_path, buf)
        # Remove the temporary file
        os.remove("output/mutated_scores.png")

    # outputs a text file containing logged information about image generation
    def generation_stats(self,
                        minio_path,
                        generation_speed,
                        num_prompts,
                        avg_score_before_mutation,
                        avg_score_after_mutation):

        # General Info
        current_date = datetime.now().strftime("%Y-%m-%d")
        operation = "Phrase substitution"


        # Create the text content
        content = f"================ General Info ==================\n"
        content += f"Date: {current_date}\n"
        content += f"Policy name: {GENERATION_POLICY}\n"
        content += f"Operation: {operation}\n"
        content += f"Scoring model: {self.scoring_model}\n"
        content += f"Dataset: {self.dataset_name}\n\n"

        content += f"================ Generation Stats ==================\n"
        content += f"Number of generated prompts: {num_prompts}\n"
        content += f"Generation speed: {generation_speed:.2f} prompts/sec\n"
        content += f"Loading time: {self.loading_time:.2f} seconds\n"

        content += f"Average sigma score before mutation: {avg_score_before_mutation:.2f}\n"
        content += f"Average sigma score after mutation: {avg_score_after_mutation:.2f}\n"

        # Write content to a text file
        file_path = "generation_stats.txt"  # Update with the desired file path
        with open(file_path, "w") as file:
            file.write(content)

        # Read the contents of the text file
        with open(file_path, 'rb') as file:
            txt_content = file.read()

        #Upload the text file to Minio
        buffer = io.BytesIO(txt_content)
        buffer.seek(0)

        minio_path= minio_path + "/generation_stats.txt"
        cmd.upload_data(self.minio_client, 'datasets', minio_path, buffer)
        # Remove the temporary file
        os.remove(file_path)

    # function to update the initial list of prompts in minIO
    def update_prompt_list(self):
        embedding_paths = self.get_embedding_paths("environmental")
        df_data=[]

        for embedding in embedding_paths:
            print(f"updated {embedding}")
            # get prompt embedding
            data = self.minio_client.get_object('datasets', embedding)
            # Read the content of the msgpack file
            content = data.read()

            # Deserialize the content using msgpack
            msgpack_data = msgpack.loads(content)

            # get positive prompt  
            positive_prompt=msgpack_data['positive_prompt']
        
            # get negative prompt  
            negative_prompt=msgpack_data['negative_prompt']

            # save data 
            df_data.append({
                    'job_uuid':msgpack_data['job_uuid'],
                    'creation_time':msgpack_data['creation_time'],
                    'dataset':msgpack_data['dataset'],
                    'file_path':embedding,
                    'positive_prompt':positive_prompt,
                    'negative_prompt':negative_prompt
                })
        
        # save data locally
        pd.DataFrame(df_data).to_csv('output/initial_prompts.csv', index=False)

        # Read the contents of the CSV file
        with open('output/initial_prompts.csv', 'rb') as file:
            csv_content = file.read()

        #Upload the CSV file to Minio
        buffer = io.BytesIO(csv_content)
        buffer.seek(0)

        minio_path = DATA_MINIO_DIRECTORY + "/input/initial_prompts.csv"
        cmd.upload_data(self.minio_client, 'datasets', minio_path, buffer)

        # Remove the temporary file
        os.remove('output/initial_prompts.csv')

    # get list of initial prompts from minIO
    def get_initial_prompts(self, num_prompts):
        try:
            # Get the CSV file as BytesIO object
            minio_path = DATA_MINIO_DIRECTORY + "/input/initial_prompts.csv"
            data = self.minio_client.get_object('datasets', minio_path)
            csv_data = io.BytesIO(data.read())

            # Read the CSV into a DataFrame
            df = pd.read_csv(csv_data)

            # Filter the DataFrame based on the condition
            filtered_df = df[df['positive_prompt'].str.split(', ').apply(len)>=10]

    
            # get sample prompts
            sampled_df = filtered_df.sample(n=num_prompts)

            return sampled_df

        except ResponseError as err:
            print(f"Error: {err}")
            return None

    # store self training datapoints
    def store_self_training_data(self, training_data):
        # get minio paths for existing self training data
        dataset_path=DATA_MINIO_DIRECTORY + f"/self_training/"
        dataset_files=self.minio_client.list_objects('datasets', prefix=dataset_path, recursive=True)
        dataset_files= [file.object_name for file in dataset_files]
        index= len(dataset_files) + 1

        # store data in msgpack files
        for data in training_data:
            self.store_in_msgpack_file(data, index)
            index+=1

    # store one training datapoint in a msgpack file
    def store_in_msgpack_file(self, prompt_data, index):
        packed_data = msgpack.packb(prompt_data, use_single_float=True)

        # Define the local directory path for embedding
        local_directory = 'output/prompt_mutator/data/'

        # Ensure the local directory exists, create it if necessary
        os.makedirs(local_directory, exist_ok=True)

        # Create a local file with the packed data
        local_file_path = local_directory + f"{str(index).zfill(6)}_substitution.msgpack"
        with open(local_file_path, 'wb') as local_file:
            local_file.write(packed_data)
        
        # Read the contents of the CSV file
        with open(local_file_path, 'rb') as file:
            content = file.read()

        # Upload the local file to MinIO
        buffer = io.BytesIO(content)
        buffer.seek(0)

        minio_path=DATA_MINIO_DIRECTORY + f"/self_training/{str(index).zfill(6)}_substitution.msgpack"
        cmd.upload_data(self.minio_client, 'datasets',minio_path, buffer)

        # Remove the temporary file
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
        minio_path = DATA_MINIO_DIRECTORY + f"/input/phrase_embeddings.npz"
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


def main():
    args = parse_args()
    prompt_mutator= PromptSubstitutionGenerator(minio_access_key=args.minio_access_key,
                                  minio_secret_key=args.minio_secret_key,
                                  minio_ip_addr=args.minio_addr,
                                  csv_phrase=args.csv_phrase,
                                  scoring_model=args.scoring_model,
                                  rejection_policy=args.rejection_policy,
                                  dataset_name=args.dataset_name,
                                  update_prompts=args.update_prompts,
                                  store_embeddings=args.store_embeddings,
                                  self_training=args.self_training,
                                  send_job=args.send_job)
    
    # generate n number of images
    prompt_mutator.generate_images(num_images=args.n_data)
    
if __name__ == "__main__":
    main()