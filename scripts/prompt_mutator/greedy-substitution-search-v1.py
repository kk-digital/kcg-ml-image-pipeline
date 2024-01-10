import argparse
import csv
from datetime import datetime
import io
import os
import random
import re
import sys
import time
import traceback
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import msgpack
from tqdm import tqdm

base_directory = "./"
sys.path.insert(0, base_directory)

from training_worker.prompt_mutator.models.substitution_ranking_xgboost import XgboostSubstitutionModel
from training_worker.prompt_mutator.models.substitution_ranking_linear import LinearSubstitutionModel
from utility.ensemble.ensemble_helpers import Binning, SigmaScoresWithEntropy
from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from training_worker.ab_ranking.model.ab_ranking_linear import ABRankingModel
from stable_diffusion.model.clip_text_embedder.clip_text_embedder import CLIPTextEmbedder
from prompt_job_generator.independent_approx_v1.independent_approx_v1 import IndependentApproxV1
from utility.boltzman.boltzman_phrase_scores_loader import BoltzmanPhraseScoresLoader
from utility.boltzman.boltzman import find_first_element_binary_search, get_cumulative_probability_arr_without_upload
from utility.minio import cmd

from worker.prompt_generation.prompt_generator import generate_image_generation_jobs, generate_prompts_from_csv_with_base_prompt_prefix, load_base_prompts, generate_inpainting_job

GENERATION_POLICY="greedy-substitution-search-v1"
DATA_MINIO_DIRECTORY="data/prompt-generator/substitution"
MAX_LENGTH=77

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
    parser.add_argument('--model-dataset', default='environmental')
    parser.add_argument('--substitution-model', help="substitution model type: xgboost or linear", default='xgboost')
    parser.add_argument('--scoring-model', help="elm or linear", default="linear")
    parser.add_argument('--sigma-threshold', type=float, help="threshold of rejection policy for increase of sigma score", default=-0.1)
    parser.add_argument('--variance-weight', type=float, help="weight of variance when optimizing score", default=0)
    parser.add_argument('--boltzman-temperature', type=int, default=11)
    parser.add_argument('--boltzman-k', type=float, default=1.0)
    parser.add_argument('--max-iterations', type=int, help="number of mutation iterations", default=80)
    parser.add_argument('--self-training', action='store_true', default=False)
    parser.add_argument('--store-embeddings', action='store_true', default=False)
    parser.add_argument('--store-token-lengths', action='store_true', default=False)
    parser.add_argument('--save-csv', action='store_true', default=False)
    parser.add_argument('--initial-generation-policy', help="the generation policy used for generating the initial seed prompts", default="independant_approximation")
    parser.add_argument('--top-k', type=float, help="top percentage of prompts taken from generation to be mutated", default=0.1)
    parser.add_argument('--num_choices', type=int, help="Number of substituion choices tested every iteration", default=128)
    parser.add_argument('--clip-batch-size', type=int, help="Batch size for clip embeddings", default=1000)
    parser.add_argument('--substitution-batch-size', type=int, help="Batch size for the substitution model", default=100000)

    return parser.parse_args()

class PromptData:
    def __init__(self, 
                 positive_prompt,
                 negative_prompt,
                 positive_embedding,
                 negative_embedding,
                 positive_score,
                 variance_score,
                 positive_phrase_embeddings=None,
                 positive_phrase_token_lengths=None):
        
        self.positive_prompt=positive_prompt
        self.negative_prompt=negative_prompt
        self.positive_embedding= positive_embedding
        self.negative_embedding= negative_embedding
        self.positive_score= positive_score
        self.variance_score= variance_score
        self.positive_phrase_embeddings= positive_phrase_embeddings
        self.positive_phrase_token_lengths= positive_phrase_token_lengths

    def get_phrases(self):
        # split prompt phrases by comma
        phrases = self.positive_prompt.split(', ')

        # merge quoted phrases that have commas within them that were split
        merged_phrases = []
        merge_next = False

        for phrase in phrases:
            if phrase.startswith('"') and not phrase.endswith('"'):
                # Start of a quoted phrase
                merge_next = True
                merged_phrases.append(phrase)
            elif merge_next and not phrase.endswith('"'):
                # Middle of a quoted phrase
                merged_phrases[-1] += ', ' + phrase
            elif merge_next and phrase.endswith('"'):
                # End of a quoted phrase
                merged_phrases[-1] += ', ' + phrase
                print(merged_phrases[-1])
                merge_next = False
            else:
                # Regular, non-quoted phrase
                merged_phrases.append(phrase)

        # Merge phrases that were incorrectly split
        return merged_phrases

class PromptSubstitutionGenerator:
    def __init__(
        self,
        minio_access_key,
        minio_secret_key,
        minio_ip_addr,
        csv_phrase,
        csv_base_prompts,
        model_dataset,
        substitution_model,
        scoring_model,
        max_iterations,
        sigma_threshold,
        variance_weight,
        boltzman_temperature,
        boltzman_k,
        dataset_name,
        store_embeddings,
        store_token_lengths,
        self_training,
        send_job,
        save_csv,
        initial_generation_policy,
        top_k,
        num_choices_per_iteration,
        clip_batch_size,
        substitution_batch_size
    ):
        start=time.time()

        # parameters
        # csv file containing civitai phrases
        self.csv_phrase=csv_phrase
        # the scoring model used for prompt mutation (elm or linear)
        self.scoring_model= scoring_model
        # number of iterations for prompt mutation
        self.max_iterations= max_iterations
        # average score by iteration to track score improvement
        self.average_score_by_iteration=np.zeros(self.max_iterations)
        # rejection threshold for increase in sigma score
        self.sigma_threshold= sigma_threshold
        # variance weight for optimisation score
        self.variance_weight= variance_weight
        # boltzman temperature and k value
        self.boltzman_temperature = boltzman_temperature
        self.boltzman_k = boltzman_k
        # name of dataset to generate for
        self.model_dataset= model_dataset
        # name of dataset
        self.dataset_name=dataset_name
        # wheher to self training or not
        self.self_training=self_training
        # whether to send jobs to server or not
        self.send_job=send_job
        # whether to save csv of prompts or not
        self.save_csv=save_csv
        # substitution model (linear or xgboost)
        self.model_type= substitution_model
        # initial generation policy
        self.initial_generation_policy= initial_generation_policy
        # top k value for generating initial prompts
        self.top_k=top_k
        # number of substitution choices tested every iteration
        self.num_choices_per_iteration= num_choices_per_iteration
        # batch size for clip embeddings
        self.clip_batch_size= clip_batch_size
        # batch size for xgboost inference
        self.substitution_batch_size=substitution_batch_size
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
        self.mean, self.std= float(self.scorer.mean), float(self.scorer.standard_deviation)
        self.positive_mean, self.positive_std= float(self.positive_scorer.mean), float(self.positive_scorer.standard_deviation)
        
        # get ensemble models
        self.ensemble_models=self.get_ensemble_models()

        # load the substitution model 
        if(self.model_type=="xgboost"):
            self.substitution_model= XgboostSubstitutionModel(minio_client=self.minio_client, 
                                                          ranking_model=self.scoring_model, 
                                                          dataset=self.model_dataset)
        elif(self.model_type=="linear"):
            self.substitution_model= LinearSubstitutionModel(minio_client=self.minio_client, 
                                                          ranking_model=self.scoring_model, 
                                                          dataset=self.model_dataset)
            
        self.substitution_model.load_model()
        
        if(self.initial_generation_policy=="fixed_probabilities"):
            # get list of phrases and their token lengths
            phrase_df=pd.read_csv(self.csv_phrase).sort_values(by="index")
            self.phrase_list=phrase_df['phrase str'].tolist()

        else:
            # get list of boltzman phrase score
            self.positive_phrase_scores_csv,self.negative_phrase_scores_csv=self.get_boltzman_scores_csv()
            
            # loading positive phrase scores to use for rejection sampling
            phrase_loader=BoltzmanPhraseScoresLoader(dataset_name="environmental",
                                                    phrase_scores_csv=self.positive_phrase_scores_csv,
                                                    minio_client=self.minio_client)
            phrase_loader.load_dataset()
            self.phrase_score_data= phrase_loader.index_phrase_score_data
            self.phrase_list=[self.phrase_score_data[i].phrase for i in range(len(self.phrase_score_data))]

            # get cumulative probabilities
            self.positive_phrase_origin_indexes, self.positive_cumulative_probability_arr = get_cumulative_probability_arr_without_upload(
            index_phrase_score_data=self.phrase_score_data,
            boltzman_temperature=boltzman_temperature,
            boltzman_k=boltzman_k)

            # get min and max
            self.positive_cumulative_probability_arr_min = self.positive_cumulative_probability_arr.min()
            self.positive_cumulative_probability_arr_max = self.positive_cumulative_probability_arr.max()

        # store phrase embeddings in a file in minio 
        if(store_embeddings):
            self.store_phrase_embeddings()
            
        # get phrase embeddings    
        self.phrase_embeddings= self.load_phrase_embeddings()

        # store phrase token lengths
        if(store_token_lengths):
            self.store_phrase_token_lengths()
        
        # store list of token lengths for each phrase
        self.phrase_token_lengths=self.load_phrase_token_lengths()

        # create a dictionarry to get phrase index from phrase str
        self.phrase_index_dictionarry={phrase:i for i, phrase in enumerate(self.phrase_list)}

        # get base prompt list
        base_prompts = load_base_prompts(self.csv_base_prompts)
        # create a dictionarry for base prompts
        self.base_prompt_embeddings={phrase: self.get_mean_pooled_embedding(self.get_prompt_embedding(phrase)) for phrase in base_prompts}
        self.base_prompt_token_lengths={phrase: self.get_token_length(phrase) for phrase in base_prompts}

        end=time.time()
        # log time taken for each step
        self.loading_time= end-start
        self.generation_time=0
        self.mutation_time=0
        self.clip_speed=0
        self.inference_speed=0
    
    # load ensemble elm model for entropy calculation
    def get_ensemble_models(self):
        input_path = f"{self.model_dataset}/models/ranking/"

        model_class = ABRankingModel

        # Get all model files
        model_files = cmd.get_list_of_objects_with_prefix(self.minio_client, 'datasets', input_path)

        # Filter relevant model files
        relevant_models = [
            model_file for model_file in model_files
            if model_file.endswith(f"score-linear-embedding.safetensors")
        ]

        # Sort the model files by timestamp (assuming the file names include a timestamp)
        relevant_models=np.flip(relevant_models)

        # Load the latest num_models models
        loaded_models = []
        for i in range(min(16, len(relevant_models))):
            most_recent_model = relevant_models[i]

            # Get the model file data
            model_file_data = cmd.get_file_from_minio(self.minio_client, 'datasets', most_recent_model)

            # Create a BytesIO object and write the downloaded content into it
            byte_buffer = io.BytesIO()
            for data in model_file_data.stream(amt=8192):
                byte_buffer.write(data)
            # Reset the buffer's position to the beginning
            byte_buffer.seek(0)

            # Load the model
            embedding_model = model_class(768*2)
            embedding_model.load_safetensors(byte_buffer)
            embedding_model.model=embedding_model.model.to(self.device)

            loaded_models.append(embedding_model)

        return loaded_models
    
    # get sigma scores for ensemble models
    def get_ensemble_sigma_scores(self, positive_embedding, negative_embedding):
        sigma_scores=[]
        for model in self.ensemble_models:
            mean=float(model.mean)
            std=float(model.standard_deviation)
            with torch.no_grad():
                score=model.predict_pooled_embeddings(positive_embedding,negative_embedding).item()
            
            score=(score - mean)/std
            sigma_scores.append(score)
        
        return np.array(sigma_scores)
        
    # get prompt mean, entropy and variance for ensemble scores
    def get_prompt_entropy(self, positive_embedding, negative_embedding, start=-2, bins=8, step=1):
        # convert embeddings to tensors
        positive_embedding= torch.from_numpy(positive_embedding).to(self.device)
        negative_embedding= torch.from_numpy(negative_embedding).to(self.device)

        # get ensemble sigma scores
        sigma_scores=self.get_ensemble_sigma_scores(positive_embedding, negative_embedding)

        # get entropy classes
        binning= Binning(start=start,count=bins,step=step)
        entropy_data=SigmaScoresWithEntropy(sigma_scores, binning)

        # get entropy, variance and average
        entropy= entropy_data.entropy
        variance= entropy_data.variance
        mean= entropy_data.mean

        return entropy, variance, mean

    # get combined quality score and variance
    def get_variance_score(self, positive_embedding, negative_embedding, score):

        if(self.variance_weight==0):
            return score

        # convert embeddings to tensors
        positive_embedding= torch.from_numpy(positive_embedding).to(self.device)
        negative_embedding= torch.from_numpy(negative_embedding).to(self.device)
        
        # get ensemble sigma scores
        sigma_scores=self.get_ensemble_sigma_scores(positive_embedding, negative_embedding)

        # get variance
        variance= np.var(sigma_scores, axis=-1)

        # get comibned variance score
        variance_score= score + (self.variance_weight * variance)

        return variance_score

    # load elm or linear scoring models
    def load_model(self, embedding_type, scoring_model="linear", input_size=768):
        input_path=f"{self.model_dataset}/models/ranking/"

        if(scoring_model=="elm"):
            embedding_model = ABRankingELMModel(input_size)
            file_name=f"score-elm-v1-embedding"
        else:
            embedding_model= ABRankingModel(input_size)
            file_name=f"score-linear-embedding"
        
        if(embedding_type=="positive" or embedding_type=="negative"):
            file_name+=f"-{embedding_type}.safetensors"
        else:
            file_name+=".safetensors"

        model_files=cmd.get_list_of_objects_with_prefix(self.minio_client, 'datasets', input_path)
        most_recent_model = None

        for model_file in model_files:
            if model_file.endswith(file_name):
                most_recent_model = model_file

        if most_recent_model:
            model_file_data =cmd.get_file_from_minio(self.minio_client, 'datasets', most_recent_model)
        else:
            print("No .safetensors files found in the list.")
            return
        
        print(most_recent_model)

        # Create a BytesIO object and write the downloaded content into it
        byte_buffer = io.BytesIO()
        for data in model_file_data.stream(amt=8192):
            byte_buffer.write(data)
        # Reset the buffer's position to the beginning
        byte_buffer.seek(0)

        embedding_model.load_safetensors(byte_buffer)
        embedding_model.model=embedding_model.model.to(self.device)

        return embedding_model

    # get path for phrase score csv files
    def get_boltzman_scores_csv(self):
        score_csvs=cmd.get_list_of_objects_with_prefix(self.minio_client, 'datasets', 'environmental/output/phrases-score-csv')
        positive_score_csv = None
        negative_score_csv = None

        for csv_file in score_csvs:
            if csv_file.endswith('positive-phrases-score.csv'):
                positive_score_csv = csv_file.split("/")[-1]
            if csv_file.endswith('negative-phrases-score.csv'):
                negative_score_csv = csv_file.split("/")[-1]
        
        return positive_score_csv, negative_score_csv

    # get the clip text embedding of a prompt or a phrase
    def get_prompt_embedding(self, prompts):
        # Ensure phrases is a list
        if not isinstance(prompts, list):
            prompts = [prompts]

        with torch.no_grad():
            embeddings= self.embedder(prompts)
        
        embeddings= embeddings.unsqueeze(1)
        embeddings=embeddings.to(self.device)
        
        if len(embeddings) == 1:
            return embeddings[0].unsqueeze(0)
        
        return embeddings

    # get linear or elm positive score of an embedding
    def get_positive_score(self, embedding):
        embedding= torch.from_numpy(embedding).to(self.device).unsqueeze(0)

        with torch.no_grad():
            prompt_score=self.positive_scorer.predict_positive_or_negative_only_pooled(embedding)
        
        return prompt_score.item()
    
    # get linear or elm score of an embedding
    def get_combined_score(self, positive_embedding, negative_embedding):
        # convert embeddings to tensors
        positive_embedding= torch.from_numpy(positive_embedding).to(self.device).unsqueeze(0)
        negative_embedding= torch.from_numpy(negative_embedding).to(self.device).unsqueeze(0)

        with torch.no_grad():
            prompt_score=self.scorer.predict_pooled_embeddings(positive_embedding, negative_embedding)
        
        return prompt_score.item()

    # get the mean pool of an embedding
    def get_mean_pooled_embedding(self, embedding):
        embedding=torch.mean(embedding, dim=2)
        embedding = embedding.reshape(len(embedding), -1).squeeze(0)

        return embedding.detach().cpu().numpy()
    
    # get token length of a phrase
    def get_token_length(self, phrase):
        # Tokenize the phrase
        batch_encoding = self.embedder.tokenizer(phrase, truncation=False, max_length=MAX_LENGTH, return_length=True,
                                        return_overflowing_tokens=False, return_tensors="pt")
        
        input_ids = batch_encoding['input_ids']
        num_tokens = input_ids.numel()

        return num_tokens

    # selection policy
    def selection_policy(self, max_token_length):
        if(self.initial_generation_policy=="fixed_probabilities"):
            random_index, phrase= self.choose_random_phrase(max_token_length)
        else:
            random_index, phrase= self.choose_phrase_by_temperature(max_token_length)
        
        return random_index, phrase
    
    # function to get a random phrase from civitai with a max token size for substitutions
    def choose_random_phrase(self, max_token_length):
        phrase_token_length=max_token_length + 1

        while(phrase_token_length > max_token_length):
            random_index=random.randrange(0, len(self.phrase_list))
            phrase= self.phrase_list[random_index]
            phrase_token_length=self.phrase_token_lengths[random_index]

        return random_index, phrase
    
    # function to choose a phrase based on binary search
    def choose_phrase_by_temperature(self, max_token_length):
        phrase_token_length=max_token_length + 1

        random_float = random.uniform(self.positive_cumulative_probability_arr_min,
                                      self.positive_cumulative_probability_arr_max)
            
        random_index = find_first_element_binary_search(self.positive_cumulative_probability_arr, 
                                                        random_float)
        prompt_index = self.positive_phrase_origin_indexes[random_index]
        phrase= self.phrase_list[prompt_index]
        phrase_token_length=self.phrase_token_lengths[prompt_index]

        counter= 1 if prompt_index < (len(self.phrase_list)-prompt_index) else -1

        while(phrase_token_length > max_token_length):
            prompt_index+=counter
            phrase= self.phrase_list[prompt_index]
            phrase_token_length=self.phrase_token_lengths[prompt_index]

        return prompt_index, phrase

    # rejection sampling function
    def rejection_sampling(self, prompts):
        # arrays to save substitution data
        substitution_inputs=[]
        sampled_phrases=[]
        sampled_embeddings=[]
        substitution_positions=[]
        
        # number of choices per iteration
        num_choices=self.num_choices_per_iteration

        for prompt in prompts:
            # get number of phrases
            prompt_list = prompt.get_phrases()
            num_phrases= len(prompt_list)

            # create a substitution for each position in the prompt
            for i in range(num_choices):
                phrase_position= random.randint(0, num_phrases - 1)
                # get the substituted phrase token length
                substituted_phrase_length=prompt.positive_phrase_token_lengths[phrase_position]
                # get the substituted phrase embedding
                substituted_embedding = prompt.positive_phrase_embeddings[phrase_position]
                # get a random phrase from civitai to substitute with
                phrase_index, random_phrase= self.choose_random_phrase(max_token_length=substituted_phrase_length)   
                # get phrase string
                substitute_phrase = random_phrase
                # get phrase embedding by its index
                substitute_embedding = self.phrase_embeddings[phrase_index]
                # concatenate input in one array to use for inference
                substitution_input = np.concatenate([prompt.positive_embedding, substituted_embedding, 
                                                     substitute_embedding, [phrase_position], [prompt.positive_score]])
                # save data in an array to use for inference and rejection sampling
                substitution_inputs.append(substitution_input)
                sampled_phrases.append(substitute_phrase)
                sampled_embeddings.append(substitute_embedding)
                substitution_positions.append(phrase_position)
        
        # Predict sigma score for every substitution
        start=time.time()
        predictions = self.substitution_model.predict(data=substitution_inputs, batch_size=self.substitution_batch_size)
        end=time.time()
        self.inference_speed+= (num_choices * len(prompts))/ (end-start)

        prompt_index=0
        choices_count=1
        current_prompt_substitution_choices=[]
        prompts_substitution_choices=[]
        # Filter with rejection sampling
        for index, sigma_score in enumerate(predictions):
            # only take substitutions that increase score by more then a set threshold
            if sigma_score > prompts[prompt_index].positive_score + self.sigma_threshold:
                phrase_position=substitution_positions[index]
                substitution_data={
                    'position':phrase_position,
                    'substitute_phrase':sampled_phrases[index],
                    'substitute_embedding':sampled_embeddings[index],
                    'substituted_embedding':prompts[prompt_index].positive_phrase_embeddings[phrase_position],
                    'score':sigma_score
                }
                current_prompt_substitution_choices.append(substitution_data)
            
            if(choices_count == num_choices):
                prompt_index+=1
                choices_count=0
                # substitutions are sorted from highest sigma score to lowest
                current_prompt_substitution_choices= sorted(current_prompt_substitution_choices, key=lambda s: s['score'], reverse=True) 
                prompts_substitution_choices.append(current_prompt_substitution_choices)
                current_prompt_substitution_choices=[]
            
            choices_count+=1
        
        return prompts_substitution_choices

    # function to mutate prompts
    def mutate_prompts(self, prompts):
        start= time.time()
        # self training datapoints
        self_training_data=[]
        num_prompts=len(prompts)

        # run mutation process for a set number of iterations
        for i in range(self.max_iterations):
            print(f"Iteration {i} -----------------------------")
            # return a list of potential substitution choices, filtered by the rejection policy
            print("ranking the best substitution choices")
            prompt_substitutions=self.rejection_sampling(prompts)

            print("Mutating prompts")
            index=0
            for substitution_choices in tqdm(prompt_substitutions):
                for substitution in substitution_choices:
                    # get substitution data
                    position=substitution['position']
                    substitute_phrase=substitution['substitute_phrase']
                    substitute_embedding=substitution['substitute_embedding']
                    substituted_embedding=substitution['substituted_embedding']
                    predicted_score=substitution['score']

                    #Create a modified prompt with the substitution
                    prompt_list = prompts[index].get_phrases()
                    prompt_list[position] = substitute_phrase
                    modified_prompt_str = ", ".join(prompt_list)

                    #calculate modified prompt embedding and sigma score
                    modified_prompt_embedding=self.get_prompt_embedding(modified_prompt_str)
                    modified_prompt_embedding=self.get_mean_pooled_embedding(modified_prompt_embedding)
                    modified_prompt_score= self.get_positive_score(modified_prompt_embedding)
                    modified_prompt_score= (modified_prompt_score - self.positive_mean) / self.positive_std

                    # get variance score
                    variance_score= self.get_variance_score(modified_prompt_embedding,
                                                            prompts[index].negative_embedding,
                                                            modified_prompt_score)

                    if(self.self_training):
                        # collect self training data
                        data=np.concatenate((prompts[index].positive_embedding, 
                                            substituted_embedding, 
                                            substitute_embedding)).tolist(),
                    
                        prompt_data={
                        'input': data[0],
                        'position_encoding': position,
                        'score_encoding': prompts[index].positive_prompt,
                        'output': modified_prompt_score,
                        'delta': abs(modified_prompt_score - predicted_score)
                        }
                        self_training_data.append(prompt_data)

                    # check if score improves
                    if(prompts[index].variance_score < variance_score):
                        # if it does improve, the new prompt is saved and it jumps to the next iteration
                        prompts[index].positive_prompt= modified_prompt_str
                        prompts[index].positive_embedding= modified_prompt_embedding
                        prompts[index].positive_phrase_embeddings[position]= substitute_embedding
                        prompts[index].positive_score= modified_prompt_score
                        prompts[index].variance_score= variance_score
                        break
                
                self.average_score_by_iteration[i]+=prompts[index].positive_score
                index+=1
            
            # save average score for current iteration
            self.average_score_by_iteration[i]=self.average_score_by_iteration[i] / num_prompts
            print(f"Average score: {self.average_score_by_iteration[i]}")

        # taking top k training datapoints with highest delta
        self_training_data = sorted(self_training_data, key=lambda d: d['delta'], reverse=True)[:10 * len(prompts)]  
        
        end=time.time()
        self.mutation_time= end-start
        self.inference_speed= self.inference_speed / self.max_iterations

        return prompts, self_training_data

    # function to generate n images
    def generate_images(self, num_images):
        # dataframe for saving csv of generated prompts
        df_data=[]
        index=0

        start=time.time()
        # get initial prompts
        if(self.initial_generation_policy=="independant_approximation"):
            prompt_list = self.generate_initial_prompts_with_independant_approximation(num_images)
        else:
            prompt_list = self.generate_initial_prompts_with_fixed_probs(num_images)

        #mutate positive prompts
        prompts, self_training_data= self.mutate_prompts(prompt_list)
        end=time.time()
      
        # mutate prompts one by one
        for prompt in prompts:
            # calculate combined prompt score
            prompt_score=self.get_combined_score(prompt.positive_embedding, prompt.negative_embedding)
            sigma_score= (prompt_score - self.mean) / self.std

            entropy, variance, mean= self.get_prompt_entropy(prompt.positive_embedding, prompt.negative_embedding)
           
            # sending a job to generate an image with the mutated prompt
            if self.send_job:
                try:
                    if self.model_dataset in ["environmental", "propaganda-poster", "waifu"]:
                        response = generate_image_generation_jobs(
                            positive_prompt=prompt.positive_prompt,
                            negative_prompt=prompt.negative_prompt,
                            prompt_scoring_model=f'image-pair-ranking-{self.scoring_model}',
                            prompt_score=prompt_score,
                            prompt_generation_policy=GENERATION_POLICY,
                            top_k='',
                            dataset_name=self.dataset_name
                        )

                    elif self.model_dataset in ["icons"]:
                        response = generate_inpainting_job(
                            positive_prompt=prompt.positive_prompt,
                            negative_prompt=prompt.negative_prompt,
                            prompt_scoring_model=f'image-pair-ranking-{self.scoring_model}',
                            prompt_score=prompt_score,
                            prompt_generation_policy=GENERATION_POLICY,
                            top_k='',
                            dataset_name=self.dataset_name,
                            init_img_path="./test/test_inpainting/white_512x512.jpg",
                            mask_path="./test/test_inpainting/icon_mask.png"
                        )
                    
                    elif self.model_dataset in ["mech"]:
                        response = generate_inpainting_job(
                            positive_prompt=prompt.positive_prompt,
                            negative_prompt=prompt.negative_prompt,
                            prompt_scoring_model=f'image-pair-ranking-{self.scoring_model}',
                            prompt_score=prompt_score,
                            prompt_generation_policy=GENERATION_POLICY,
                            top_k='',
                            dataset_name=self.dataset_name,
                            init_img_path="./test/test_inpainting/white_512x512.jpg",
                            mask_path="./input/mask/mech/mech_mask_2x2.png"
                        )
                        
                    elif self.model_dataset in ["character"]:
                        response = generate_inpainting_job(
                            positive_prompt=prompt.positive_prompt,
                            negative_prompt=prompt.negative_prompt,
                            prompt_scoring_model=f'image-pair-ranking-{self.scoring_model}',
                            prompt_score=prompt_score,
                            prompt_generation_policy=GENERATION_POLICY,
                            top_k='',
                            dataset_name=self.dataset_name,
                            init_img_path="./test/test_inpainting/white_512x512.jpg",
                            mask_path = "./test/test_inpainting/character_mask.png"
                        ) 

                    task_uuid = response['uuid']
                    task_time = response['creation_time']
                except:
                    print('Error occured:')
                    print(traceback.format_exc())
                    task_uuid = -1
                    task_time = -1

            if self.save_csv:   
                # storing job data to put in csv file later
                df_data.append({
                    'task_uuid': task_uuid,
                    'score': prompt_score,
                    'sigma_score': sigma_score,
                    'entropy': entropy,
                    'variance': variance,
                    'mean': mean,
                    'positive_prompt': prompt.positive_prompt,
                    'negative_prompt': prompt.negative_prompt,
                    'generation_policy_string': GENERATION_POLICY,
                    'time': task_time
                })

            index+=1

        print(f"time taken for {num_images} prompts is {end - start:.2f} seconds")

        # logging speed of generation
        generation_speed= num_images/(end - start)

        # save generated prompts in csv
        if self.save_csv:
            self.store_prompts_in_csv_file(df_data)

        # creating path to save generation data
        current_date=datetime.now().strftime("%Y-%m-%d-%H:%M")
        generation_path=DATA_MINIO_DIRECTORY + f"/generated-images/{current_date}-generated-data"
        
        # save a graph for score improvement by number of iterations
        self.score_improvement_graph(minio_path=generation_path)

        # save a txt file containing generation stats
        self.generation_stats(minio_path=generation_path,
                        generation_speed=generation_speed,
                        num_prompts=num_images,
                        avg_score_before_mutation= self.average_score_by_iteration[0],
                        avg_score_after_mutation= self.average_score_by_iteration[-1],
                        )
        
        # save self training data
        if self.self_training:
            self.store_self_training_data(self_training_data)
        
    # function to return prompt progression in mutation
    def prompt_mutation_progression(self, prompts):
        num_prompts=len(prompts)
        prompt_progression= [[] for _ in range(num_prompts)]

        # run mutation process for a set number of iterations
        for i in range(self.max_iterations):
            print(f"Iteration {i} -----------------------------")
            # return a list of potential substitution choices, filtered by the rejection policy
            print("ranking the best substitution choices")
            prompt_substitutions=self.rejection_sampling(prompts)

            print("Mutating prompts")
            index=0
            for substitution_choices in tqdm(prompt_substitutions):
                for substitution in substitution_choices:
                    # get substitution data
                    position=substitution['position']
                    substitute_phrase=substitution['substitute_phrase']
                    substitute_embedding=substitution['substitute_embedding']

                    #Create a modified prompt with the substitution
                    prompt_list = prompts[index].get_phrases()
                    prompt_list[position] = substitute_phrase
                    modified_prompt_str = ", ".join(prompt_list)

                    #calculate modified prompt embedding and sigma score
                    modified_prompt_embedding=self.get_prompt_embedding(modified_prompt_str)
                    modified_prompt_embedding=self.get_mean_pooled_embedding(modified_prompt_embedding)
                    modified_prompt_score= self.get_positive_score(modified_prompt_embedding)
                    modified_prompt_score= (modified_prompt_score - self.positive_mean) / self.positive_std

                    # get variance score
                    entropy, variance, mean, variance_score= self.get_variance_score(modified_prompt_embedding,
                                                                                     prompts[index].negative_embedding,
                                                                                     modified_prompt_score
                                                                                    )

                    # check if score improves
                    if(prompts[index].variance_score < variance_score):
                        # if it does improve, the new prompt is saved and it jumps to the next iteration
                        prompts[index].positive_prompt= modified_prompt_str
                        prompts[index].positive_embedding= modified_prompt_embedding
                        prompts[index].positive_phrase_embeddings[position]= substitute_embedding
                        prompts[index].positive_score= modified_prompt_score
                        prompts[index].variance_score= variance_score
                        prompts[index].entropy= entropy
                        prompts[index].variance= variance
                        prompts[index].mean= mean
                        break
                
                self.average_score_by_iteration[i]+=prompts[index].positive_score

                prompt_progression[index].append({
                    "positive_prompt": prompts[index].positive_prompt,
                    "negative_prompt": prompts[index].negative_prompt,
                    "positive_embedding":prompts[index].positive_embedding, 
                    "negative_embedding":prompts[index].negative_embedding,
                    "entropy": prompts[index].entropy,
                    "variance": prompts[index].variance,
                    "mean": prompts[index].mean
                })
                index+=1
            
            # save average score for current iteration
            self.average_score_by_iteration[i]=self.average_score_by_iteration[i] / num_prompts
            print(f"Average score: {self.average_score_by_iteration[i]}")

        return prompt_progression
    
    # function to generate images each iteration
    def generate_image_progression(self, num_images):
        # dataframe for saving csv of generated prompts
        df_data=[]
        index=0

        start=time.time()
        # get initial prompts
        prompt_list = self.generate_initial_prompts(num_images)

        #mutate positive prompts
        prompts_progression= self.prompt_mutation_progression(prompt_list)
        end=time.time()
      
        # mutate prompts one by one
        for prompt_index, prompt_progres in enumerate(prompts_progression):

            for iteration, prompt in enumerate(prompt_progres):
                # calculate combined prompt score
                prompt_score=self.get_combined_score(prompt['positive_embedding'], prompt['negative_embedding'])
                sigma_score= (prompt_score - self.mean) / self.std
            
                # sending a job to generate an image with the mutated prompt
                if self.send_job:
                    try:
                        if self.model_dataset in ["environmental", "propaganda-poster", "waifu"]:
                            response = generate_image_generation_jobs(
                                positive_prompt=prompt['positive_prompt'],
                                negative_prompt=prompt['negative_prompt'],
                                prompt_scoring_model=f'image-pair-ranking-{self.scoring_model}',
                                prompt_score=prompt_score,
                                prompt_generation_policy=GENERATION_POLICY,
                                top_k='',
                                dataset_name=self.dataset_name
                            )

                        elif self.model_dataset in ["icons"]:
                            response = generate_inpainting_job(
                                positive_prompt=prompt['positive_prompt'],
                                negative_prompt=prompt['negative_prompt'],
                                prompt_scoring_model=f'image-pair-ranking-{self.scoring_model}',
                                prompt_score=prompt_score,
                                prompt_generation_policy=GENERATION_POLICY,
                                top_k='',
                                dataset_name=self.dataset_name,
                                init_img_path="./test/test_inpainting/white_512x512.jpg",
                                mask_path="./test/test_inpainting/icon_mask.png"
                            )
                        
                        elif self.model_dataset in ["mech"]:
                            response = generate_inpainting_job(
                                positive_prompt=prompt['positive_prompt'],
                                negative_prompt=prompt['negative_prompt'],
                                prompt_scoring_model=f'image-pair-ranking-{self.scoring_model}',
                                prompt_score=prompt_score,
                                prompt_generation_policy=GENERATION_POLICY,
                                top_k='',
                                dataset_name=self.dataset_name,
                                init_img_path="./test/test_inpainting/white_512x512.jpg",
                                mask_path="./input/mask/mech/mech_mask_2x2.png"
                            )
                            
                        elif self.model_dataset in ["character"]:
                            response = generate_inpainting_job(
                                positive_prompt=prompt['positive_prompt'],
                                negative_prompt=prompt['negative_prompt'],
                                prompt_scoring_model=f'image-pair-ranking-{self.scoring_model}',
                                prompt_score=prompt_score,
                                prompt_generation_policy=GENERATION_POLICY,
                                top_k='',
                                dataset_name=self.dataset_name,
                                init_img_path="./test/test_inpainting/white_512x512.jpg",
                                mask_path = "./test/test_inpainting/character_mask.png"
                            ) 

                        task_uuid = response['uuid']
                        task_time = response['creation_time']
                    except:
                        print('Error occured:')
                        print(traceback.format_exc())
                        task_uuid = -1
                        task_time = -1

                if self.save_csv:   
                    # storing job data to put in csv file later
                    df_data.append({
                        'task_uuid': task_uuid,
                        'prompt_index': prompt_index + 1,
                        'iteration': iteration + 1,
                        'score': prompt_score,
                        'sigma_score': sigma_score,
                        'entropy': prompt['entropy'],
                        'variance': prompt['variance'],
                        'mean': prompt['mean'],
                        'positive_prompt': prompt['positive_prompt'],
                        'negative_prompt': prompt['negative_prompt'],
                        'generation_policy_string': GENERATION_POLICY,
                        'time': task_time
                    })

                index+=1

        print(f"time taken for {num_images} prompts is {end - start:.2f} seconds")

        # save generated prompts in csv
        if self.save_csv:
            self.store_prompts_in_csv_file(df_data)

    # generate initial prompts with top k
    def generate_initial_prompts_with_independant_approximation(self, num_prompts):
        total_start=time.time()
        print("---------generating initial prompts")
        # number of prompts to generate
        prompt_count= int(num_prompts / self.top_k) if num_prompts>10 else 100

        # generate initial prompts before mutation
        prompt_generator = IndependentApproxV1(self.model_dataset)
        prompt_generator.load_csv(self.minio_client, self.positive_phrase_scores_csv, self.negative_phrase_scores_csv)
        
        prompts =  prompt_generator.generate_prompts(prompt_count, self.boltzman_temperature, self.boltzman_k)

            
        prompt_data=[]
        clip_time=0
        # calculate scores and rank
        print("---------scoring prompts")
        for batch_start in tqdm(range(0, len(prompts), self.clip_batch_size)):
            batch_end = batch_start + self.clip_batch_size
            prompt_batch = prompts[batch_start:batch_end]

            # Prepare data for batch processing
            positive_prompts = [p['positive_prompt'] for p in prompt_batch]
            negative_prompts = [p['negative_prompt'] for p in prompt_batch]

            # Compute token lengths for the batch
            positive_token_lengths = [self.embedder.compute_token_length(prompt) for prompt in positive_prompts]
            negative_token_lengths = [self.embedder.compute_token_length(prompt) for prompt in negative_prompts]

            # Filter out prompts with too many tokens
            valid_positive_indices = [i for i in range(len(positive_token_lengths)) if positive_token_lengths[i] <= 77 and negative_token_lengths[i] <= 77]
            valid_negative_indices = [i for i in range(len(negative_token_lengths)) if negative_token_lengths[i] <= 77 and negative_token_lengths[i] <= 77]

            # Process only valid prompts
            valid_positive_prompts = [positive_prompts[i] for i in valid_positive_indices]
            valid_negative_prompts = [negative_prompts[i] for i in valid_negative_indices]

            # Get embeddings for the batch
            start=time.time()
            positive_embeddings = self.get_prompt_embedding(valid_positive_prompts)
            negative_embeddings = self.get_prompt_embedding(valid_negative_prompts)
            end= time.time()

            clip_time+= end-start

            # Normalize scores and calculate mean pooled embeddings for the batch
            for i, index in enumerate(valid_positive_indices):
                # Mean pooling and other processing
                positive_embedding = self.get_mean_pooled_embedding(positive_embeddings[i].unsqueeze(0))
                negative_embedding = self.get_mean_pooled_embedding(negative_embeddings[i].unsqueeze(0))
                
                # calculate positive score and variance score
                positive_score=self.get_positive_score(positive_embedding)
                positive_score = (positive_score - self.positive_mean) / self.positive_std
                variance_score= self.get_variance_score(positive_embedding, negative_embedding, positive_score)

                # Storing prompt data
                prompt = prompt_batch[index]

                # save prompt data
                prompt_data.append(PromptData(
                    positive_prompt=prompt['positive_prompt'],
                    negative_prompt=prompt['negative_prompt'],
                    positive_embedding=positive_embedding,
                    negative_embedding=negative_embedding,
                    positive_score=positive_score,
                    variance_score=variance_score
                ))
           
        # Sort and select prompts
        sorted_scored_prompts = sorted(prompt_data, key=lambda data: data.positive_score, reverse=True)
        chosen_scored_prompts = sorted_scored_prompts[:num_prompts]

        # Calculate phrase embeddings and token lengths for chosen prompts
        print("Calculating phrase embeddings and token lengths for each phrase in each prompt")
        for prompt in tqdm(chosen_scored_prompts):
            phrases = prompt.get_phrases()
            prompt.positive_phrase_embeddings = [self.load_phrase_embedding(phrase) for phrase in phrases]
            prompt.positive_phrase_token_lengths = [self.load_phrase_token_length(phrase) for phrase in phrases] 

        total_end=time.time() 
        self.generation_time= total_end - total_start  
        self.clip_speed= (num_prompts * 2) / clip_time

        return chosen_scored_prompts

    # generate initial prompts with top k
    def generate_initial_prompts_with_fixed_probs(self, num_prompts):
        total_start=time.time()
        print("---------generating initial prompts")
        # number of prompts to generate
        prompt_count= int(num_prompts / self.top_k) if num_prompts>10 else 100

        prompts = generate_prompts_from_csv_with_base_prompt_prefix(csv_dataset_path=self.csv_phrase,
                                                                csv_base_prompts_path=self.csv_base_prompts,
                                                                prompt_count=prompt_count)
            
        prompt_data=[]
        clip_time=0
        # calculate scores and rank
        print("---------scoring prompts")
        for batch_start in tqdm(range(0, len(prompts), self.clip_batch_size)):
            batch_end = batch_start + self.clip_batch_size
            prompt_batch = prompts[batch_start:batch_end]

            # Prepare data for batch processing
            positive_prompts = [p.positive_prompt_str for p in prompt_batch]
            negative_prompts = [p.negative_prompt_str for p in prompt_batch]

            # Compute token lengths for the batch
            positive_token_lengths = [self.embedder.compute_token_length(prompt) for prompt in positive_prompts]
            negative_token_lengths = [self.embedder.compute_token_length(prompt) for prompt in negative_prompts]

            # Filter out prompts with too many tokens
            valid_positive_indices = [i for i in range(len(positive_token_lengths)) if positive_token_lengths[i] <= 77 and negative_token_lengths[i] <= 77]
            valid_negative_indices = [i for i in range(len(negative_token_lengths)) if negative_token_lengths[i] <= 77 and negative_token_lengths[i] <= 77]

            # Process only valid prompts
            valid_positive_prompts = [positive_prompts[i] for i in valid_positive_indices]
            valid_negative_prompts = [negative_prompts[i] for i in valid_negative_indices]

            # Get embeddings for the batch
            start=time.time()
            positive_embeddings = self.get_prompt_embedding(valid_positive_prompts)
            negative_embeddings = self.get_prompt_embedding(valid_negative_prompts)
            end= time.time()

            clip_time+= end-start

            # Normalize scores and calculate mean pooled embeddings for the batch
            for i, index in enumerate(valid_positive_indices):
                # Mean pooling and other processing
                positive_embedding = self.get_mean_pooled_embedding(positive_embeddings[i].unsqueeze(0))
                negative_embedding = self.get_mean_pooled_embedding(negative_embeddings[i].unsqueeze(0))
                
                # calculate positive score and variance score
                positive_score=self.get_positive_score(positive_embedding)
                positive_score = (positive_score - self.positive_mean) / self.positive_std
                variance_score= self.get_variance_score(positive_embedding, negative_embedding, positive_score)

                # Storing prompt data
                prompt = prompt_batch[index]
                # Strip whitespace and filter out empty phrases
                positive_prompt = [phrase for phrase in prompt.positive_prompt_str.split(', ') if phrase!=""]
                prompt.positive_prompt_str = ', '.join(positive_prompt)

                # save prompt data
                prompt_data.append(PromptData(
                    positive_prompt=prompt.positive_prompt_str,
                    negative_prompt=prompt.negative_prompt_str,
                    positive_embedding=positive_embedding,
                    negative_embedding=negative_embedding,
                    positive_score=positive_score,
                    variance_score=variance_score
                ))
           
        # Sort and select prompts
        sorted_scored_prompts = sorted(prompt_data, key=lambda data: data.positive_score, reverse=True)
        chosen_scored_prompts = sorted_scored_prompts[:num_prompts]

        # Calculate phrase embeddings and token lengths for chosen prompts
        print("Calculating phrase embeddings and token lengths for each phrase in each prompt")
        for prompt in tqdm(chosen_scored_prompts):
            phrases = prompt.get_phrases()
            prompt.positive_phrase_embeddings = [self.load_phrase_embedding(phrase) for phrase in phrases]
            prompt.positive_phrase_token_lengths = [self.load_phrase_token_length(phrase) for phrase in phrases] 

        total_end=time.time() 
        self.generation_time= total_end - total_start  
        self.clip_speed= (num_prompts * 2) / clip_time

        return chosen_scored_prompts

    # load the embedding of a phrase
    def load_phrase_embedding(self, phrase):
        # get the phrase index
        index=self.phrase_index_dictionarry.get(phrase)

        if index is not None:
            return self.phrase_embeddings[index]
        else:
            return self.base_prompt_embeddings[phrase]
    
    # load the token length of a phrase
    def load_phrase_token_length(self, phrase):
        # get the phrase index
        index=self.phrase_index_dictionarry.get(phrase)

        if index is not None:
            return self.phrase_token_lengths[index]
        else:
            return self.base_prompt_token_lengths[phrase]
            
    # get paths for embeddings of all prompts in a dataset
    def get_embedding_paths(self, dataset):
            objects=self.minio_client.list_objects('datasets', dataset, recursive=True)
            embedding_files = []
            for obj in objects: 
                if obj.object_name.endswith("_embedding.msgpack"):
                    embedding_files.append(obj.object_name)
                    
            return embedding_files

    # store list of initial prompts in a csv to use for prompt mutation
    def store_prompts_in_csv_file(self, data):
        minio_path=f"{self.model_dataset}/output/generated-prompts-csv"
        local_path="output/generated_prompts.csv"
        pd.DataFrame(data).to_csv(local_path, index=False)
        # Read the contents of the CSV file
        with open(local_path, 'rb') as file:
            csv_content = file.read()

        #Upload the CSV file to Minio
        buffer = io.BytesIO(csv_content)
        buffer.seek(0)

        current_date=datetime.now().strftime("%Y-%m-%d-%H:%M")
        minio_path= minio_path + f"/{current_date}-{GENERATION_POLICY}-{self.model_dataset}.csv"
        cmd.upload_data(self.minio_client, 'datasets', minio_path, buffer)
        # Remove the temporary file
        os.remove(local_path)

    # outputs a graph of average score improvement in each iteration
    def score_improvement_graph(self, minio_path):
        plt.plot(range(1, self.max_iterations+1), self.average_score_by_iteration)
        plt.xlabel('Iterations')
        plt.ylabel('Average Score')
        plt.title('Average score in each iteration')

        plt.savefig("output/average_score_by_iteration.png")

        # Save the figure to a file
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # upload the graph report
        minio_path= minio_path + "/average_score_by_iteration.png"
        cmd.upload_data(self.minio_client, 'datasets', minio_path, buf)
        # Remove the temporary file
        os.remove("output/average_score_by_iteration.png")
        # Clear the current figure
        plt.clf()

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
        # Clear the current figure
        plt.clf()

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
        content += f"Initial top-k prompt generation time: {self.generation_time:.2f} seconds\n"
        content += f"Prompt Mutation Time: {self.mutation_time:.2f} seconds\n\n"

        content += f"================ Model Stats ==================\n"
        content += f"Clip batch size: {self.clip_batch_size}\n"
        content += f"Clip embedding speed: {self.clip_speed:.2f} embeddings/second\n"
        content += f"Xgboost batch size: {self.substitution_batch_size}\n"
        content += f"Xgboost inference speed: {self.inference_speed:.2f} predictions/second\n\n"

        content += f"================ Generator Parameters ==================\n"
        content += f"Number of Iterations: {self.max_iterations}\n"
        content += f"Number of substitution choices per iteration: {self.num_choices_per_iteration}\n\n"

        content += f"================ Results ==================\n"
        content += f"Average sigma score before mutation: {avg_score_before_mutation:.2f}\n"
        content += f"Average sigma score after mutation: {avg_score_after_mutation:.2f}\n\n"

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
        phrase_list=self.phrase_list
        phrase_embeddings_list=[]
        
        for phrase in tqdm(phrase_list):
            embedding= self.get_prompt_embedding(phrase)
            mean_pooled_embedding= self.get_mean_pooled_embedding(embedding)
            phrase_embeddings_list.append(mean_pooled_embedding)
        
        # Convert the list of numpy arrays to a 2D numpy array
        phrase_embeddings = np.array(phrase_embeddings_list)

        # Save the numpy array to an .npz file
        if self.initial_generation_policy=="fixed_probabilities":
            local_file_path='phrase_embeddings.npz'
        else:
            local_file_path='independant_approx_phrase_embeddings.npz'

        np.savez_compressed(local_file_path, phrase_embeddings)

        # Read the contents of the .npz file
        with open(local_file_path, 'rb') as file:
            content = file.read()

        # Upload the local file to MinIO
        buffer = io.BytesIO(content)
        buffer.seek(0)

        minio_path=DATA_MINIO_DIRECTORY + f"/input/{local_file_path}"
        cmd.upload_data(self.minio_client, 'datasets',minio_path, buffer)

        # Remove the temporary file
        os.remove(local_file_path)
    
    # store toke nlength of all phrases in civitai in a file in minIO
    def store_phrase_token_lengths(self):
        phrase_list=self.phrase_list
        token_lengths = [self.get_token_length(phrase) for phrase in tqdm(phrase_list)]

        # Save the numpy array to an .npz file
        if self.initial_generation_policy=="fixed_probabilities":
            local_file_path='token_lengths.csv'
        else:
            local_file_path='independant_approx_token_lengths.csv'
        
        # Write to a CSV file
        with open(local_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Phrase index', 'Phrase str', 'Token Length'])
            for phrase_index, length in enumerate(token_lengths):
                writer.writerow([phrase_index, phrase_list[phrase_index], length])
        
        # Read the contents of the .npz file
        with open(local_file_path, 'rb') as file:
            content = file.read()

        # Upload the local file to MinIO
        buffer = io.BytesIO(content)
        buffer.seek(0)

        minio_path=DATA_MINIO_DIRECTORY + f"/input/{local_file_path}"
        cmd.upload_data(self.minio_client, 'datasets',minio_path, buffer)

        # Remove the temporary file
        os.remove(local_file_path)

    # get civitai phrase embeddings from minIO
    def load_phrase_embeddings(self):
        # get file name
        if self.initial_generation_policy=="fixed_probabilities":
            filename='phrase_embeddings.npz'
        else:
            filename='independant_approx_phrase_embeddings.npz'

        # Get the file data from MinIO
        minio_path = f"environmental/data/prompt-generator/substitution/input/{filename}"
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
    
    # get civitai phrase token lengths, calculated by the tokenizer
    def load_phrase_token_lengths(self):
        # get file name
        if self.initial_generation_policy=="fixed_probabilities":
            filename='token_lengths.csv'
        else:
            filename='independant_approx_token_lengths.csv'

        # Get the file data from MinIO
        minio_path = f"environmental/data/prompt-generator/substitution/input/{filename}"
        # Download the file from MinIO
        try:
            data = cmd.get_file_from_minio(self.minio_client, 'datasets', minio_path)
            data_stream = io.BytesIO(data.read())  # Read data into an in-memory stream
        except Exception as e:
            print(f"Error downloading file from MinIO: {e}")
            return None
        
        # Read the contents directly into a Pandas DataFrame
        phrase_df = pd.read_csv(data_stream)

        # get token lengths array
        token_lengths=phrase_df['Token Length'].tolist()
        
        return token_lengths


def main():
    args = parse_args()

    # set the minio data path
    global DATA_MINIO_DIRECTORY
    DATA_MINIO_DIRECTORY= f"{args.model_dataset}/" + DATA_MINIO_DIRECTORY

    # set the base prompts csv path
    if(args.model_dataset=="icons"):
        csv_base_prompts='input/dataset-config/icon/base-prompts-dsp.csv'
    elif(args.model_dataset=="propaganda-poster"):
        csv_base_prompts='input/dataset-config/propaganda-poster/base-prompts-propaganda-poster.csv'
    elif(args.model_dataset=="mech"):
        csv_base_prompts='input/dataset-config/mech/base-prompts-dsp.csv'
    elif(args.model_dataset=="character" or args.model_dataset=="waifu"):
        csv_base_prompts='input/dataset-config/character/base-prompts-waifu.csv'
    elif(args.model_dataset=="environmental"):  
        csv_base_prompts='input/dataset-config/environmental/base-prompts-environmental.csv'

    prompt_mutator= PromptSubstitutionGenerator(minio_access_key=args.minio_access_key,
                                  minio_secret_key=args.minio_secret_key,
                                  minio_ip_addr=args.minio_addr,
                                  csv_phrase=args.csv_phrase,
                                  csv_base_prompts=csv_base_prompts,
                                  model_dataset=args.model_dataset,
                                  substitution_model=args.substitution_model,
                                  scoring_model=args.scoring_model,
                                  max_iterations=args.max_iterations,
                                  sigma_threshold=args.sigma_threshold,
                                  variance_weight=args.variance_weight,
                                  boltzman_temperature=args.boltzman_temperature,
                                  boltzman_k=args.boltzman_k,
                                  dataset_name=args.dataset_name,
                                  store_embeddings=args.store_embeddings,
                                  store_token_lengths=args.store_token_lengths,
                                  self_training=args.self_training,
                                  send_job=args.send_job,
                                  save_csv=args.save_csv,
                                  initial_generation_policy=args.initial_generation_policy,
                                  top_k=args.top_k,
                                  num_choices_per_iteration=args.num_choices,
                                  clip_batch_size=args.clip_batch_size,
                                  substitution_batch_size=args.substitution_batch_size)
    
    # generate n number of images
    prompt_mutator.generate_images(num_images=args.n_data)
    
if __name__ == "__main__":
    main()