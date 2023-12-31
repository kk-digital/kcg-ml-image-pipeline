import argparse
from datetime import datetime
import io
import os
import random
import sys
import time
import traceback
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch

base_directory = "./"
sys.path.insert(0, base_directory)

from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from training_worker.ab_ranking.model.ab_ranking_linear import ABRankingModel
from stable_diffusion.model.clip_text_embedder.clip_text_embedder import CLIPTextEmbedder
from utility.boltzman.boltzman_phrase_scores_loader import BoltzmanPhraseScoresLoader
from utility.minio import cmd

from prompt_job_generator.independent_approx_v1.independent_approx_v1 import IndependentApproxV1
from worker.prompt_generation.prompt_generator import generate_image_generation_jobs

GENERATION_POLICY="independent-approx-substitution-search-v1"
DATA_MINIO_DIRECTORY="environmental/data/prompt-generator/substitution"

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
    parser.add_argument('--max-iterations', type=int, help="number of mutation iterations", default=150)
    parser.add_argument('--save-csv', action='store_true', default=False)
    parser.add_argument('--top-k', type=float, help="top percentage of prompts taken from generation to be mutated", default=0.1)
    parser.add_argument('--boltzman-temperature', type=int, default=64)
    parser.add_argument('--boltzman-k', type=float, default=1.0)
    parser.add_argument(
        '--csv_base_prompts', help='CSV containing base prompts', 
        default='input/dataset-config/environmental/base-prompts-environmental.csv'
    )

    return parser.parse_args()

class BoltzmanPromptSubstitutionGenerator:
    def __init__(
        self,
        minio_access_key,
        minio_secret_key,
        minio_ip_addr,
        csv_phrase,
        csv_base_prompts,
        scoring_model,
        max_iterations,
        dataset_name,
        send_job,
        save_csv,
        top_k,
        boltzman_temperature,
        boltzman_k
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
        # name of dataset
        self.dataset_name=dataset_name
        # whether to send jobs to server or not
        self.send_job=send_job
        # whether to save csv of prompts or not
        self.save_csv=save_csv
        # substitution model (binary of sigma score)
        self.substitution_model= None
        # top k value for generating initial prompts
        self.top_k=top_k
        # get list of base prompts
        self.csv_base_prompts=csv_base_prompts
        # boltzman temperature and k value
        self.boltzman_temperature = boltzman_temperature
        self.boltzman_k = boltzman_k

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
        

        # get list of boltzman phrase score
        self.positive_phrase_scores_csv,self.negative_phrase_scores_csv=self.get_boltzman_scores_csv()
        print(self.positive_phrase_scores_csv)
        print(self.negative_phrase_scores_csv)

        # loading positive phrase scores to use for rejection sampling
        phrase_loader=BoltzmanPhraseScoresLoader(dataset_name="environmental",
                                                 phrase_scores_csv=self.positive_phrase_scores_csv,
                                                 minio_client=self.minio_client)
        phrase_loader.load_dataset()
        self.phrase_score_data= phrase_loader.index_phrase_score_data
        print(f"number of positive phrase scores {len(self.phrase_score_data)}")
 
        # get dictionarry of indexes by phrase to make lookup faster
        self.phrase_dictionarry={self.phrase_score_data[index].phrase:index for index in range(len(self.phrase_score_data))}

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

        embedding_model.load_pth(byte_buffer)
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
    
    # function to choose a random phrase under the max token length
    def choose_random_phrase(self, max_token_length):
        random_index=random.randrange(0, len(self.phrase_score_data))
        phrase_data= self.phrase_score_data[random_index]

        while(phrase_data.token_length> max_token_length):
            random_index=random.randrange(0, len(self.phrase_score_data))
            phrase_data= self.phrase_score_data[random_index]
        
        return phrase_data

    # function for rejection sampling with sigma scores
    def rejection_sampling_by_boltzman_score(self,
                                    prompt_str,
                                    phrases_energy,
                                    phrase_token_lengths):

        # get number of phrases
        prompt_list = prompt_str.split(', ')
        num_phrases= len(prompt_list)
        # get number of tokens
        current_length= sum(phrase_token_lengths) + num_phrases - 1
        # list of potential substitution choices for current iteration
        substitution_choices=[]

        # create a substitution for each position in the prompt
        for phrase_index in range(num_phrases):
            # get max accepted token length for substitute phrase
            max_token_length= 75 - (current_length - phrase_token_lengths[phrase_index]) 
            # get a random phrase from civitai to substitute with
            random_phrase=self.choose_random_phrase(max_token_length)
            # get phrase string
            substitute_phrase = random_phrase.phrase
            # get phrase score by its index
            substitute_energy = random_phrase.energy_per_token
            # get the substituted phrase energy
            substituted_energy = phrases_energy[phrase_index]

            # compare the energy of substitute and substituted phrase
            if substitute_energy > substituted_energy:
                substitution_data={
                    'position':phrase_index,
                    'substitute_phrase':substitute_phrase,
                    'substitute_energy': substitute_energy,
                    'substitute_token_length': random_phrase.token_length,
                    'increase':substitute_energy - substituted_energy,
                }
                substitution_choices.append(substitution_data)
            
        # substitutions are sorted from highest sigma score to lowest
        substitution_choices= sorted(substitution_choices, key=lambda s: s['increase'], reverse=True) 
        
        return substitution_choices

    # function for mutating a prompt
    def mutate_prompt(self,
                    prompt_str,
                    prompt_embedding,
                    prompt_score):
        
        phrase_scores=[]
        phrase_token_lengths=[]
        for phrase in prompt_str.split(', '):
            phrase_scores.append(self.phrase_score_data[self.phrase_dictionarry[phrase]].energy_per_token)
            phrase_token_lengths.append(self.phrase_score_data[self.phrase_dictionarry[phrase]].token_length)

        rejection_policy_time=0
        substitution_time=0
        num_attempts=0
        num_success=0
        
        # run mutation process for a set number of iterations
        for i in range(self.max_iterations):
            
            start= time.time()
            # return a list of potential substitution choices, filtered by the rejection policy
            substitution_choices=self.rejection_sampling_by_boltzman_score(prompt_str, phrase_scores, phrase_token_lengths)
            end= time.time()

            rejection_policy_time+= end - start
            
            start= time.time()
            # test every choice and take the first choice that increases score
            for substitution in substitution_choices:
                # get substitution data
                position=substitution['position']
                substitute_phrase=substitution['substitute_phrase']

                #Create a modified prompt with the substitution
                prompt_list = prompt_str.split(', ')
                prompt_list[position] = substitute_phrase
                modified_prompt_str = ", ".join(prompt_list)

                #calculate modified prompt embedding and sigma score
                modified_prompt_embedding=self.get_prompt_embedding(modified_prompt_str)
                modified_prompt_score= self.get_prompt_score(modified_prompt_embedding)
                modified_prompt_score= (modified_prompt_score - self.positive_mean) / self.positive_std

                num_attempts+=1
                # check if score improves
                if(prompt_score < modified_prompt_score):
                    # if it does improve, the new prompt is saved and it jumps to the next iteration
                    prompt_str= modified_prompt_str
                    prompt_embedding= modified_prompt_embedding
                    prompt_score= modified_prompt_score
                    phrase_scores[position]=substitution['substitute_energy']
                    phrase_token_lengths[position]=substitution['substitute_token_length']
                    num_success+=1
                    break
            
            self.average_score_by_iteration[i]+=prompt_score
            end= time.time()
            substitution_time+= end - start
        
        print(f"time for rejection policy {rejection_policy_time}")
        print(f"time for substitutions {substitution_time}")
        print(f"success rate: {num_success}/{num_attempts}")
 
        return prompt_str, prompt_embedding

    # function to generate n images
    def generate_images(self, num_images):
        # dataframe for saving csv of generated prompts
        df_data=[]
        # scores before mutation
        original_scores=[]
        # scores after mutation
        mutated_scores=[]

        index=0

        # get initial prompts
        prompt_list = self.generate_initial_prompts(num_images)
    
        start=time.time()
        # mutate prompts one by one
        for prompt in prompt_list:

            #getting negative and positive prompts
            positive_prompt=prompt['positive_prompt'] 
            negative_prompt=prompt['negative_prompt']
            
            #getting prompt embeddings
            positive_embedding=prompt['positive_embedding']
            negative_embedding=prompt['negative_embedding']

            # calculating combined score and positive score of prompt before mutation
            seed_score=prompt['score'] 
            positive_score=prompt['positive_score'] 
            # substract mean and divide by std to get sigma scores
            positive_score= (positive_score - self.positive_mean) / self.positive_std
            seed_sigma_score=(seed_score - self.mean) / self.std
            # append to the list of scores before mutation
            original_scores.append(seed_sigma_score)

            #mutate positive prompt
            mutated_positive_prompt, mutated_positive_embedding= self.mutate_prompt(
                            prompt_str=positive_prompt, 
                            prompt_embedding=positive_embedding,
                            prompt_score=positive_score)

            # calculating new score with the mutated positive prompt
            score=self.scorer.predict(mutated_positive_embedding, negative_embedding).item()
            sigma_score=(score - self.mean) / self.std
            # append to list of scores after mutation
            mutated_scores.append(sigma_score)

            print(f"prompt {index} mutated.")
            print(f"----initial score: {seed_score}.")
            print(f"----final score: {score}.")

            task_uuid=-1
            task_time = -1
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

        # save generated prompts in csv
        if self.save_csv:
            self.store_prompts_in_csv_file(df_data)

        # creating path to save generation data
        current_date=datetime.now().strftime("%Y-%m-%d-%H:%M")
        generation_path=DATA_MINIO_DIRECTORY + f"/generated-images/{current_date}-generated-data"
        
        # save a histogram of score distribution before and after mutation for comparison
        self.compare_distributions(generation_path, original_scores, mutated_scores)
        # save a txt file containing generation stats
        self.generation_stats(minio_path=generation_path,
                        generation_speed=generation_speed,
                        num_prompts=num_images,
                        avg_score_before_mutation= np.mean(original_scores),
                        avg_score_after_mutation= np.mean(mutated_scores),
                        )
        
        # save a graph for score improvement by number of iterations
        self.average_score_by_iteration= np.divide(self.average_score_by_iteration, num_images)
        self.score_improvement_graph(minio_path=generation_path)

    # function to generate initial prompts
    def generate_initial_prompts(self, num_prompts):
        # generate initial prompts before mutation
        prompt_generator = IndependentApproxV1("environmental", self.boltzman_temperature, self.boltzman_k)
        prompt_generator.load_csv(self.minio_client, self.positive_phrase_scores_csv, self.negative_phrase_scores_csv)
        
        prompts =  prompt_generator.generate_prompts(prompt_count=int(num_prompts/self.top_k))
        
        prompt_data=[]
        # add base prompts and calculate scores
        print("---------scoring prompts")
        for i, prompt in enumerate(prompts):
            print(f"prompt {i}")

            # get positive and negative prompt
            positive_prompt = prompt['positive_prompt']
            negative_prompt = prompt['negative_prompt']

            try:
                # get positive and negative embeddings
                positive_embedding=self.get_prompt_embedding(positive_prompt)
                negative_embedding=self.get_prompt_embedding(negative_prompt)
            except:
                print(positive_prompt)
                print(negative_prompt)

           
            # calculating combined score and positive score of prompt
            with torch.no_grad():
                prompt_score=self.scorer.predict(positive_embedding, negative_embedding).item()
                positive_score=self.positive_scorer.predict_positive_or_negative_only(positive_embedding).item()

            # storing prompt data
            prompt_data.append({
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "positive_embedding": positive_embedding,
                "negative_embedding" : negative_embedding,
                "score": prompt_score,
                "positive_score": positive_score
            })
        
        # Sort the list based on score
        sorted_scored_prompts = sorted(prompt_data, key=lambda s: s['score'], reverse=True)
        chosen_scored_prompts = sorted_scored_prompts[:num_prompts]

        return chosen_scored_prompts

    # store list of initial prompts in a csv to use for prompt mutation
    def store_prompts_in_csv_file(self, data):
        minio_path="environmental/output/generated-prompts-csv"
        local_path="output/generated_prompts.csv"
        pd.DataFrame(data).to_csv(local_path, index=False)
        # Read the contents of the CSV file
        with open(local_path, 'rb') as file:
            csv_content = file.read()

        #Upload the CSV file to Minio
        buffer = io.BytesIO(csv_content)
        buffer.seek(0)

        current_date=datetime.now().strftime("%Y-%m-%d-%H:%M")
        minio_path= minio_path + f"/{current_date}-{GENERATION_POLICY}-environmental.csv"
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


def main():
    args = parse_args()
    prompt_mutator= BoltzmanPromptSubstitutionGenerator(minio_access_key=args.minio_access_key,
                                  minio_secret_key=args.minio_secret_key,
                                  minio_ip_addr=args.minio_addr,
                                  csv_phrase=args.csv_phrase,
                                  csv_base_prompts=args.csv_base_prompts,
                                  scoring_model=args.scoring_model,
                                  max_iterations=args.max_iterations,
                                  dataset_name=args.dataset_name,
                                  send_job=args.send_job,
                                  save_csv=args.save_csv,
                                  top_k=args.top_k,
                                  boltzman_temperature=args.boltzman_temperature,
                                  boltzman_k=args.boltzman_k)
    
    # generate n number of images
    prompt_mutator.generate_images(num_images=args.n_data)
    
if __name__ == "__main__":
    main()