import sys
import queue
import math

base_directory = "./"
sys.path.insert(0, base_directory)

from worker.prompt_generation.prompt_generator import generate_prompts_proportional_selection, generate_base_prompts, load_base_prompts

class PromptGenerationPromptQueue:
    def __init__(self, queue_size):
        # prompt queue
        # contains the prompts used in generation jobs
        self.queue_dictionary = {}
        self.queue_size = queue_size

        # dataset dictionary for base prompts
        # maps dataset => csv base prompt path
        self.dataset_base_prompt_dictionary = {}

    def set_dataset_base_prompt(self, dataset, base_prompt_path):
        self.dataset_base_prompt_dictionary[dataset] = base_prompt_path

    def get_dataset_base_prompt(self, dataset):
        if dataset in self.dataset_base_prompt_dictionary:
            return self.dataset_base_prompt_dictionary[dataset]

        return None

    def get_dataset_prompt(self, dataset):
        if dataset not in self.queue_dictionary:
            return None

        dataset_queue = self.queue_dictionary[dataset]

        if dataset_queue.qsize() <= 0:
            return None

        scored_prompt = dataset_queue.get()

        return scored_prompt

    def database_prompt_available(self, dataset):
        if dataset not in self.queue_dictionary:
            return None

        dataset_queue = self.queue_dictionary[dataset]

        if dataset_queue.qsize() <= 0:
            return False

        return True

    def update(self, prompt_job_generator_state, dataset):
        if dataset not in self.queue_dictionary:
            self.queue_dictionary[dataset] = queue.Queue()

        dataset_queue = self.queue_dictionary[dataset]

        if dataset_queue.qsize() < self.queue_size:
            prompt_count = self.queue_size - dataset_queue.qsize()
            prompts = self.generate_prompts(prompt_job_generator_state, dataset, prompt_count)
            for prompt in prompts:
                dataset_queue.put(prompt)

    def generate_prompts(self, prompt_job_generator_state, dataset, prompt_count):

        generation_policy = prompt_job_generator_state.get_dataset_prompt_generation_policy(dataset)

        clip_text_embedder = prompt_job_generator_state.clip_text_embedder
        base_prompts_csv_path = self.get_dataset_base_prompt(dataset)

        top_k = prompt_job_generator_state.get_dataset_top_k(dataset)

        # number of total prompts to generate before choosing n prompts
        if generation_policy == 'top-k':
            # if the generation policy is top-k we generate
            # more prompts so that the top-k are allways equal to prompt_count
            # example:  if top-k is 0.1 and we need 1 prompt, we generate 10 and choose best one
            total_prompt_count = prompt_count * (1.0 / top_k)
        else:
            total_prompt_count = prompt_count * 8

        total_prompt_count = int(total_prompt_count)

        scoring_model = prompt_job_generator_state.get_dataset_scoring_model(dataset)

        base_prompt_population = load_base_prompts(base_prompts_csv_path)

        prompts = []

        if generation_policy == 'top-k':
            prompts = generate_prompts_proportional_selection(prompt_job_generator_state.phrases,
                                                              prompt_job_generator_state.phrases_token_size,
                                                              prompt_job_generator_state.positive_count_list,
                                                              prompt_job_generator_state.negative_count_list,
                                                              total_prompt_count,
                                                              '')
            prompt_list = []
            for prompt in prompts:
                # N Base Prompt Phrases
                # Hard coded probability of choose 0,1,2,3,4,5, etc base prompt phrases
                # Chance for 0 base prompt phrases should be 30%
                # choose_probability = [0.3, 0.3, 0.2, 0.2, 0.2]
                choose_probability = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

                if len(base_prompt_population) != 0:
                    base_prompt_list = generate_base_prompts(base_prompt_population, choose_probability)
                else:
                    base_prompt_list = []

                base_prompts = ''

                for base_prompt in base_prompt_list:
                    base_prompts = base_prompts + base_prompt + ', '

                positive_text_prompt = base_prompts + prompt.positive_prompt_str
                negative_text_prompt = prompt.negative_prompt_str
                prompt_list.append(ScoredPrompt(0,
                                                positive_text_prompt,
                                                negative_text_prompt,
                                                'N/A',
                                                'N/A',
                                                0))

            prompts = prompt_list

        elif generation_policy == 'combined-top-k':
            number_of_positive_prompts_to_generate = int(math.sqrt(total_prompt_count) + 1.0)

            prompts = generate_prompts_proportional_selection(prompt_job_generator_state.phrases,
                                                              prompt_job_generator_state.phrases_token_size,
                                                              prompt_job_generator_state.positive_count_list,
                                                              prompt_job_generator_state.negative_count_list,
                                                              number_of_positive_prompts_to_generate,
                                                              '')
            positive_prompts = []
            negative_prompts = []

            for prompt in prompts:
                # N Base Prompt Phrases
                # Hard coded probability of choose 0,1,2,3,4,5, etc base prompt phrases
                # Chance for 0 base prompt phrases should be 30%
                # choose_probability = [0.3, 0.3, 0.2, 0.2, 0.2]
                choose_probability = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

                base_prompt_list = generate_base_prompts(base_prompt_population, choose_probability)

                base_prompts = ''

                for base_prompt in base_prompt_list:
                    base_prompts = base_prompts + base_prompt + ', '

                positive_text_prompt = base_prompts + prompt.positive_prompt_str
                negative_text_prompt = prompt.negative_prompt_str

                positive_prompts.append(positive_text_prompt)
                negative_prompts.append(negative_text_prompt)

            # Create a list of all possible combinations
            prompts = [ScoredPrompt(0, positive, negative, 'N/A', 'N/A', 0) for positive in positive_prompts for negative in
                                negative_prompts]

            # remove excess prompts, will only remove a tiny bit of prompts
            # so no one cares
            prompts = prompts[:total_prompt_count]

        scored_prompts = []
        for prompt in prompts:

            positive_text_prompt = prompt.positive_prompt
            negative_text_prompt = prompt.negative_prompt

            prompt_score = 0
            if scoring_model is not None and clip_text_embedder is not None:
                # get prompt embeddings
                positive_prompt_embeddings = clip_text_embedder(positive_text_prompt)
                negative_prompt_embeddings = clip_text_embedder(negative_text_prompt)

                prompt_score = scoring_model.predict(positive_prompt_embeddings,
                                                               negative_prompt_embeddings).item()

            scored_prompt = ScoredPrompt(prompt_score,
                                         positive_text_prompt,
                                         negative_text_prompt,
                                         scoring_model.model_type,
                                         generation_policy,
                                         top_k)
            scored_prompts.append(scored_prompt)

        # Sort the list based on the maximize_int1 function
        sorted_scored_prompts = sorted(scored_prompts, key=maximize_score)

        chosen_scored_prompts = sorted_scored_prompts[:prompt_count]

        return chosen_scored_prompts

class ScoredPrompt:
    def __init__(self, score,
                 positive_prompt,
                 negative_prompt,
                 scoring_model,
                 generation_policy,
                 top_k):
        self.score = score
        self.scoring_model = scoring_model
        self.generation_policy = generation_policy
        self.top_k = top_k
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt


def maximize_score(scored_prompt):
    return -scored_prompt.score