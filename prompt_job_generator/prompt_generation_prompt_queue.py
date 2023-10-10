import sys
import queue

base_directory = "./"
sys.path.insert(0, base_directory)

from worker.prompt_generation.prompt_generator import generate_prompts_proportional_selection, generate_base_prompts

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

    def update(self, prompt_job_generator_state, dataset):
        if dataset not in self.queue_dictionary:
            self.queue_dictionary[dataset] = queue.Queue()

        dataset_queue = self.queue_dictionary[dataset]

        if dataset_queue.qsize() < self.queue_size:
            prompt_count = self.queue_size - dataset_queue.qsize()
            prompts = self.generate_prompts(prompt_job_generator_state, dataset, prompt_count)
            for prompt in prompts:
                dataset_queue.put(prompt)
        else:
            print("Queue is full. Element not added.")

    def generate_prompts(self, prompt_job_generator_state, dataset, prompt_count):

        clip_text_embedder = prompt_job_generator_state.clip_text_embedder

        base_prompts_csv_path = self.get_dataset_base_prompt(dataset)

        print('dataset : ', dataset)
        print('base_prompts_csv_path ', base_prompts_csv_path)

        # number of total prompts to generate before choosing n prompts
        total_prompt_count = prompt_count * 8

        if base_prompts_csv_path is None:
            print('base prompt file is not found for dataset ', dataset)


        efficient_net_model = prompt_job_generator_state.get_efficient_net_model(dataset)

        if efficient_net_model is None:
            print('efficient net model is not found for dataset ', dataset)

        prompts = generate_prompts_proportional_selection(prompt_job_generator_state.phrases,
                                                          prompt_job_generator_state.phrases_token_size,
                                                          prompt_job_generator_state.positive_count_list,
                                                          prompt_job_generator_state.negative_count_list,
                                                          total_prompt_count,
                                                          '')

        scored_prompts = []
        for prompt in prompts:

            # N Base Prompt Phrases
            # Hard coded probability of choose 0,1,2,3,4,5, etc base prompt phrases
            # Chance for 0 base prompt phrases should be 30%
            # choose_probability = [0.3, 0.3, 0.2, 0.2, 0.2]
            choose_probability = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

            base_prompt_list = generate_base_prompts(base_prompts_csv_path, choose_probability)

            base_prompts = ''

            for base_prompt in base_prompt_list:
                base_prompts = base_prompts + base_prompt + ', '

            positive_text_prompt = base_prompts + prompt.positive_prompt_str
            negative_text_prompt = prompt.negative_prompt_str

            prompt_score = 0
            if efficient_net_model is not None and clip_text_embedder is not None:
                # get prompt embeddings
                positive_prompt_embeddings = clip_text_embedder(positive_text_prompt)
                negative_prompt_embeddings = clip_text_embedder(negative_text_prompt)

                prompt_score = efficient_net_model.predict_positive_negative(positive_prompt_embeddings,
                                                                             negative_prompt_embeddings).item()

            scored_prompt = ScoredPrompt(prompt_score, positive_text_prompt, negative_text_prompt)
            scored_prompts.append(scored_prompt)

        # Sort the list based on the maximize_int1 function
        sorted_scored_prompts = sorted(scored_prompts, key=maximize_score)

        chosen_scored_prompts = sorted_scored_prompts[:prompt_count]

        return chosen_scored_prompts

class ScoredPrompt:
    def __init__(self, score, positive_prompt, negative_prompt):
        self.score = score
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt


def maximize_score(scored_prompt):
    return -scored_prompt.score