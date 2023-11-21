import sys
import threading
import io
import random

base_directory = "./"
sys.path.insert(0, base_directory)

from configs.model_config import ModelPathConfig
from stable_diffusion.model_paths import (SDconfigs, CLIPconfigs)
from stable_diffusion import CLIPTextEmbedder
from utility.minio import cmd
from training_worker.ab_ranking.model.ab_ranking_efficient_net import ABRankingEfficientNetModel
from training_worker.ab_ranking.model.ab_ranking_linear import ABRankingModel
from training_worker.ab_ranking.model.ab_ranking_elm_v1 import ABRankingELMModel
from worker.prompt_generation.prompt_generator import (initialize_prompt_list_from_csv)
from prompt_generation_prompt_queue import PromptGenerationPromptQueue
from prompt_job_generator_constants import (PROMPT_QUEUE_SIZE, DEFAULT_PROMPT_GENERATION_POLICY,
                                            DEFAULT_TOP_K_VALUE, DEFAULT_DATASET_RATE, DEFAULT_HOURLY_LIMIT)
from utility.path import  separate_bucket_and_file_path

class PromptJobGeneratorState:
    def __init__(self, device):
        self.total_rate = 0
        # keep the dataset_job_queue_size in this dictionary
        # should update using orchestration api
        self.dataset_job_queue_size = {}
        self.dataset_job_queue_target = {}
        self.dataset_job_queue_size_lock = threading.Lock()
        # used to store prompt generation data like top-k, dataset_rate value
        self.dataset_prompt_generation_data_dictionary = {}
        self.dataset_prompt_generation_data_lock = threading.Lock()
        # each dataset will have a list of masks
        # only relevent if its an inpainting job
        self.dataset_masks = {}
        # each dataset will have one callback to spawn the jobs
        self.dataset_callbacks = {}
        # efficient net model we use for scoring prompts
        # each dataset will have its own  model
        # input : prompts
        # output : prompt_score
        self.prompt_efficient_net_model_dictionary = {}
        self.prompt_linear_model_dictionary = {}
        self.prompt_elm_v1_model_dictionary = {}
        self.dataset_model_list = {}
        self.dataset_model_lock = threading.Lock()

        # minio connection
        self.minio_client = None

        self.prompt_queue = PromptGenerationPromptQueue(PROMPT_QUEUE_SIZE)

        self.phrases = None
        self.phrases_token_size = None
        self.positive_count_list = None
        self.negative_count_list = None
        self.device = device
        self.config = ModelPathConfig()
        self.clip_text_embedder = CLIPTextEmbedder(device=self.device)

    def configure_minio(self, minio_access_key, minio_secret_key):
        self.minio_client = cmd.get_minio_client(minio_access_key, minio_secret_key)

    def load_clip_model(self):
        # Load the clip model
        self.clip_text_embedder.load_submodels(
            tokenizer_path=self.config.get_model_folder_path(CLIPconfigs.TXT_EMB_TOKENIZER),
            transformer_path=self.config.get_model_folder_path(CLIPconfigs.TXT_EMB_TEXT_MODEL)
        )

    def load_efficient_net_model(self, dataset, dataset_bucket, model_path):

        efficient_net_model = ABRankingEfficientNetModel(in_channels=2)

        model_file_data = cmd.get_file_from_minio(self.minio_client, dataset_bucket, model_path)

        if model_file_data is None:
            return

        # Create a BytesIO object and write the downloaded content into it
        byte_buffer = io.BytesIO()
        for data in model_file_data.stream(amt=8192):
            byte_buffer.write(data)
        # Reset the buffer's position to the beginning
        byte_buffer.seek(0)

        efficient_net_model.load(byte_buffer)

        with self.dataset_model_lock:
            self.prompt_efficient_net_model_dictionary[dataset] = efficient_net_model

    def get_efficient_net_model(self, dataset):
        # try to get the efficient net model
        # if the efficient net model is not found
        # for the dataset return None
        with self.dataset_model_lock:
            if dataset in self.prompt_efficient_net_model_dictionary:
                return self.prompt_efficient_net_model_dictionary[dataset]

        return None

    def load_linear_model(self, dataset, dataset_bucket, model_path):

        linear_model = ABRankingModel(768*2)

        model_file_data = cmd.get_file_from_minio(self.minio_client, dataset_bucket, model_path)

        if model_file_data is None:
            print(f'count not find model file data at {model_path}')
            return

        # Create a BytesIO object and write the downloaded content into it
        byte_buffer = io.BytesIO()
        for data in model_file_data.stream(amt=8192):
            byte_buffer.write(data)
        # Reset the buffer's position to the beginning
        byte_buffer.seek(0)

        linear_model.load(byte_buffer)

        with self.dataset_model_lock:
            self.prompt_linear_model_dictionary[dataset] = linear_model

        print('Linear model loaded successfully')


    def load_elm_v1_model(self, dataset, dataset_bucket, model_path):

        elm_model = ABRankingELMModel(768*2)

        model_file_data = cmd.get_file_from_minio(self.minio_client, dataset_bucket, model_path)

        if model_file_data is None:
            print(f'count not find model file data at {model_path}')
            return

        # Create a BytesIO object and write the downloaded content into it
        byte_buffer = io.BytesIO()
        for data in model_file_data.stream(amt=8192):
            byte_buffer.write(data)
        # Reset the buffer's position to the beginning
        byte_buffer.seek(0)

        elm_model.load(byte_buffer)

        with self.dataset_model_lock:
            self.prompt_elm_v1_model_dictionary[dataset] = elm_model

        print('Elm model loaded successfully')


    def get_linear_model(self, dataset):
        # try to get the linear model
        # if the linear model is not found
        # for the dataset return None
        with self.dataset_model_lock:
            if dataset in self.prompt_linear_model_dictionary:
                return self.prompt_linear_model_dictionary[dataset]

        return None


    def get_elm_v1_model(self, dataset):
        # try to get the linear model
        # if the linear model is not found
        # for the dataset return None
        with self.dataset_model_lock:
            if dataset in self.prompt_elm_v1_model_dictionary:
                return self.prompt_elm_v1_model_dictionary[dataset]

        return None


    def load_prompt_list_from_csv(self, csv_dataset_path, csv_phrase_limit):
        phrases, phrases_token_size, positive_count_list, negative_count_list = initialize_prompt_list_from_csv(csv_dataset_path, csv_phrase_limit)

        self.phrases = phrases
        self.phrases_token_size = phrases_token_size
        self.positive_count_list = positive_count_list
        self.negative_count_list = negative_count_list

    def register_callback(self, dataset, callback):
        self.dataset_callbacks[dataset] = callback

    def get_callback(self, dataset):
        if dataset in self.dataset_callbacks:
            return self.dataset_callbacks[dataset]
        else:
            return None

    def set_dataset_model_list(self, dataset, model_list):
        with self.dataset_model_lock:
            self.dataset_model_list[dataset] = model_list

    def get_dataset_model_list(self, dataset):
        with self.dataset_model_lock:
            if dataset in self.dataset_model_list:
                return self.dataset_model_list[dataset]
            return {}

    def get_dataset_model_info(self, dataset, model_name):
        model_dictionary = self.get_dataset_model_list(dataset)

        if model_name in model_dictionary:
            return model_dictionary[model_name]

        return None

    def set_dataset_data(self, dataset, dataset_data):
        with self.dataset_prompt_generation_data_lock:
            self.dataset_prompt_generation_data_dictionary[dataset] = dataset_data

    def get_dataset_prompt_generation_policy(self, dataset):
        with self.dataset_prompt_generation_data_lock:
            if dataset in self.dataset_prompt_generation_data_dictionary:
                if 'generation_policy' in self.dataset_prompt_generation_data_dictionary[dataset]:
                    return self.dataset_prompt_generation_data_dictionary[dataset]['generation_policy']
            return DEFAULT_PROMPT_GENERATION_POLICY

    def get_dataset_top_k(self, dataset):
        with self.dataset_prompt_generation_data_lock:
            if dataset in self.dataset_prompt_generation_data_dictionary:
                if 'top-k' in self.dataset_prompt_generation_data_dictionary[dataset]:
                    return self.dataset_prompt_generation_data_dictionary[dataset]['top_k']
            return DEFAULT_TOP_K_VALUE

    def get_dataset_rate(self, dataset):
        with self.dataset_prompt_generation_data_lock:
            if dataset in self.dataset_prompt_generation_data_dictionary:
                if 'dataset_rate' in self.dataset_prompt_generation_data_dictionary[dataset]:
                    return self.dataset_prompt_generation_data_dictionary[dataset]['dataset_rate']
            return DEFAULT_DATASET_RATE

    def get_dataset_hourly_limit(self, dataset):
        with self.dataset_prompt_generation_data_lock:
            if dataset in self.dataset_prompt_generation_data_dictionary:
                if 'hourly_limit' in self.dataset_prompt_generation_data_dictionary[dataset]:
                    hourly_limit = int(self.dataset_prompt_generation_data_dictionary[dataset]['hourly_limit'])
                    if hourly_limit <= 0:
                        return DEFAULT_HOURLY_LIMIT
                    else:
                        return hourly_limit
            return DEFAULT_HOURLY_LIMIT

    def get_dataset_relevance_model(self, dataset):
        with self.dataset_prompt_generation_data_lock:
            if dataset in self.dataset_prompt_generation_data_dictionary:
                if 'relevance_model' in self.dataset_prompt_generation_data_dictionary[dataset]:
                    return self.dataset_prompt_generation_data_dictionary[dataset]['relevance_model']
            return ""

    def get_dataset_ranking_model(self, dataset):
        with self.dataset_prompt_generation_data_lock:
            if dataset in self.dataset_prompt_generation_data_dictionary:
                if 'ranking_model' in self.dataset_prompt_generation_data_dictionary[dataset]:
                    return self.dataset_prompt_generation_data_dictionary[dataset]['ranking_model']
            return ""

    def get_dataset_scoring_model(self, dataset):
        model_name = self.get_dataset_ranking_model(dataset)

        print('model name : ', model_name)
        model_info = self.get_dataset_model_info(dataset, model_name)

        print('model_info :', model_info)

        if model_info is None:
            return

        if not isinstance(model_info, dict):
            return

        model_type = model_info['model_type']

        model = None

        if model_type == 'image-pair-ranking-efficient-net':
            model = self.get_efficient_net_model(dataset)
        elif model_type == 'ab_ranking_efficient_net':
            model = self.get_efficient_net_model(dataset)
        elif model_type == 'ab_ranking_linear':
            model = self.get_linear_model(dataset)
        elif model_type == 'image-pair-ranking-linear':
            model = self.get_linear_model(dataset)
        elif model_type == 'ab_ranking_elm_v1':
            model = self.get_elm_v1_model(dataset)
        elif model_type == 'image-pair-ranking-elm-v1':
            model = self.get_elm_v1_model(dataset)

        return model

    def set_total_rate(self, total_rate):
        self.total_rate = total_rate

    def set_dataset_job_queue_size(self, dataset, job_queue_size):
        with self.dataset_job_queue_size_lock:
            self.dataset_job_queue_size[dataset] = job_queue_size

    def append_dataset_job_queue_size(self, dataset, value):
        with self.dataset_job_queue_size_lock:
            self.dataset_job_queue_size[dataset] += value

    def set_dataset_job_queue_target(self, dataset, job_queue_target):
        with self.dataset_job_queue_size_lock:
            self.dataset_job_queue_target[dataset] = job_queue_target

    def get_dataset_job_queue_size(self, dataset):
        with self.dataset_job_queue_size_lock:
            if dataset in self.dataset_job_queue_size:
                return self.dataset_job_queue_size[dataset]

            return None

    def get_dataset_job_queue_target(self, dataset):
        with self.dataset_job_queue_size_lock:
            if dataset in self.dataset_job_queue_target:
                return self.dataset_job_queue_target[dataset]

            return None

    def add_dataset_mask(self, dataset, init_image_path, mask_path):
        if dataset not in self.dataset_masks:
            self.dataset_masks[dataset] = []

        self.dataset_masks[dataset].append({
            'init_image' : init_image_path,
            'mask' : mask_path
        })

    def get_random_dataset_mask(self, dataset):
        if dataset in self.dataset_masks:
            mask_list = self.dataset_masks[dataset]
        else:
            mask_list = None

        if mask_list is None:
            return None
        random_index = random.randint(0, len(mask_list) - 1)
        return mask_list[random_index]

