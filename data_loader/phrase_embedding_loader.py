import os
import sys
import json
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pytz import timezone
import io
import csv
import tiktoken
import numpy as np

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.minio import cmd
from data_loader.utils import get_object, get_phrases_from_prompt, get_datasets
from data_loader.generated_image_data import GeneratedImageData
from stable_diffusion.model.clip_text_embedder import CLIPTextEmbedder
from utility.clip.clip_text_embedder import tensor_attention_pooling

class PhraseEmbeddingLoader:
    def __init__(self,
                 dataset_name,
                 minio_ip_addr=None,
                 minio_access_key=None,
                 minio_secret_key=None,):
        self.dataset_name = dataset_name

        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.minio_client = cmd.get_minio_client(minio_access_key=self.minio_access_key,
                                                 minio_secret_key=self.minio_secret_key,
                                                 minio_ip_addr=minio_ip_addr)

        self.phrase_index_dict = {}
        self.phrase_arr = np.array([])
        self.phrase_embedding_arr = np.array([])

        self.text_embedder = None


    def load_phrase_embeddings(self):
        count = 0
        while True:
            # try current date
            date_now = datetime.now(tz=timezone("Asia/Hong_Kong")) - timedelta(days=count)
            date_now = date_now.strftime('%Y-%m-%d')
            full_path = os.path.join(self.dataset_name, "output/phrase-embeddings", "{}-phrase-embeddings-average-pooled.npz".format(date_now))
            print(f"Getting phrase embeddings: ", full_path)

            # check if exists
            if  cmd.is_object_exists(self.minio_client, "datasets", full_path):
                data = get_object(self.minio_client, full_path)
                data_buffer = io.BytesIO(data)
                npz_data = np.load(data_buffer, allow_pickle=True)
                self.phrase_arr = npz_data["phrase_arr"]
                self.phrase_embedding_arr = npz_data["embeddings_arr"]
                print("phrase arr shape=", self.phrase_arr.shape)
                print("embedding arr shape=", self.phrase_embedding_arr.shape)

                # add keys to phrase index dict
                for i in range(len(self.phrase_arr)):
                    phrase = self.phrase_arr[i]
                    self.phrase_index_dict[phrase] = i

                print("Phrase embeddings data loaded...")
                return

            if count == 30:
                raise Exception("No phrase embeddings data in minio server...")

            count += 1

    def load_dataset_phrases(self):
        start_time = time.time()
        print("Loading phrase embedding data...")

        dataset_list = get_datasets(self.minio_client)
        if self.dataset_name not in dataset_list:
            raise Exception("Dataset is not in minio server")


        # check if there is a phrase embedding data
        phrase_embeddings_prefix = os.path.join(self.dataset_name, "output/phrase-embeddings")
        all_objects = cmd.get_list_of_objects_with_prefix(self.minio_client, 'datasets', phrase_embeddings_prefix)
        if len(all_objects) != 0:
            # then load phrase embeddings
            self.load_phrase_embeddings()
            print("Data loaded...")
            print("Time elapsed: {0}s".format(format(time.time() - start_time, ".2f")))
        else:
            print("No phrase embeddings data for the dataset in minio server")
            return

    def upload_phrases_embedding_npz(self):
        date_now = datetime.now(tz=timezone("Asia/Hong_Kong")).strftime('%Y-%m-%d')
        filename = "{}-phrase-embeddings-average-pooled.npz".format(date_now)
        phrase_embedding_path = os.path.join(self.dataset_name, "output/phrase-embeddings", filename)

        compressed_array = io.BytesIO()
        np.savez_compressed(compressed_array,
                            phrase_arr=self.phrase_arr,
                            embeddings_arr=self.phrase_embedding_arr)

        compressed_array.seek(0)

        cmd.upload_data(self.minio_client, 'datasets', phrase_embedding_path, compressed_array)

    def update_dataset_phrases(self, phrases_arr):
        # load text embedder
        text_embedder = CLIPTextEmbedder()
        text_embedder.load_submodels()
        self.text_embedder = text_embedder

        print("Updating phrase embeddings data...")
        len_phrases_arr = len(phrases_arr)
        count_added = 0
        max_batch_size = 64
        batch_phrase = []
        i = 0
        for phrase in tqdm(phrases_arr):
            if phrase not in self.phrase_index_dict:
                batch_phrase.append(phrase)

                if len(batch_phrase) == max_batch_size or i >= (len_phrases_arr - max_batch_size):
                    # get embedding of phrase
                    phrase_embeddings, _, phrase_attention_masks = text_embedder.forward_return_all(batch_phrase)
                    phrase_average_pooled = tensor_attention_pooling(phrase_embeddings, phrase_attention_masks)
                    phrase_average_pooled_results = phrase_average_pooled.cpu().detach().numpy()

                    for i in range(len(batch_phrase)):
                            phrase = batch_phrase[i]

                            curr_len = len(self.phrase_arr)
                            self.phrase_index_dict[phrase] = curr_len
                            self.phrase_arr = np.append(self.phrase_arr, phrase)

                    if len(self.phrase_embedding_arr) == 0:
                        self.phrase_embedding_arr = phrase_average_pooled_results

                    else:
                        if self.phrase_embedding_arr.shape == (768,):
                            self.phrase_embedding_arr = np.expand_dims(self.phrase_embedding_arr, axis=0)
                        self.phrase_embedding_arr = np.append(self.phrase_embedding_arr, phrase_average_pooled_results, axis=0)

                    count_added += len(batch_phrase)
                    # update every 30k data are newly added
                    if count_added % 30000 == 0:
                        self.upload_phrases_embedding_npz()

                    batch_phrase = []
            i += 1

        # save after update
        self.upload_phrases_embedding_npz()

    def get_embedding(self, phrase):
        if phrase in self.phrase_index_dict:
            index = self.phrase_index_dict[phrase]
        else:
            # calculate embedding
            phrase_embeddings, _, phrase_attention_masks = self.text_embedder.forward_return_all(phrase)
            phrase_average_pooled = tensor_attention_pooling(phrase_embeddings, phrase_attention_masks)
            phrase_average_pooled_results = phrase_average_pooled.cpu().detach().numpy()

            curr_len = len(self.phrase_arr)
            self.phrase_index_dict[phrase] = curr_len
            self.phrase_arr = np.append(self.phrase_arr, phrase)

            self.phrase_embedding_arr = np.append(self.phrase_embedding_arr, phrase_average_pooled_results, axis=0)

            return phrase_average_pooled_results

        return self.phrase_embedding_arr[index]

    def get_embeddings(self, phrases):
        phrase_embeddings = []
        for phrase in phrases:
            embedding = self.get_embedding(phrase)
            phrase_embeddings.append(embedding)

        return phrase_embeddings








