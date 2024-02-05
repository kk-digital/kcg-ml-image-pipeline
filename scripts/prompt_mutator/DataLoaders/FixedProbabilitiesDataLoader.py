import io
import sys
import numpy as np

import pandas as pd
from data_loader.phrase_embedding_loader import PhraseEmbeddingLoader

base_directory = "./"
sys.path.insert(0, base_directory)
from utility.minio import cmd
from utility.boltzman.boltzman_phrase_scores_loader import BoltzmanPhraseScoresLoader

MAX_LENGTH=77
DATA_PATH="environmental/data/prompt-generator/substitution/input/"

class FixedProbabilitiesDataLoader:
    def __init__(self,
                 dataset_name,
                 phrase_csv,
                 text_encoder=None,
                 minio_ip_addr=None,
                 minio_access_key=None,
                 minio_secret_key=None):
        self.dataset_name = dataset_name
        self.csv_phrase= phrase_csv
        self.text_encoder= text_encoder
        self.phrase_list=[]
        self.token_length_dict={}
        self.embedding_dict={}

        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.minio_ip_addr= minio_ip_addr
        self.minio_client = cmd.get_minio_client(minio_access_key=self.minio_access_key,
                                                 minio_secret_key=self.minio_secret_key,
                                                 minio_ip_addr=self.minio_ip_addr)
        
    def load_phrases(self):
        # get list of phrases and their token lengths
        phrase_df=pd.read_csv(self.csv_phrase).sort_values(by="index")
        self.phrase_list=phrase_df['phrase str'].tolist()

        # store list of token lengths for each phrase
        phrase_token_lengths=self.load_phrase_token_lengths()

        # get phrase embeddings    
        phrase_embeddings= self.load_phrase_embeddings()

        # create dictionarries for embeddings and token lengths
        for index, phrase in enumerate(self.phrase_list):
            self.token_length_dict[phrase]= phrase_token_lengths[index]
            self.embedding_dict[phrase]= phrase_embeddings[index]
    
    # get civitai phrase embeddings from minIO
    def load_phrase_embeddings(self):
        filename='phrase_embeddings.npz'

        # Get the file data from MinIO
        minio_path = DATA_PATH + filename
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
        filename='token_lengths.csv'

        # Get the file data from MinIO
        minio_path = DATA_PATH + filename
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

    # get token length of a phrase
    def get_token_length(self, phrase: str):
        if self.token_length_dict.get(phrase) is not None:
            return self.token_length_dict[phrase]
        else:
            return None
    
    # function to compute the token length
    def compute_token_length(self, phrase:str):
        # Tokenize the phrase
        batch_encoding = self.text_encoder.tokenizer(phrase, truncation=False, max_length=MAX_LENGTH, return_length=True,
                                        return_overflowing_tokens=False, return_tensors="pt")
        
        input_ids = batch_encoding['input_ids']
        num_tokens = input_ids.numel()

        return num_tokens
    
    # get phrase embedding
    def get_phrase_embedding(self, phrase:str):
        # return the embedding of the specified phrase
        if self.embedding_dict.get(phrase) is not None:
            return self.embedding_dict[phrase]
        else:
            return None 