import sys
from data_loader.phrase_embedding_loader import PhraseEmbeddingLoader

base_directory = "./"
sys.path.insert(0, base_directory)
from utility.minio import cmd
from utility.boltzman.boltzman_phrase_scores_loader import BoltzmanPhraseScoresLoader

MAX_LENGTH=77

class IndependantApproximationDataLoader:
    def __init__(self,
                 dataset_name,
                 phrase_csv,
                 text_encoder=None,
                 minio_ip_addr=None,
                 minio_access_key=None,
                 minio_secret_key=None,):
        self.dataset_name = dataset_name
        self.phrase_csv= phrase_csv
        self.text_encoder= text_encoder
        self.phrase_list=[]
        self.token_length_dict={}
        self.phrase_embedding_loader=None

        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.minio_ip_addr= minio_ip_addr
        self.minio_client = cmd.get_minio_client(minio_access_key=self.minio_access_key,
                                                 minio_secret_key=self.minio_secret_key,
                                                 minio_ip_addr=self.minio_ip_addr)
   

    def load_phrases(self):
        # loading positive phrase scores to use for rejection sampling
        phrase_loader=BoltzmanPhraseScoresLoader(dataset_name=self.dataset_name,
                                                phrase_scores_csv=self.phrase_csv,
                                                minio_client=self.minio_client)
        phrase_loader.load_dataset()
        phrase_score_data= phrase_loader.index_phrase_score_data
        # get the list of phrases
        self.phrase_list=[phrase_score_data[i].phrase for i in range(len(phrase_score_data))]
        # create dictionarries for phrase token lengths and embeddings
        for index, phrase in enumerate(self.phrase_list):
            self.token_length_dict[phrase]= phrase_loader.get_token_size(index)
        
        # get embeddings of the phrases
        self.phrase_embedding_loader= PhraseEmbeddingLoader(dataset_name=self.dataset_name,
                                                        minio_access_key=self.minio_access_key,
                                                        minio_secret_key=self.minio_access_key,
                                                        minio_ip_addr=self.minio_ip_addr)
        self.phrase_embedding_loader.load_dataset_phrases()
        self.phrase_embedding_loader.text_embedder= self.text_encoder

    # get token length of a phrase
    def get_token_length(self, phrase: str):
        if self.token_length_dict.get(phrase) is not None:
            return self.token_length_dict[phrase]
        else:
            return self.compute_token_length(phrase)
    
    def compute_token_length(self, phrase:str):
        # Tokenize the phrase
        batch_encoding = self.text_encoder.tokenizer(phrase, truncation=False, max_length=MAX_LENGTH, return_length=True,
                                        return_overflowing_tokens=False, return_tensors="pt")
        
        input_ids = batch_encoding['input_ids']
        num_tokens = input_ids.numel()

        return num_tokens
    
    # get phrase embedding
    def get_phrase_embedding(self, phrase:str):
        phrase_embed= self.phrase_embedding_loader.get_embedding(phrase)

        if phrase_embed.shape == (1, 768):
            phrase_embed= phrase_embed.reshape(-1)
        
        return phrase_embed

    
