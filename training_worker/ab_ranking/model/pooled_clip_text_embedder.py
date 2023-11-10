import numpy as np
import msgpack
import torch
import gc

import hashlib

from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel

from stable_diffusion.utils_backend import get_device


class PooledClipTextEembedder:
    '''
    manually load and unload cliptextmodel
    '''
    def __init__(self, model_type='openai/clip-vit-large-patch14', device='cpu', verbose=True):
        
        self.model_type = model_type
        self.device = device
        self.verbose = verbose

    def load_model(self, local_files_only=True):

        if self.verbose: print("Loading CLIPTextModel.")
            
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_type, local_files_only=local_files_only)
        # self.model = CLIPTextModel.from_pretrained(self.model_type, local_files_only=local_files_only).to(self.device).eval()
        self.model = CLIPModel.from_pretrained(self.model_type, local_files_only=local_files_only).text_model.to(self.device).eval()

        if self.verbose: print("CLIPTextModel succesfully.")

    def unload_model(self):
        
        for module_name in self.registered_modules.keys():
            setattr(self, module_name, None)
            
        gc.collect()
        torch.cuda.empty_cache()
        if self.verbose: print("CLIPTextModel unloaded.")
        
    def inference(self, prompts: str, pooling_strategy: str = 'AVERAGE_POOLING'):
        
        encoding = self.tokenizer(
            prompts,
            truncation=True, max_length=77, return_length=True,
            return_overflowing_tokens=False, padding="max_length", return_tensors="pt"
        )
        
        with torch.no_grad():

            clip_opt = self.model(input_ids=encoding["input_ids"].to(self.device))

            last_hidden_state = clip_opt.last_hidden_state[0]
            
            if pooling_strategy == 'AVERAGE_POOLING':

                embedding = last_hidden_state.mean(dim=0)

            elif pooling_strategy == 'MAX_POOLING':

                embedding = last_hidden_state.max(dim=0).values

            elif pooling_strategy == 'MAX_ABS_POOLING':

                last_hidden_state_abs = torch.abs(last_hidden_state)
                last_hidden_state_max_indices = torch.max(last_hidden_state_abs, dim=0).indices
                embedding = last_hidden_state.gather(0, last_hidden_state_max_indices.unsqueeze(0)).squeeze(0)

            elif pooling_strategy == 'ATTENTION_POOLING':
                
                attention_mask = encoding.attention_mask[0].unsqueeze(0).to(self.device)
                embedding = (last_hidden_state * attention_mask).sum(dim=0) / attention_mask.sum(dim=0)

            elif pooling_strategy == 'CLIP_POOLING':

                embedding = clip_opt.pooler_output[0]
                
            else:
                raise(f'ERROR! Unknown pooling strateg: {pooling_strategy}')
            
        return PooledClipTextEembedderOutput(
            model_type=self.model_type,
            prompts=prompts,
            embedding=embedding.detach().cpu().numpy(),
            pooling_strategy=pooling_strategy
        )



class PooledClipTextEembedderOutput:
    
    def __init__(self, model_type: str, prompts: str, embedding: np.ndarray, pooling_strategy: str):
        
        self.model_type = model_type
        self.prompts = prompts
        self.embedding = embedding.astype(np.float32)
        self.pooling_strategy = pooling_strategy
        
        self.embedding_hash = hashlib.md5(self.embedding.tobytes()).hexdigest()
        
    def save(self, file_path: str):
        
        decoded_data = msgpack.packb(dict(
            model_type=self.model_type,
            prompts=self.prompts,
            embedding={'__ndarray__': self.embedding.tolist()},
            pooling_strategy=self.pooling_strategy,
            embedding_hash=self.embedding_hash
        ), use_bin_type=True)
        
        with open(file_path, 'wb') as f:
            f.write(decoded_data)
    
    @classmethod
    def load(cls, file_path: str):
        
        with open(file_path, 'rb') as f:
            data = f.read()
            
        decoded_data = msgpack.unpackb(data)
        
        output = PooledClipTextEembedderOutput(
            model_type=decoded_data['model_type'],
            prompts=decoded_data['prompts'],
            embedding=np.array(decoded_data['embedding']['__ndarray__']),
            pooling_strategy=decoded_data['pooling_strategy']
        )

        assert decoded_data['embedding_hash'] == output.embedding_hash, f'ERROR! hash miss match in {file_path}: stored {decoded_data["embedding_hash"]} != calculated {output.embedding_hash}'

        return output

