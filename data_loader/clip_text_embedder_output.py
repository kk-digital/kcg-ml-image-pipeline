import numpy as np
import msgpack
import hashlib

class CLIPTextEmbedderOutput:
    
    def __init__(self, model_name: str, prompts: str, embedding: np.ndarray, pooler_output: np.ndarray, attention_mask: np.ndarray):
        
        self.model_name = model_name
        self.prompts = prompts
        self.embedding = embedding
        self.pooler_output = pooler_output
        self.attention_mask = attention_mask.astype(np.uint8)
        
        self.token_length = self.attention_mask.sum()
        
    def get_msgpack_string(self, use_single_float=True):

        if use_single_float:
            embedding = self.embedding.astype(np.float32)
            pooler_output = self.pooler_output.astype(np.float32)
        else:
            embedding = self.embedding.astype(np.float64)
            pooler_output = self.pooler_output.astype(np.float64)
        
        msgpack_string = msgpack.packb(dict(
            model_name=self.model_name,
            prompts=self.prompts,
            embedding={'__ndarray__': embedding.tolist()},
            pooler_output={'__ndarray__': pooler_output.tolist()},
            attention_mask={'__ndarray__': self.attention_mask.tolist()}
        ), use_bin_type=True)
        
        return msgpack_string
    
    @classmethod
    def from_msgpack_string(cls, msgpack_string: str):
            
        decoded_data = msgpack.unpackb(msgpack_string)
        
        output = cls(
            model_name=decoded_data['model_name'],
            prompts=decoded_data['prompts'],
            embedding=np.array(decoded_data['embedding']['__ndarray__']),
            pooler_output=np.array(decoded_data['pooler_output']['__ndarray__']),
            attention_mask=np.array(decoded_data['attention_mask']['__ndarray__'])
        )

        # assert decoded_data['embedding_hash'] == output.embedding_hash, f'ERROR! hash miss match in {file_path}: stored {decoded_data["embedding_hash"]} != calculated {output.embedding_hash}'

        return output


class PooledCLIPTextEmbedderOutput:
    
    def __init__(self, model_name: str, prompts: str, pooled_embedding: np.ndarray, pooling_strategy: str):
        
        self.model_name = model_name
        self.prompts = prompts
        self.pooled_embedding = pooled_embedding
        self.pooling_strategy = pooling_strategy
        
        # self.embedding_hash = hashlib.sha256(self.embedding.tobytes()).hexdigest()
        
    def get_msgpack_string(self, use_single_float=True):

        if use_single_float:
            pooled_embedding = self.pooled_embedding.astype(np.float32)
        else:
            pooled_embedding = self.pooled_embedding.astype(np.float64)
        
        msgpack_string = msgpack.packb(dict(
            model_name=self.model_name,
            prompts=self.prompts,
            pooled_embedding={'__ndarray__': pooled_embedding.tolist()},
            pooling_strategy=self.pooling_strategy,
            # embedding_hash=self.embedding_hash
        ), use_bin_type=True)
        
        return msgpack_string
    
    @classmethod
    def from_msgpack_string(cls, msgpack_string: str):
            
        decoded_data = msgpack.unpackb(msgpack_string)
        
        output = cls(
            model_name=decoded_data['model_name'],
            prompts=decoded_data['prompts'],
            pooled_embedding=np.array(decoded_data['pooled_embedding']['__ndarray__']),
            pooling_strategy=decoded_data['pooling_strategy']
        )

        # assert decoded_data['embedding_hash'] == output.embedding_hash, f'ERROR! hash miss match in {file_path}: stored {decoded_data["embedding_hash"]} != calculated {output.embedding_hash}'

        return output

# deal with pooling for batch / instance

def pooling_wrapper(x, pooling_function, pooling_strategy):

    if isinstance(x, CLIPTextEmbedderOutput):
        pooled_embedding = pooling_function(x.embedding)
        return PooledCLIPTextEmbedderOutput(
            model_name=x.model_name, 
            prompts=x.prompts, 
            pooled_embedding=pooled_embedding, 
            pooling_strategy=pooling_strategy
        )
    
    embedding = np.stack([i.embedding for i in x], axis=0)
    pooled_embedding = pooling_function(embedding)

    input_type = type(x)
    results = input_type()
    for i, p in zip(x, pooled_embedding):
        results.append(PooledCLIPTextEmbedderOutput(
            model_name=i.model_name, 
            prompts=i.prompts, 
            pooled_embedding=p, 
            pooling_strategy=pooling_strategy
        ))

    return results

# pooling functions

def average_pooling(x):
    def f(embedding):
        return embedding.mean(axis=-2)
    return pooling_wrapper(x, f, 'AVERAGE_POOLING')

def max_pooling(x):
    def f(embedding):
        return embedding.max(axis=-2)
    return pooling_wrapper(x, f, 'MAX_POOLING')

def max_abs_pooling(x):
    def f(embedding):
        embedding_abs = np.abs(embedding)
        embedding_max_indices = np.argmax(embedding_abs, axis=-2)
        return np.take_along_axis(embedding, embedding_max_indices[..., None, :], axis=-2)[..., 0, :]
    return pooling_wrapper(x, f, 'MAX_ABS_POOLING')

def attention_pooling(x):

    def f(embedding, attention_mask):
        attention_mask = attention_mask[..., None]
        pooled_embedding = (embedding * attention_mask).sum(axis=-2) / attention_mask.sum(axis=-2)
        return pooled_embedding
    
    if isinstance(x, CLIPTextEmbedderOutput):
        return PooledCLIPTextEmbedderOutput(
            model_name=x.model_name, 
            prompts=x.prompts, 
            pooled_embedding=f(x.embedding, x.attention_mask), 
            pooling_strategy='ATTENTION_POOLING'
        )
    
    embedding = np.stack([i.embedding for i in x], axis=0)
    attention_mask = np.stack([i.attention_mask for i in x], axis=0)
    pooled_embedding = f(embedding, attention_mask)

    input_type = type(x)
    results = input_type()
    for i, p in zip(x, pooled_embedding):
        results.append(PooledCLIPTextEmbedderOutput(
            model_name=i.model_name, 
            prompts=i.prompts, 
            pooled_embedding=p, 
            pooling_strategy='ATTENTION_POOLING'
        ))
        
    return results

def clip_pooling(x):

    def f(o):
        return PooledCLIPTextEmbedderOutput(
            model_name=o.model_name, 
            prompts=o.prompts, 
            pooled_embedding=o.pooler_output, 
            pooling_strategy='CLIP_POOLING'
        )
    
    if isinstance(x, CLIPTextEmbedderOutput):
        return f(x)
    
    input_type = type(x)
    return input_type(map(f, x))
