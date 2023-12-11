import numpy as np
import pandas as pd
import torch

from tqdm.auto import tqdm


class PromptDataset(torch.utils.data.Dataset):
    
    def __init__(self, prompt_table: pd.DataFrame, tokenizer, min_prompt_length=3, max_prompt_length=150, p_shuffle=0., max_shuffle=1, p_cut=0., overflow_method='drop', p_length=0.):
                 
        self.tokenizer = tokenizer 
                 
        self.constants = {
            'POSITIVE': self.tokenize('<positive prompt>:'),
            'SHORT_POSITIVE': self.tokenize('<short positive prompt>:'),
            'MEDIUM_POSITIVE': self.tokenize('<medium positive prompt>:'),
            'LONG_POSITIVE': self.tokenize('<long positive prompt>:'),
            'FULL_POSITIVE': self.tokenize('<full positive prompt>:'),
            'NEGATIVE': self.tokenize('<negative prompt>:'),
            'SHORT_NEGATIVE': self.tokenize('<short negative prompt>:'),
            'MEDIUM_NEGATIVE': self.tokenize('<medium negative prompt>:'),
            'LONG_NEGATIVE': self.tokenize('<long negative prompt>:'),
            'FULL_NEGATIVE': self.tokenize('<full negative prompt>:'),
            'BOS': torch.tensor([self.tokenizer.bos_token_id]),
            'EOS': torch.tensor([self.tokenizer.eos_token_id]),
            'PAD': torch.tensor([self.tokenizer.pad_token_id]),
            'END_OF_PROMPT': torch.tensor([self.tokenizer.vocab[';']]),
            'START_OF_CONDITION': self.tokenize('{{'),
            'END_OF_CONDITION': self.tokenize('}}'),
            'SPLIT': torch.tensor([self.tokenizer.vocab[',']]),
        }
        
        self.min_prompt_length = min_prompt_length
        self.max_prompt_length = max_prompt_length
        self.overflow_method = overflow_method
        
        self.p_shuffle = p_shuffle
        self.max_shuffle = max_shuffle
        self.p_cut = p_cut
        self.p_length = p_length
        
        self.load(prompt_table)
                
    def load(self, prompt_table):
        
        self.samples = list()
        
        if 'negative_prompt' in prompt_table:
            negative_prompts = prompt_table['negative_prompt']
            negative_hashs = prompt_table['negative_hash']
        else:
            negative_prompts = [''] * prompt_table.shape[0]
            negative_hashs = [''] * prompt_table.shape[0]
        
        positive_hash_s, negative_hash_s = set(), set()
        
        for positive_prompt, positive_hash, negative_prompt, negative_hash in tqdm(zip(prompt_table['positive_prompt'], prompt_table['positive_hash'], negative_prompts, negative_hashs), total=prompt_table.shape[0]):
            
            positive_prompt = self.preprocess_prompt(positive_prompt)
            
            if len(positive_prompt) > 0 and positive_hash not in positive_hash_s:
                
                positive_hash_s.add(positive_hash)
            
                positive_tokens = self.tokenize(positive_prompt)
                positive_tokens = self.format_tokens(positive_tokens)

                if positive_tokens is not None and len(positive_tokens) > self.min_prompt_length:
                    if len(positive_tokens) < self.max_prompt_length:
                        self.samples.append((positive_tokens, True))
                    elif self.overflow_method != 'drop':
                        self.samples.append((positive_tokens[:self.max_prompt_length-1], True))
                        if self.overflow_method == 'split':
                            for splited_tokens in self.split_long_prompt(positive_tokens):
                                self.samples.append((splited_tokens, True))
            
            negative_prompt = self.preprocess_prompt(negative_prompt)
            
            if len(negative_prompt) > 0 and negative_hash not in negative_hash_s:
                
                negative_hash_s.add(negative_hash)
            
                negative_tokens = self.tokenize(negative_prompt)
                negative_tokens = self.format_tokens(negative_tokens)

                if negative_tokens is not None and len(negative_tokens) > self.min_prompt_length:
                    if len(negative_tokens) < self.max_prompt_length:
                        self.samples.append((negative_tokens, False))
                    elif self.overflow_method == 'drop':
                        self.samples.append((negative_tokens[:self.max_prompt_length-1], False))
                        if self.overflow_method == 'split':
                            for splited_tokens in self.split_long_prompt(negative_tokens):
                                self.samples.append((splited_tokens, False))
    
    def __len__(self):
        return len(self.samples)
    
    def preprocess_prompt(self, prompt):
        if type(prompt) != str:
            return ''
        return prompt
    
    def format_tokens(self, tokens):
        if len(tokens) == 0:
            return None
        return tokens
    
    def shuffle_tags(self, tokens):
        
        if self.p_shuffle <= 0 or np.random.rand() > self.p_shuffle:
            return tokens
    
        split_id = self.constants['SPLIT'][0]
        
        n_shuffle = np.random.randint(self.max_shuffle) + 1
        
        for _ in range(n_shuffle):
            
            offset = np.random.randint(len(tokens))
            
            pos = -1
            for i in range(offset + 1, len(tokens) - 1):
                if tokens[i] == split_id:
                    pos = i
                    break
            
            if pos > 0:
                tokens = torch.concat([tokens[i+1:], self.constants['SPLIT'], tokens[:i]], dim=0)
                
        return tokens

    def cut_tags(self, tokens):
        
        length = len(tokens)

        if length < 50:
            return tokens

        if self.p_cut <= 0 or np.random.rand() > self.p_cut:
            return tokens
    
        split_id = self.constants['SPLIT'][0]

        split_indices = [-1]
        for i in range(length):
            if tokens[i] == split_id:
                split_indices.append(i)
        split_indices.append(length)

        min_lengths = [20]
        if length > 75:
            min_lengths.append(40)
        if length > 150:
            min_lengths.append(60)

        min_length = min_lengths[np.random.randint(len(min_lengths))]
        max_length = int(min_length * 1.25)
            
        for i, start in enumerate(split_indices[:-1]):
            for end in split_indices[i+1:]:
                if (end - start - 1) > min_length and (end - start - 1) < max_length:
                    return tokens[start+1:end]
        
        return tokens
    
    def split_long_prompt(self, tokens):
        
        length = len(tokens)
    
        split_id = self.constants['SPLIT'][0]

        split_indices = [-1]
        for i in range(length):
            if tokens[i] == split_id:
                split_indices.append(i)
        split_indices.append(length)
        
        last_start = None
        for i, start in enumerate(split_indices[:-1]):
            last_end = None
            for end in split_indices[i+1:]:
                if (end - start - 1) >= self.max_prompt_length:
                    break
                last_end = end
            if last_end is not None:
                if last_start is None or (start - last_start - 1) > 50:
                    if (last_end - start - 1) >= 50:
                        last_start = start
                        yield tokens[last_start+1:last_end]
        
    
    def add_prefix(self, tokens, positive=True):
        
        if np.random.rand() > self.p_length:
            return torch.concat([self.constants['POSITIVE' if positive else 'NEGATIVE'], tokens])
        
        length = len(tokens)
        
        if length < 25:
            return torch.concat([self.constants['SHORT_POSITIVE' if positive else 'SHORT_NEGATIVE'], tokens])
        if length < 50:
            return torch.concat([self.constants['MEDIUM_POSITIVE' if positive else 'MEDIUM_NEGATIVE'], tokens])
        if length < 75:
            return torch.concat([self.constants['LONG_POSITIVE' if positive else 'LONG_NEGATIVE'], tokens])
        
        return torch.concat([self.constants['FULL_POSITIVE' if positive else 'FULL_NEGATIVE'], tokens])
        
    def add_suffix(self, tokens):
        
        return torch.concat([tokens, self.constants['END_OF_PROMPT']])
        
    def add_start_end(self, tokens):
        
        return torch.concat([self.constants['BOS'], tokens, self.constants['EOS']])
                 
    def tokenize(self, text):
            encoding = self.tokenizer(
                text,
                truncation=False, max_length=None, return_length=True,
                return_overflowing_tokens=False, padding=False, return_tensors="pt"
            )
            return encoding.input_ids[0]
                 
    def __getitem__(self, index):
        
        prompt, is_positive = self.samples[index]

        prompt = self.shuffle_tags(prompt)
        
        prompt = self.cut_tags(prompt)
        
        prompt = self.add_prefix(prompt, positive=is_positive)
        
        prompt = self.add_suffix(prompt)
        
        prompt = self.add_start_end(prompt)
        
        return {'input_ids': prompt}
    
    
class PairedPromptDataset(PromptDataset):
    
    def load(self, prompt_table):
   
        self.samples = list()
    
        pair_hash_s = set()
        
        for positive_prompt, positive_hash, negative_prompt, negative_hash in tqdm(prompt_table[['positive_prompt', 'positive_hash', 'negative_prompt', 'negative_hash']].itertuples(index=False, name=None), total=prompt_table.shape[0]):
            
            positive_prompt = self.preprocess_prompt(positive_prompt)
            positive_tokens = self.tokenize(positive_prompt)
            positive_tokens = self.format_tokens(positive_tokens)
            
            if positive_tokens is None or len(positive_tokens) <= self.min_prompt_length:
                continue
            
            negative_prompt = self.preprocess_prompt(negative_prompt)
            negative_tokens = self.tokenize(negative_prompt)
            negative_tokens = self.format_tokens(negative_tokens)
            
            if negative_tokens is None or len(negative_tokens) <= self.min_prompt_length:
                continue
                
            pair_hash = positive_hash + negative_hash
            if pair_hash in pair_hash_s:
                continue
            pair_hash_s.add(pair_hash)
            
            if len(positive_tokens) + len(negative_tokens) < self.max_prompt_length * 2:
                self.samples.append((positive_tokens, negative_tokens))
                continue
            
            positive_tokens_s = list()
            if len(positive_tokens) < self.max_prompt_length:
                positive_tokens_s.append(positive_tokens)
            else:
                if self.overflow_method == 'drop':
                    continue
                positive_tokens_s.append(positive_tokens[:self.max_prompt_length-1])
                if self.overflow_method == 'split':
                    positive_tokens_s.extend(list(self.split_long_prompt(positive_tokens)))
                    
            negative_tokens_s = list()
            if len(negative_tokens) < self.max_prompt_length:
                negative_tokens_s.append(negative_tokens)
            else:
                if self.overflow_method == 'drop':
                    continue
                negative_tokens_s.append(negative_tokens[:self.max_prompt_length-1])
                if self.overflow_method == 'split':
                    negative_tokens_s.extend(list(self.split_long_prompt(negative_tokens)))
                    
            for positive_tokens in positive_tokens_s:
                for negative_tokens in negative_tokens_s:
                    self.samples.append((positive_tokens, negative_tokens))
        
    def add_condition_start_end(self, tokens):
        
        return torch.concat([self.constants['START_OF_CONDITION'], tokens, self.constants['END_OF_CONDITION']])
                    
    def __getitem__(self, index):
        
        positive_prompt, negative_prompt = self.samples[index]


        positive_prompt = self.shuffle_tags(positive_prompt)
        
        positive_prompt = self.cut_tags(positive_prompt)


        negative_prompt = self.shuffle_tags(negative_prompt)
        
        negative_prompt = self.cut_tags(negative_prompt)
        
        
        if np.random.rand() > 0.5:
            
            # pos2neg
            
            prefix = self.add_condition_start_end(positive_prompt)
        
            negative_prompt = self.add_prefix(negative_prompt, positive=False)
            
            negative_prompt = self.add_suffix(negative_prompt)
            
            prompt = torch.concat([prefix, negative_prompt])
            
        else:
            
            # neg2pos
            
            prefix = self.add_condition_start_end(negative_prompt)
        
            positive_prompt = self.add_prefix(positive_prompt, positive=True)
            
            positive_prompt = self.add_suffix(positive_prompt)
            
            prompt = torch.concat([prefix, positive_prompt])

        prompt = self.add_start_end(prompt)
        
        return {'input_ids': prompt}
    
    
class MultipleDataset(torch.utils.data.Dataset):
    
    def __init__(self, datasets, probabilities = None):
        
        self.datasets = datasets
        
        if probabilities is not None:
            probabilities = np.array(list(probabilities))
            
            assert len(datasets) == len(probabilities)
            
            probabilities = probabilities / probabilities.sum()
        
        self.probabilities = probabilities
        
    def __len__(self):
        return max(map(len, self.datasets))
    
    def __getitem__(self, index):
        
        if self.probabilities is None:
            dataset_index = np.random.randint(len(self.datasets))
        else:
            dataset_index = np.random.choice(len(self.datasets), 1, p=self.probabilities)[0]
            
        dataset = self.datasets[dataset_index]
        
        index = index % len(dataset)
        
        return dataset[index]
    
class InferenceDataset(torch.utils.data.Dataset):
    
    def __init__(self, prompts):
    
        self.prompts = prompts
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return self.prompts[index]
    