import re
import numpy as np
import hashlib


def is_ascii_string(prompt):
    if type(prompt) != str or len(prompt) == 0:
        return False
    return max(map(ord, prompt)) <= 127

def contains_link(prompt):
    # if url_pattern.search(prompt):
    prompt = prompt.lower()
    
    if '://' in prompt or '.jpg' in prompt or '.png' in prompt or '.com' in prompt or '.org' in prompt or 'http' in prompt:
        return True
    return False

def remove_extra(prompt):
    
    # remove lora, hypernets
    prompt = re.sub(r'<[^<>]+>', '', prompt)
    
    return prompt

def check_complexity(prompt):
    
    # mix
    if re.search(r'\([^\)\|]+(\|[^\)\|]*)+\)', prompt):
        return True
    if re.search(r'\[[^\]\|]+(\|[^\]\|]*)+\]', prompt):
        return True
    if re.search(r'\{[^\}\|]+(\|[^\}\|]*)+\}', prompt):
        return True
    if re.search(r'\<[^\>\|]+(\|[^\>\|]*)+\>', prompt):
        return True
    
    #
    if re.search(r'\([^\):]+(:[^\):]*){2,}\)', prompt):
        return True
    if re.search(r'\[[^\]:]+(:[^\]:]*){2,}\]', prompt):
        return True
    if re.search(r'\{[^\}:]+(:[^\}:]*){2,}\}', prompt):
        return True
    if re.search(r'\<[^\>:]+(:[^\>:]*){2,}\>', prompt):
        return True
    
    return False

def remove_weight(prompt):
    
    
    # # remove neg weighted tags
    
    if re.search(':\s*-[0-9,\.]+', prompt):
        prompt = re.sub('\([^\(]+:\s*-[0-9,\.]+[^)]*\)', ', ', prompt)
        prompt = re.sub('\[[^\[]+:\s*-[0-9,\.]+[^]]*\]', ', ', prompt)
        prompt = re.sub('\{[^\{]+:\s*-[0-9,\.]+[^}]*\}', ', ', prompt)
    
    # remove weight
    prompt = re.sub(':[-\s0-9,\.]*', ', ', prompt)
    
    prompt = re.sub(r'[\(\[\{\<\>\}\]\)]+', '', prompt)
    
    return prompt


def contains_cfg(prompt):
    
    prompt = prompt.lower()
    
    if 'sampling' in prompt or 'cfg' in prompt or 'denoising' in prompt:
        return True
    
    return False

def contains_info(prompt):
    
    if len(re.sub(r'[^\w]+', '', prompt)) < 10:
        return False
    
    return True

def contains_junk(prompt):
    
    if re.search(r'[^,\s;:]{50,}', prompt):
        return True

    if re.search(r'[^,\s;:\-_\./]{25,}', prompt):
        return True
    
    if re.search(r'[\d]{6,}', prompt):
        return True
    
    return False

def check_word_redundancy(prompt):
    
    parts = re.split(r'[\s,_\-\.;:]', prompt)
    parts = [i.strip() for i in parts]
    parts = [i for i in parts if len(i) > 0]
    
    if len(parts) == 0:
        return True
    
    words, counts = np.unique(parts, return_counts=True)
    
    if len(words) == 1:
        return False
    
    max_counts = max(counts)
    lengths = [i*len(j) for i, j in zip(counts, words)]
    
    if max_counts <= 2 or (max(lengths) / sum(lengths)) < 0.67:
        return False

    return True


def format_prompt(prompt):
    
    prompt = prompt.strip()
    
    prompt = re.sub(r'\\n', ', ', prompt)
    prompt = re.sub(r'\\', ' ', prompt)
    # prompt = re.sub(r'\\([\(\)\[\]])', r'\1', prompt)
    # prompt = re.sub(r'\\[\\\s]+', ' ', prompt)
    # prompt = re.sub(r'[/\/]{2,}', ' ', prompt)
    
    while re.search(r'(\([\s,\.:;\|]*\))|(\<[\s,\.:;\|]*\>)|(\[[\s,\.:;\|]*\])|(\{[\s,\.:;\|]*\})', prompt):
        prompt = re.sub(r'(\([\s,\.:;\|]*\))|(\<[\s,\.:;\|]*\>)|(\[[\s,\.:;\|]*\])|(\{[\s,\.:;\|]*\})', '', prompt)
    
    prompt = re.sub(r'([\[\(\{\<])\s', r'\1', prompt)
    prompt = re.sub(r'\s([\]\)\}\>])', r'\1', prompt)
    prompt = re.sub(r'\s+', ' ', prompt)
    prompt = re.sub(r'(\s?[,;])+', r',', prompt)
    
    prompt = re.sub(r'^[\.,;\s]+', '', prompt)
    prompt = re.sub(r'[\.,;\s]+$', '', prompt)
    
    return prompt


def hash_prompt(prompt):
    return hashlib.md5(prompt.encode()).hexdigest()


def remove_redundant_tags(prompt):
    
    tags = list()
    exists = set()
    for tag in prompt.split(','):
        tag = tag.strip()
        t = re.sub(r'\s+', '', tag)
        if len(t) > 0 and not tag in exists:
            exists.add(t)
            tags.append(tag)
    
    return ', '.join(tags)

