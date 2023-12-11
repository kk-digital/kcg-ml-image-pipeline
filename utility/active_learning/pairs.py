import pandas as pd

import numpy as np

from sklearn.cluster import KMeans, MiniBatchKMeans


def get_candidate_pairs_within_category(job_uuids: list, categories: np.ndarray, max_pairs: int):
    
    '''
    
    Input:
        - job_uuids: list[str], length of N
        - categories: np.ndarray[int], shape is (N,)
        - max_pairs: int, max selecting pairs. 
            max_pairs should 0 < max_pairs < (N / n_categories) ** 2.
            we will attempt to select (max_pairs / n_categories) pairs within each category.
            
    Output:
        - pairs: list[(str, str)], seleted job_uuid pairs.
    
    '''
    
    df = pd.DataFrame(np.stack([job_uuids, categories], axis=-1), columns=['job_uuid', 'category'])
    
    n_bins = len(np.unique(categories))
    
    max_pairs_within_bins = max_pairs // n_bins
    
    pairs = list()
    
    for c, g in df.groupby('category'):
        
        if g.shape[0] <= 1:
            continue
            
        sub_uuids = list(g['job_uuid'])
        np.random.shuffle(sub_uuids)
        
        cn2 = len(sub_uuids) * (len(sub_uuids) - 1) / 2
        
        step = max(1, int(cn2 // max_pairs_within_bins))
        
        sub_pairs = list()
        
        for i, uuid_i in enumerate(sub_uuids[:-1]):

            for uuid_j in sub_uuids[i+1::step]:

                sub_pairs.append((uuid_i, uuid_j))
                
            if len(sub_pairs) > max_pairs_within_bins:
                break
                
        if len(sub_pairs) == 0:
            sub_pairs.append((sub_uuids[0], sub_uuids[1]))
            
        pairs += sub_pairs[:max_pairs_within_bins]
        
    return pairs


def get_bins(min_value: float, max_value: float, n_bins: int):
    
    bins = np.linspace(min_value, max_value, n_bins - 1)
    
    return bins


def score_to_category_with_bins(scores: np.ndarray, bins: np.ndarray):
    
    return np.digitize(scores, bins)


def score_to_category_with_quantities(scores: np.ndarray, n_categories: int):
    
    rank = np.argsort(scores)
    
    n_samples = len(rank)
    
    step = int(np.ceil(n_samples / n_categories))
    
    return (rank + ((step - n_samples % step) // 2)) // step


def get_candidate_pairs_by_score(job_uuids: list, scores: np.ndarray, max_pairs: int, n_bins: int, use_quantities: bool = False):
    
    '''
    
    Input:
        - job_uuids: list[str], length of N
        - scores: np.ndarray[float], shape is (N,)
        - max_pairs: int, max selecting pairs. 
            max_pairs should 0 < max_pairs < (N / n_bins) ** 2.
            we will attempt to select (max_pairs / n_bins) pairs within each category.
        - n_bins: int, number of categories to be divided
        - use_quantities: bool, to use quantities or fixed step bins
            
    Output:
        - pairs: list[(str, str)], seleted job_uuid pairs.
    
    '''
    
    if use_quantities:
        
        categories = score_to_category_with_quantities(scores=scores, n_categories=n_bins)
    
    else:

        bins = get_bins(min_value=min(scores), max_value=max(scores), n_bins=n_bins)

        categories = score_to_category_with_bins(scores=scores, bins=bins)
    
    return get_candidate_pairs_within_category(
        job_uuids=job_uuids, 
        categories=categories, 
        max_pairs=max_pairs
    )


def embedding_to_category(embeddings: np.ndarray, n_clusters: int):
    
    model = MiniBatchKMeans(n_clusters=n_clusters, max_iter=100, n_init=3)
    
    labels = model.fit_predict(embeddings)
    
    return labels


def get_candidate_pairs_by_embedding(job_uuids: list, embeddings: np.ndarray, max_pairs: int, n_clusters: int):
    
    '''
    
    Input:
        - job_uuids: list[str], length of N
        - embeddings: np.ndarray, shape is (N, 768)
        - max_pairs: int, max selecting pairs. 
            max_pairs should 0 < max_pairs < (N / n_clusters) ** 2.
            we will attempt to select (max_pairs / n_clusters) pairs within each category.
        - n_clusters: int, number of categories to be divided
            
    Output:
        - pairs: list[(str, str)], seleted job_uuid pairs.
    
    '''
    
    categories = embedding_to_category(embeddings=embeddings, n_clusters=n_clusters)

    return get_candidate_pairs_within_category(
        job_uuids=job_uuids, 
        categories=categories, 
        max_pairs=max_pairs
    )


