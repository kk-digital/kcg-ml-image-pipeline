import pandas as pd

import numpy as np

from sklearn.cluster import MiniBatchKMeans


def get_candidate_pairs_within_category(categories: np.ndarray, max_pairs: int):
    
    '''
    
    Input:
        - categories: np.ndarray[int], shape is (N,)
        - max_pairs: int, max selecting pairs. 
            max_pairs should 0 < max_pairs < (N / n_categories) ** 2.
            we will attempt to select (max_pairs / n_categories) pairs within each category.
            
    Output:
        - pairs: list[(index, index)], seleted pairs, index of input categories.
    
    '''
    
    df = pd.DataFrame()
    df['category'] = categories
    
    n_bins = len(np.unique(categories))
    
    max_pairs_within_bins = max_pairs // n_bins
    
    pairs = list()
    
    for c, g in df.groupby('category'):
        
        if g.shape[0] <= 1:
            continue
            
        sub_ids = list(g.index)
        np.random.shuffle(sub_ids)
        
        cn2 = len(sub_ids) * (len(sub_ids) - 1) / 2
        
        step = max(1, int(cn2 // max_pairs_within_bins))
        
        sub_pairs = list()
        
        for i, id_i in enumerate(sub_ids[:-1]):

            for id_j in sub_ids[i+1::step]:

                sub_pairs.append((id_i, id_j))
                
            if len(sub_pairs) > max_pairs_within_bins:
                break
                
        if len(sub_pairs) == 0:
            sub_pairs.append((sub_ids[0], sub_ids[1]))
            
        pairs += sub_pairs[:max_pairs_within_bins]
        
    return pairs


def get_bins(min_value: float, max_value: float, n_bins: int):
    
    bins = np.linspace(min_value, max_value, n_bins - 1)
    
    return bins


def score_to_category_with_bins(scores: np.ndarray, bins: np.ndarray):
    
    return np.digitize(scores, bins)


def score_to_category_with_quantiles(scores: np.ndarray, n_categories: int):
    
    rank = np.argsort(scores)
    
    n_samples = len(rank)
    
    step = int(np.ceil(n_samples / n_categories))
    
    return (rank + ((step - n_samples % step) // 2)) // step


def get_candidate_pairs_by_score(scores: np.ndarray, max_pairs: int, n_bins: int, use_quantiles: bool = False):
    
    '''
    
    Input:
        - scores: np.ndarray[float], shape is (N,)
        - max_pairs: int, max selecting pairs. 
            max_pairs should 0 < max_pairs < (N / n_bins) ** 2.
            we will attempt to select (max_pairs / n_bins) pairs within each category.
        - n_bins: int, number of categories to be divided
        - use_quantiles: bool, to use quantiles or fixed step bins
            
    Output:
        - pairs: list[(index, index)], seleted pairs, index of input scores.
    
    '''
    
    if use_quantiles:
        
        categories = score_to_category_with_quantiles(scores=scores, n_categories=n_bins)
    
    else:

        bins = get_bins(min_value=min(scores), max_value=max(scores), n_bins=n_bins)

        categories = score_to_category_with_bins(scores=scores, bins=bins)
    
    return get_candidate_pairs_within_category(
        categories=categories, 
        max_pairs=max_pairs
    )


def embedding_to_category(embeddings: np.ndarray, n_clusters: int):
    
    model = MiniBatchKMeans(n_clusters=n_clusters, max_iter=100, n_init=3)
    
    labels = model.fit_predict(embeddings)
    
    return labels


def get_candidate_pairs_by_embedding(embeddings: np.ndarray, max_pairs: int, n_clusters: int):
    
    '''
    
    Input:
        - embeddings: np.ndarray, shape is (N, 768)
        - max_pairs: int, max selecting pairs. 
            max_pairs should 0 < max_pairs < (N / n_clusters) ** 2.
            we will attempt to select (max_pairs / n_clusters) pairs within each category.
        - n_clusters: int, number of categories to be divided
            
    Output:
        - pairs: list[(index, index)], seleted pairs, index of input embeddings.
    
    '''
    
    categories = embedding_to_category(embeddings=embeddings, n_clusters=n_clusters)

    return get_candidate_pairs_within_category(
        categories=categories, 
        max_pairs=max_pairs
    )


