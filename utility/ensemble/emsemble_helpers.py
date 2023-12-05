import numpy as np

def get_sigma_scores_mean(sigma_scores: np.ndarray):
    
    '''
    
    Input:
        - sigma_scores: np.ndarray, shape is (n_samples, n_models)
            
    Output:
        - mean: np.ndarray, shape is (n_samples,)
    
    '''

    mean = np.mean(sigma_scores, axis=-1)

    return mean

def get_sigma_scores_variance(sigma_scores: np.ndarray):
    
    '''
    
    Input:
        - sigma_scores: np.ndarray, shape is (n_samples, n_models)
            
    Output:
        - variance: np.ndarray, shape is (n_samples,)
    
    '''

    variance = np.var(sigma_scores, axis=-1)

    return variance

def get_sigma_scores_max(sigma_scores: np.ndarray):
    
    '''
    
    Input:
        - sigma_scores: np.ndarray, shape is (n_samples, n_models)
            
    Output:
        - max: np.ndarray, shape is (n_samples,)
    
    '''

    mean = np.max(sigma_scores, axis=-1)

    return mean

def get_sigma_scores_min(sigma_scores: np.ndarray):
    
    '''
    
    Input:
        - sigma_scores: np.ndarray, shape is (n_samples, n_models)
            
    Output:
        - variance: np.ndarray, shape is (n_samples,)
    
    '''

    variance = np.var(sigma_scores, axis=-1)

    return variance


def get_bins(min_value: float, max_value: float, n_bins: int):
    
    '''
    
    Input:
        - min_value: float
        - max_value: float
        - n_bins: int, number of expected bins
            
    Output:
        - bins: np.ndarray, shape is (n_bins,)
    
    '''
    
    bins = np.linspace(min_value, max_value, n_bins - 1)
    
    return bins


def get_category_with_bins(values: np.ndarray, bins: np.ndarray):
    
    '''

    convert score to its corresponding bin id
    
    Input:
        - values:  np.ndarray, shape is (n_samples,) or (n_samples, n_models)
        - bins: np.ndarray
            
    Output:
        - category: np.ndarray, shape is (n_samples,) or (n_samples, n_models)
    
    '''
    
    return np.digitize(scores, bins)


def get_category_with_quantiles(values: np.ndarray, n_categories: int):
    
    '''

    convert score to its corresponding quantile bin id
    
    Input:
        - values:  np.ndarray, shape is (n_samples,) or (n_samples, n_models)
        - n_categories: int, number of expected bins
            
    Output:
        - category: np.ndarray, shape is (n_samples,) or (n_samples, n_models)
    
    '''
    
    rank = np.argsort(scores)
    
    n_samples = len(rank)
    
    step = int(np.ceil(n_samples / n_categories))
    
    return (rank + ((step - n_samples % step) // 2)) // step


def get_entropy(categories: np.ndarray):
    
    '''

    calculate information entropy

    entropy_i = - sum(p_ij * log2(p_ij))

    p_ij = n_ij / n_models

    n_ij is the number occurence of category j on sample i. 
    
    Input:
        - categories:  np.ndarray, shape is (n_samples, n_models)
        - n_categories: int, number of expected bins
            
    Output:
        - entropy: np.ndarray, shape is (n_samples,)
    
    '''
    
    n_bins = int(categories.max()) + 1
    
    one_hot = np.eye(n_bins)[categories].sum(axis=-2)
    
    probabilities = one_hot / one_hot.sum(axis=-1, keepdims=True)
    
    entropy = - (probabilities * np.log2(probabilities + 1e-7)).sum(axis=-1)
    
    return entropy