from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_min_distance_to_representative_samples(samples: np.ndarray, representative_samples: np.ndarray, distance_type: str = 'cosine'):
    
    '''
    
    select representative samples. can continue from previous representative samples.
    the min distance between representative samples should be greater than the given threshold.
    the min distance between a unselected sample with representative samples should be lower than the given threshold.
    
    Input:
        - samples: np.ndarray, shape of (n_samples, n_features)
        - representative_samples: np.ndarray, shape of (n_existed_samples, n_features).
        - distance_type: str, method to compute distance, cosine as default. 
            
    Output:
        - distances: np.ndarray, shape of (n_samples, ).
    
    '''

    if distance_type == 'cosine':

        norm_samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
        norm_representative_samples = representative_samples / np.linalg.norm(representative_samples, axis=1, keepdims=True)
        similarity = np.dot(norm_samples, norm_representative_samples.T)

        distance_matrix = 1 - similarity
        
    else:
        raise f'ERROR! unknown distance_type: {distance_type}'


    distances = distance_matrix.min(axis=-1)

    return distances
    
    
def representative_sample_selection(samples: np.ndarray, threshold: float, existed_samples: np.ndarray = None, distance_type: str = 'cosine', display: bool = True):
    
    '''
    
    select representative samples. can continue from previous representative samples.
    the min distance between representative samples should be greater than the given threshold.
    the min distance between a unselected sample with representative samples should be lower than the given threshold.
    
    Input:
        - samples: np.ndarray, shape of (n_samples, n_features)
        - threshold: float.
        - existed_samples: np.ndarray, shape of (n_existed_samples, n_features), previous representative samples, None as default.
        - distance_type: str, method to compute distance, cosine as default. 
        - display: bool, whether show progressing bar, True as default.
            
    Output:
        - indices: list[int], representative sample indices from input samples.
    
    '''

    if distance_type == 'cosine':

        norm_samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
        similarity = np.dot(norm_samples, norm_samples.T)

        distance_matrix = 1 - similarity
        
        if existed_samples is not None:

            norm_existed_samples = existed_samples / np.linalg.norm(existed_samples, axis=1, keepdims=True)
            similarity = np.dot(norm_samples, norm_existed_samples.T)
            existed_distance_matrix = 1 - similarity

            distance_matrix = np.concatenate([existed_distance_matrix, distance_matrix], axis=1)

    else:
        raise f'ERROR! unknown distance_type: {distance_type}'    

    return representative_sample_selection_by_distance(distance_matrix, threshold, display)


def representative_sample_selection_by_distance(distance_matrix: np.ndarray, threshold: float, display: bool = True):
    
    '''
    
    Input:
        - distance_matrix: np.ndarray, shape of (n_samples, n_samples) or (n_samples, n_existed_samples + n_samples)
        - threshold: float.
        - display: bool, whether show progressing bar, True as default.
            
    Output:
        - indices: list[int], representative sample indices from input samples.
    
    '''

    if distance_matrix.shape[0] == distance_matrix.shape[1]:

        mean_d = distance_matrix.mean(axis=-1)
        index = np.argmin(mean_d)
        
        selected = [index]
        remaining = np.concatenate([np.arange(index+1, distance_matrix.shape[0]), np.arange(index)])
        
        sub_d = np.concatenate([
            distance_matrix[index+1:],
            distance_matrix[:index] 
        ], axis=0)
        sub_d = np.concatenate([
            sub_d[:, index:],
            sub_d[:, :index] 
        ], axis=1)

        n_selected = len(selected)

    else:
        
        selected = []
        remaining = np.arange(distance_matrix.shape[0])

        sub_d = distance_matrix

        n_selected = distance_matrix.shape[1] - distance_matrix.shape[0]

    if display:
        bar = tqdm(total=remaining.shape[0])
    else:
        bar = None
    
    while remaining.shape[0] > 0:
        
        mask = sub_d[:, :n_selected].min(axis=-1) > threshold

        remaining = remaining[mask]
        sub_d = sub_d[mask]
        sub_d = np.concatenate([
            sub_d[:, :n_selected], 
            sub_d[:, np.arange(n_selected, sub_d.shape[1])[mask]]
        ], axis=-1)

        if bar is not None:
            bar.update((~mask).sum())
        
        if mask.sum() > 0:
            
            index = np.argmin(sub_d[:, n_selected:].mean(axis=0))
    
            selected_index = remaining[index]
            
            remaining = np.concatenate([remaining[:index], remaining[index+1:]])
            
            sub_d = np.concatenate([
                sub_d[:index], 
                sub_d[index+1:]
            ], axis=0)
    
            sub_d = np.concatenate([
                sub_d[:, n_selected+index:n_selected+index+1], 
                sub_d[:, :n_selected+index], 
                sub_d[:, n_selected+index+1:]
            ], axis=-1)
            
            selected = [selected_index] + selected

            n_selected += 1

            if bar is not None:
                bar.update(1)

    return selected