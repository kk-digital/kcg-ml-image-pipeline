import numpy as np
import pandas as pd


class Binning:

    def __init__(self, start: float, count: int, step: float):

        assert count > 0 and step > 0
        
        self.thresholds = np.arange(count).astype(float) * step + start
            
        self.start = start
        self.count = count
        self.step = step
        self.end = self.thresholds[-1]

    def convert(self, X: np.ndarray):
        """
        Transform a continuous value X into an N-dimensional vector p based on given thresholds.
        This function handles arrays X of any shape and a 1D array of thresholds.
        
        The transformation is as follows:
        - If x < th0, then p[0] = 1.
        - If x > th(N-1), then p[N-1] = 1.
        - If th[i] < x < th[i+1], then p[i] = (th[i+1] - x) / s and p[i+1] = (x - th[i]) / s,
          where s is the step size (assumed to be constant between thresholds).
        
        The function uses numpy operations to efficiently process the transformation
        without explicit looping.
    
        Parameters:
        - X : ndarray
            An array of any shape containing the continuous values to be transformed.
    
        Returns:
        - p_vectors : ndarray
            An array of shape (..., N) where each "row" corresponds to the transformed
            vector for each element in X. Each vector p has the property that sum(p) = 1.
    
        Note:
        This function assumes that the thresholds are equally spaced.
        """

        # Apply clipping to handle boundary conditions
        X_clipped = np.clip(X, self.start, self.end)

        # Calculate normalized distances for each threshold
        distances = (X_clipped[..., None] - self.thresholds) / self.step
        
        # Apply clipping to handle boundary conditions
        distances_clipped = np.clip(distances, -1, 1)
        
        # Calculate p vectors
        p_vectors = np.maximum(0, 1 - np.abs(distances_clipped))

        return p_vectors
        
    def revert(self, p_vectors: np.ndarray):
    
        """
        Revert the transformed p_vectors back to the original continuous values X.
        This function assumes the input p_vectors are obtained from the previous transformation
        and uses the provided thresholds to revert the transformation.
    
        Parameters:
        - p_vectors : ndarray
            An array of shape (..., N) containing transformed vectors, where each vector p
            has the property that sum(p) = 1.
    
        Returns:
        - X_reverted : ndarray
            An array of the original continuous values from which the p_vectors were derived.
            The shape of X_reverted is the same as the shape of p_vectors, except for the last dimension.
    
        Note:
        This function assumes that the thresholds are equally spaced and that the p_vectors
        are correctly formatted (sum to 1 for each vector).
        """
    
        # Calculate the weighted sum of the thresholds
        X_reverted = np.sum(p_vectors * self.thresholds, axis=-1)
    
        return X_reverted


def get_entropy(norm_probabilities: np.ndarray):
    
    '''

    calculate information entropy

    entropy_i = - sum(p_ij * log2(p_ij))

    Input:
        - norm_probabilities:  np.ndarray, shape is (..., N), norm_probabilities.sum(-1) should be 1.
            
    Output:
        - entropy: np.ndarray, shape is (...,)
    
    '''

    entropy = - np.clip((norm_probabilities * np.log2(norm_probabilities + 1e-7)).sum(axis=-1), -np.inf, 0)
    
    return entropy


class SigmaScoresWithEntropy:

    def __init__(self, sigma_scores: np.ndarray, binning: Binning):
        
        '''
        Input:
            - sigma_scores:  np.ndarray, shape is (n_samples, n_models)
            - binning:  Binning
        '''

        self.sigma_scores = sigma_scores
        self.binning = binning

        #
        self.num_below_start = (sigma_scores < binning.start).sum(axis=-1)
        self.num_above_end = (sigma_scores > binning.end).sum(axis=-1)

        # binning
        self.p_vectors = binning.convert(sigma_scores) # (n_samples, n_models, n_bins)

        # get probability
        self.p = self.p_vectors.sum(axis=-2) # (n_samples, n_bins)
        self.p_norm = self.p / self.sigma_scores.shape[-1] # (n_samples, n_bins)

        # get entropy
        self.entropy = get_entropy(self.p_norm) # (n_samples,)

    @property
    def n_models(self):
        '''
        Output:
            - n_models: int
        '''
        return self.sigma_scores.shape[-1]

    @property
    def mean(self):
        '''
        Output:
            - mean: np.ndarray, shape is (n_samples,)
        '''
        return np.mean(self.sigma_scores, axis=-1)

    @property
    def variance(self):
        '''
        Output:
            - variance: np.ndarray, shape is (n_samples,)
        '''
        return np.var(self.sigma_scores, axis=-1)

    @property
    def min(self):
        '''
        Output:
            - min value: np.ndarray, shape is (n_samples,)
        '''
        return np.min(self.sigma_scores, axis=-1)

    @property
    def max(self):
        '''
        Output:
            - max value: np.ndarray, shape is (n_samples,)
        '''
        return np.max(self.sigma_scores, axis=-1)

    def to_dataframe(self):

        df = pd.DataFrame()
        
        df['sigma_score_mean'] = self.mean
        df['sigma_score_var'] = self.variance
        df['sigma_score_max'] = self.max
        df['sigma_score_min'] = self.min
        df['entropy'] = self.entropy
        df['n_models'] = self.num_below_start
        df['num_below_start'] = self.num_below_start
        df['num_above_end'] = self.num_above_end
        df[['bin_start', 'bin_count', 'bin_step']] = self.binning.start, self.binning.count, self.binning.step
        df['p'] = list(map(
            lambda x: ';'.join(map('{:.4g}'.format, x)), 
            self.p
        ))
        return df