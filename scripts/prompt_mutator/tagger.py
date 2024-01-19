import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def softmax(x):
    x_exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    softmax_output = x_exp / np.sum(x_exp, axis=1, keepdims=True)
    return softmax_output


class Tagger:
    
    def __init__(self, tags: dict, tag_embs: np.ndarray):

        self.tags = tags
        self.tag_embs = tag_embs

        self.tag_names = list()
        self.subtags = list()
        self.index_list = list()
        offset = 0
        for info in tags:
            self.tag_names.append(info['name'])
            self.subtags.append(info['subtags'])
            self.index_list.append([i + offset for i in range(len(info['subtags']))])
            offset += len(info['subtags'])

    def get_similarity_and_probability(self, sample_embs: np.ndarray):

        similarity_matrix = cosine_similarity(sample_embs, self.tag_embs)
        
        # Merge columns as specified in index_list
        merged_similarity_matrix = [similarity_matrix[:, indices].max(axis=-1) for indices in self.index_list]
        merged_similarity_matrix = np.stack(merged_similarity_matrix, axis=-1)

        merged_probability_matrix = softmax(merged_similarity_matrix * 100)
    
    def get_tag_similarity(self, tag_name: str, text_emb: np.ndarray):
        """
        Calculates the cosine similarity and probability for a specific tag and text embedding.

        :param tag_name: The name of the main tag.
        :param text_emb: The text embedding (numpy array).
        :return: A tuple (similarity, probability) for the specified tag.
        """
        if tag_name not in self.tag_names:
            raise ValueError("Tag name not found in the list of tags")

        # Find the index of the tag
        tag_index = self.tag_names.index(tag_name)
        tag_emb_indices = self.index_list[tag_index]

        # Calculate cosine similarity for the specific tag embeddings
        similarities = cosine_similarity(text_emb.reshape(1, -1), [self.tag_embs[index] for index in tag_emb_indices])

        # Aggregate the similarities (e.g., by taking the maximum similarity)
        aggregated_similarity = np.max(similarities)

        return aggregated_similarity