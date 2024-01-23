import torch
import numpy as np


class Tagging(torch.nn.Module):
    
    def __init__(self, tags: dict, tag_embs: torch.Tensor):

        '''
        tags: dict like [{
                'name': 'tag name',
                'subtags': ['subtag 1', 'subtag2', ...]
            }]
        tag_embs: torch.Tensor, shape is (num_all_subtags, emb_dims)
        '''
        
        super(Tagging, self).__init__()

        self.tags = tags
        self.tag_embs = torch.nn.functional.normalize(tag_embs, p=2, dim=-1).detach()

        self.tag_names = list()
        self.subtags = list()
        self.index_list = list()
        offset = 0
        for info in tags:
            self.tag_names.append(info['name'])
            self.subtags.extend(info['subtags'])
            self.index_list.append([i + offset for i in range(len(info['subtags']))])
            offset += len(info['subtags'])
        self.tag_names = np.array(self.tag_names)
        self.subtags = np.array(self.subtags)

    def get_similarity_and_probability(self, sample_embs: torch.Tensor):

        sample_embs = torch.nn.functional.normalize(sample_embs, p=2, dim=-1)
        similarity_matrix = torch.mm(sample_embs, self.tag_embs.t())
        
        # Merge columns as specified in index_list
        merged_similarity_matrix = [similarity_matrix[:, indices].max(dim=-1).values for indices in self.index_list]
        merged_similarity_matrix = torch.stack(merged_similarity_matrix, dim=-1)

        merged_probability_matrix = torch.softmax(merged_similarity_matrix * 100, dim=-1)

        return similarity_matrix, merged_similarity_matrix, merged_probability_matrix

    def forward(self, sample_embs: torch.Tensor):
        '''
        sample_embs: torch.Tensor, shape is (num_samples, emb_dims)
        '''

        similarity_matrix, merged_similarity_matrix, merged_probability_matrix = self.get_similarity_and_probability(sample_embs)

        result = merged_probability_matrix.max(dim=-1)
        tag_indices, tag_probas = result.indices, result.values

        tag_logits = torch.gather(merged_similarity_matrix, dim=-1, index=tag_indices.unsqueeze(-1)).squeeze(-1)
        
        tag_names = self.tag_names[tag_indices.detach().cpu().numpy()]
        
        subtag_indices = similarity_matrix.max(dim=-1).indices
        subtag_names = self.subtags[subtag_indices.detach().cpu().numpy()]

        return tag_logits, tag_probas, tag_names, subtag_names

