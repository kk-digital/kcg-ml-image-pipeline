import faiss
import torch
import time
import json

from tqdm import tqdm

import sys
base_dir = './'
sys.path.insert(0, base_dir)


from test.load_clip_vector_over_sigma_score.clip_vector_loader import ClipVectorLoader

def main():
    loader = ClipVectorLoader(min_sigma_score=-1000, dataset='environmental')
    loader.load_clip_vector_over_sigma_score()

    clip_vectors = loader.get_all_clip_vector()

    len_clip_vectors = len(clip_vectors)

    nlist=50
    quantizer = faiss.IndexFlatL2(len_clip_vectors)
    index = faiss.IndexIVFFlat(quantizer, len_clip_vectors, nlist)
    
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    index.train(clip_vectors)
    index.add(clip_vectors)

    with open('test_pinecone_clip_vector_search.json', mode='r', newline='') as f:
        pinecone_test_data = json.load(f)
        
        test_speed_data = []
        for ele in tqdm(pinecone_test_data):
            top_k = ele['top_k']
            query_clip_vector = ele['query_clip_vector']

            start_time = time.time()
            distance, indices = index.search(query_clip_vector, top_k)
            elapsed_time = time.time() - start_time

            ele['faiss_elapsed_time'] = elapsed_time
            del ele['query_clip_vector']

            test_speed_data.append(ele)
    
    with open('output/test_speed_faiss_and_pinecone.json', mode='w+', newline='') as result_file:
        json.dump(test_speed_data, result_file, indent=4)
        

if '__name__' == '__main__':
    main()