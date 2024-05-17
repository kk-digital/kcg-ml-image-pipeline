import faiss
import torch
import time
import json
import numpy as np
import pandas as pd

from tqdm import tqdm

import sys
base_dir = './'
sys.path.insert(0, base_dir)


from test.load_clip_vector_over_sigma_score.clip_vector_loader import ClipVectorLoader

def main():
    start_time = time.time()
    
    loader = ClipVectorLoader(min_sigma_score=-1000, dataset='environmental')
    # clip_vectors, scores = loader.get_all_clip_vector()
    clip_vectors, scores = loader.get_clip_vector_by_random(10000)
    clip_vectors = np.array(clip_vectors, dtype='float32')

    print('loading clip vectors elapsed time: {} seconds'.format(time.time() - start_time))

    # set start time
    start_time = time.time()

    len_clip_vectors = 1280

    nlist=50
    quantizer = faiss.IndexFlatL2(len_clip_vectors)
    index = faiss.IndexIVFFlat(quantizer, len_clip_vectors, nlist)

    cpu_index = faiss.IndexFlatL2(len_clip_vectors)
    
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    
    cpu_index.train(clip_vectors)
    cpu_index.add(clip_vectors)

    gpu_index.train(clip_vectors)
    gpu_index.add(clip_vectors)

    # print elasped time
    print('training and adding clip vectors elapsed time: {} seconds'.format(time.time() - start_time))

    test_speed_data = []

    with open('test/test_faiss_query_speed/test_speed_of_pinecone.json', mode='r', newline='') as f:
        pinecone_test_data = json.load(f)
        
        for ele in tqdm(pinecone_test_data):
            top_k = ele['top_k']
            query_clip_vector = ele['query_clip_vector']

            start_time = time.time()
            try:
                distance, indices = gpu_index.search(np.array([query_clip_vector], dtype='float32'), top_k)
                elapsed_time = time.time() - start_time
                ele['faiss_gpu_elapsed_time'] = elapsed_time
                ele['faiss_gpu_time_per_vector'] = elapsed_time / top_k
                ele['faiss_gpu_vectors/second'] = top_k / elapsed_time
            except Exception as e:
                ele['faiss_gpu_elapsed_time'] = -1
                ele['faiss_gpu_time_per_vector'] = -1
                ele['faiss_gpu_vectors/second'] = -1
            
            start_time = time.time()
            try:
                distance, indices = cpu_index.search(np.array([query_clip_vector], dtype='float32'), top_k)
                elapsed_time = time.time() - start_time
                ele['faiss_cpu_elapsed_time'] = elapsed_time
                ele['faiss_cpu_time_per_vector'] = elapsed_time / top_k
                ele['faiss_cpu_vectors/second'] = top_k / elapsed_time
            except Exception as e:
                ele['faiss_cpu_elapsed_time'] = -1
                ele['faiss_cpu_time_per_vector'] = -1
                ele['faiss_cpu_vectors/second'] = -1

            del ele['query_clip_vector']
            test_speed_data.append(ele)
    
    df = pd.DataFrame(test_speed_data)
    df.to_csv('output/test_speed_of_pinecone.csv', index=False)

if __name__ == '__main__':
    main()