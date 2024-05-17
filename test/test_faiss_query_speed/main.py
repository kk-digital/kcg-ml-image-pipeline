import faiss
import torch
import time
import json
import numpy as np

from tqdm import tqdm

import sys
base_dir = './'
sys.path.insert(0, base_dir)


from test.load_clip_vector_over_sigma_score.clip_vector_loader import ClipVectorLoader

def main():
    start_time = time.time()
    
    loader = ClipVectorLoader(min_sigma_score=-1000, dataset='environmental')
    clip_vectors, scores = loader.get_all_clip_vector()
    clip_vectors = np.array(clip_vectors, dtype='float32')

    print('loading clip vectors elapsed time: {} seconds'.format(time.time() - start_time))

    # set start time
    start_time = time.time()

    len_clip_vectors = 1280

    nlist=50
    quantizer = faiss.IndexFlatL2(len_clip_vectors)
    index = faiss.IndexIVFFlat(quantizer, len_clip_vectors, nlist)

    index = faiss.IndexFlatL2(len_clip_vectors)
    
    # if torch.cuda.is_available():
    #     res = faiss.StandardGpuResources()
    #     index = faiss.index_cpu_to_gpu(res, 0, index)
    
    index.train(clip_vectors)
    index.add(clip_vectors)

    # print elasped time
    print('training and adding clip vectors elapsed time: {} seconds'.format(time.time() - start_time))

    test_speed_data = []

    with open('test/test_faiss_query_speed/test_pinecone_query_speed.json', mode='r', newline='') as f:
        pinecone_test_data = json.load(f)
        
        for ele in tqdm(pinecone_test_data):
            top_k = ele['top_k']
            query_clip_vector = ele['query_clip_vector']

            start_time = time.time()
            try:
                distance, indices = index.search(np.array([query_clip_vector], dtype='float32'), top_k)
                elapsed_time = time.time() - start_time
                ele['faiss_elapsed_time'] = elapsed_time
            except Exception as e:
                ele['faiss_elapsed_time'] = -1
                
            del ele['query_clip_vector']
            test_speed_data.append(ele)
    with open('output/test_speed_faiss_and_pinecone.json', mode='w+', newline='') as result_file:
        json.dump(test_speed_data, result_file, indent=4)
        

if __name__ == '__main__':
    main()