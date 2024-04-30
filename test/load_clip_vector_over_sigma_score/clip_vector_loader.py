import sys
import numpy as np
import json

base_dir = './'
sys.path.insert(0, base_dir)

def get_clip_0_sigma(start_index, end_index):

    dtype = np.float16
    shape = (1000000, 1281)

    with open('output/clip_0_sigma.json', 'r') as file:
        json_string = file.read()
    
    mmap_config = json.loads(json_string)

    with open('output/clip_0_sigma.dat', 'r') as f:
        mmapping_array = np.memmap(f, dtype=dtype, mode='r', shape=shape)

    start_index = max([0, start_index])
    end_index = max([0, end_index])
    start_index = min([start_index, mmap_config["loaded-count"]])
    end_index = min([end_index, mmap_config["loaded-count"]])

    return mmapping_array[start_index:end_index, :1280].tolist(), mmapping_array[start_index:end_index, -1].tolist()


if __name__ == '__main__':
    clip_vecotors, scores = get_clip_0_sigma(100)
    for clip_vector, score in zip(clip_vecotors, scores):
        print(clip_vector, score)