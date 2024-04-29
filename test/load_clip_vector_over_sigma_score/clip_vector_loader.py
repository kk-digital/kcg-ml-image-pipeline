import sys
import numpy as np
import json

base_dir = './'
sys.path.insert(0, base_dir)

def get_clip_0_sigma(count=0):

    dtype = np.float16
    shape = (1000000, 1281)

    with open('data.json', 'r') as file:
        json_string = file.read()
    
    mmap_config = json.loads(json_string)

    with open('output/clip_0_sigma.dat', 'r') as f:
        mmapping_array = np.memmap(f, dtype=dtype, mode='r', shape=shape)

        for i in range(100):
            print(mmapping_array[i, :])

    # update the count
    if count > mmap_config["len-mmap"]:
        count = mmap_config["len-mmap"]

    return mmapping_array[:count, 1280], mmapping_array[:count, -1]


if __name__ == '__main__':
    print(get_clip_0_sigma(100))