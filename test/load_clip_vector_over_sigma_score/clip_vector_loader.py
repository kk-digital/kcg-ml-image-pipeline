import sys
import numpy as np

base_dir = './'
sys.path.insert(0, base_dir)

def get_clip_0_sigma():

    dtype = np.float16
    shape = (1000000, 1281)

    with open('output/clip_0_sigma.dat', 'r') as f:
        mmapping_array = np.memmap(f, dtype=dtype, mode='w+', shape=shape)

        for i in range(100):
            print(mmapping_array[i, :])