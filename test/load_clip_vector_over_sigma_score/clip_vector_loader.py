import sys
import numpy as np
import json
import argparse

base_dir = './'
sys.path.insert(0, base_dir)

class ClipVectorLoader:

    def __init__(self, min_sigma_score):
        # available sigma score list
        self.support_sigma_score = [0]

        # if sigam score is not supported, it will raise the exception
        if min_sigma_score not in self.support_sigma_score:
            raise Exception('Not support such min sigma score: {}'.format(min_sigma_score))

        # data type of memory mapping of numpy
        self.dtype = np.float16
        self.min_sigma_score = min_sigma_score

        # load memory mapping of numpy and config file
        self.mmap_config, self.mmapping_array = self.load_mmap()

    # get the memory mapping file name and config file name
    def get_file_name(self, min_sigma_score):
        mmap_fname = 'output/clip_{}_sigma.dat'.format(min_sigma_score)
        config_fname = 'output/clip_{}_sigma.json'.format(min_sigma_score)

        return mmap_fname, config_fname

    # load memory mapping file and config file
    def load_mmap(self):

        mmap_fname, config_fname = self.get_file_name(self.min_sigma_score)
        
        with open(config_fname, 'r') as file:
            json_string = file.read()
        mmap_config = json.loads(json_string)

        with open(mmap_fname, 'r') as file:
            # Todo // fix the mmap config key 
            mmapping_array = np.memmap(file, dtype=self.dtype, mode='r', shape=(mmap_config['len-mmap'], mmap_config['dimension']))

        return mmap_config, mmapping_array

    # get clip vector from start index to end index in loaded memory mapping of numpy
    def get_clip_vector(self, start_index, end_index):

        start_index = max([0, start_index])
        end_index = max([0, end_index])
        start_index = min([start_index, self.mmap_config["loaded-count"]])
        end_index = min([end_index, self.mmap_config["loaded-count"]])

        return self.mmapping_array[start_index:end_index, :1280].tolist(), \
            self.mmapping_array[start_index:end_index, -1].tolist()
    
    # random sample count of clip vectors from loaded memory mapping of numpy
    def get_clip_vector_by_random(self, count):

        # generate the random index
        random_index = np.random.choice(np.arange(self.mmap_config["loaded-count"]), count)

        return self.mmapping_array[random_index, :1280].tolist(), \
            self.mmapping_array[random_index, -1].tolist()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=10)
    parser.add_argument('--count', type=int, default=2)
    parser.add_argument('--min-sigma-score', type=int, default=0)

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    # Test the function of ClipVectorLoader class
    clip_vector_loader = ClipVectorLoader(min_sigma_score=args.min_sigma_score)

    clip_vecotors, scores = \
        clip_vector_loader.get_clip_vector(start_index=args.start, end_index=args.end)
    
    clip_vecotors, scores = \
        clip_vector_loader.get_clip_vector_by_random(count=args.count)

    for clip_vector, score in zip(clip_vecotors, scores):
        print(clip_vector, score)