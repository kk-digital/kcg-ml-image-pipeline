
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd

import sys
base_dir = './'
sys.path.insert(0, base_dir)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', default=None, type=str, help='Path of csv for downloaded image data')

    return parser.parse_args() 
def main():
    args = parse_args()

    if args.path is None:
        print('No path specified')
        return
    
    try:
        downloaded_image_data = pd.read_csv(args.path)
    except Exception as e:
        print(e)
        return

    cfg_scales = np.sort(downloaded_image_data['task_cfg_scale'].unique())
    seeds = np.sort(downloaded_image_data['task_seed'].unique())

    len_cfg_scales = len(cfg_scales)
    len_seeds = len(seeds)

    cfg_scale_to_index = {}
    seed_to_index = {}

    for index, cfg_scale in enumerate(cfg_scales):
        cfg_scale_to_index[cfg_scale] = index
    for index, seed in enumerate(seeds):
        seed_to_index[seed] = index

    print(cfg_scale_to_index, seed_to_index)

    fig, ax = plt.subplots(len_seeds, len_cfg_scales, figsize=(len_cfg_scales * len_seeds, len_cfg_scales * len_seeds))

    for i in range(len_seeds):
        for j in range(len_cfg_scales):
            ax[i, j].axis('off')

    for i in range(len(downloaded_image_data)):
        image_path = downloaded_image_data.loc[i, 'downloaded_image_path']

        cfg_scale = downloaded_image_data.loc[i, 'task_cfg_scale']
        seed = downloaded_image_data.loc[i, 'task_seed']
        
        col = cfg_scale_to_index[downloaded_image_data.loc[i, 'task_cfg_scale']]
        row = seed_to_index[downloaded_image_data.loc[i, 'task_seed']]
        if image_path is not None:
            image = plt.imread(image_path, format='jpg')
            ax[row, col].imshow(image)
        else:
            ax[row, col].text(0.5, 0.5, 'Failed', ha='center', va='center', fontsize=8)
        ax[row, col].axis('off')
        ax[row, col].set_title(f'cfg_scale: {cfg_scale_to_index} seed: {seed}', fontsize=8)

    plt.savefig('test/output/cfg_scale/result_on_cfg_scale.png', format='png')
    plt.show()

if __name__ == '__main__':
    main()