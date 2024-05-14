
import matplotlib.pyplot as plt
import os
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--col', default=2, type=int)
    parser.add_argument('--path', default='test/output/cfg_scale/task_cfg_scales.json', type=str)

    return parser.parse_args() 
def main():
    args = parse_args()

    with open(args.path, 'r') as f:
        cfg_scale_to_image_path = json.load(f)
    cfg_scales = list(cfg_scale_to_image_path.keys())
    image_paths = list(cfg_scale_to_image_path.values())

    col = args.col

    fig, ax = plt.subplots(len(image_paths) // col, col, figsize=(10,10))

    for i in range(len(image_paths) // col * col):
        ax[i // col, i % col].axis('off')

    for index, image_path in enumerate(image_paths):
        if image_path is not None:
            image = plt.imread(image_path, format='jpg')
            ax[index // col, index % col].imshow(image)
        else:
            ax[index // col, index % col].text(0.5, 0.5, 'Failed', ha='center', va='center', fontsize=20)
        ax[index // col, index % col].axis('off')
        ax[index // col, index % col].set_title(f'cfg scale: {cfg_scales[index]}')

    plt.savefig('test/output/cfg_scale/result_on_cfg_scale.png', format='png')
    plt.show()

if __name__ == '__main__':
    main()