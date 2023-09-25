
import sys

base_directory = "./"
sys.path.insert(0, base_directory)

from generation_task.icon_generation_task import IconGenerationTask
from generation_task.image_generation_task import ImageGenerationTask

from worker.image_generation.scripts.generate_images_with_inpainting_from_prompt_list import run_generate_images_with_inpainting_from_prompt_list


def run_generation_task(generation_task):
    args = {
        'prompt_list_dataset_path' : generation_task['prompt_list_dataset_path'],
        'num_images' : generation_task['num_images'],
        'init_img': generation_task['init_img'],
        'init_mask': generation_task['init_mask'],
        'sampler_name': generation_task['sampler'],
        'batch_size': 1,
        'n_iter': generation_task['num_images'],
        'steps': generation_task['steps'],
        'cfg_scale': generation_task['cfg_strength'],
        'width': generation_task['width'],
        'height': generation_task['height'],
        'outpath': generation_task['output_path']

    }
    run_generate_images_with_inpainting_from_prompt_list(args)

def main():

    while True:
        task = None
        task_type = task['generation_task_type']

        if task_type == 'icon_generation_task':
            generation_task = IconGenerationTask.from_dict(task)
            run_generation_task(generation_task)

        elif task_type == 'image_generation_task':
            generation_task = ImageGenerationTask.from_dict(task)
            run_generation_task(generation_task)



