
import sys
import time
import requests

base_directory = "./"
sys.path.insert(0, base_directory)

from generation_task.icon_generation_task import IconGenerationTask
from generation_task.image_generation_task import ImageGenerationTask

from worker.image_generation.scripts.generate_images_with_inpainting_from_prompt_list import run_generate_images_with_inpainting_from_prompt_list



SERVER_ADRESS = 'http://127.0.0.1:8000'

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

def http_get_job():
    url = SERVER_ADRESS + "/get-job"
    job = requests.get(url)
    job_json = job.json()

    return job_json

def http_add_job(job):
    url = SERVER_ADRESS + "/get-list-pending-jobs"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data
    response = requests.post(url, json=job, headers=headers)

    if response.status_code != 201:
        print(f"POST request failed with status code: {response.status_code}")

def main():
    http_add_job({
        "uuid": 1,
        "task_type": "icon_generation_task",
        "task_creation_time": "ignore",
        "task_input_dict": {
            'prompt': "icon",
            'cfg_strength': 7.5,
            'iterations': 1,
            'denoiser': "",
            'seed': '',
            'output_path': "./output/inpainting/",
            'num_images': 6,
            'image_width': 512,
            'image_height': 512,
            'batch_size': 1,
            'checkpoint_path': 'input/model/sd/v1-5-pruned-emaonly/v1-5-pruned-emaonly.safetensors',
            'flash': False,
            'device': "cuda",
            'sampler': "ddim",
            'steps': 20,
            'prompt_list_dataset_path': './input/civit_ai_data_phrase_count_v6.csv',
            'init_img': './test/test_inpainting/white_512x512.jpg',
            'init_mask': './test/test_inpainting/icon_mask.png',
        },

        "task_input_file_dict": {},
        "task_output_file_dict": {},
    })


    while True:
        job = http_get_job()
        print(job)
        if job != None:
            task = None
            task_type = task['generation_task_type']

            if task_type == 'icon_generation_task':
                generation_task = IconGenerationTask.from_dict(task)
                run_generation_task(generation_task)

            elif task_type == 'image_generation_task':
                generation_task = ImageGenerationTask.from_dict(task)
                run_generation_task(generation_task)

        time.sleep(1 * 1000)



