
import sys
import time
import requests
from datetime import datetime

base_directory = "./"
sys.path.insert(0, base_directory)

from worker.image_generation.generation_task.icon_generation_task import IconGenerationTask
from worker.image_generation.generation_task.image_generation_task import ImageGenerationTask

from worker.image_generation.scripts.generate_images_with_inpainting_from_prompt_list import run_generate_images_with_inpainting_from_prompt_list



SERVER_ADRESS = 'http://127.0.0.1:8000'

# Running inpainting using the inpainting script
# TODO(): each generation task should have its own function


class GenerateImagesWithInpaintingFromPromptListArguments:
    def __init__(self, prompt_list_dataset_path, num_images, init_img, init_mask, sampler_name, batch_size, n_iter,
                 steps, cfg_scale, width, height, outpath, mask_blur, inpainting_fill, styles, resize_mode, denoising_strength,
                 image_cfg_scale, inpaint_full_res_padding, inpainting_mask_invert):

        self.prompt_list_dataset_path = prompt_list_dataset_path
        self.num_images = num_images
        self.init_img = init_img
        self.init_mask = init_mask
        self.sampler_name = sampler_name
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.width = width
        self.height = height
        self.outpath = outpath
        self.mask_blur = mask_blur
        self.inpainting_fill = inpainting_fill
        self.styles = styles
        self.resize_mode = resize_mode
        self.denoising_strength = denoising_strength
        self.image_cfg_scale = image_cfg_scale
        self.inpaint_full_res_padding = inpaint_full_res_padding
        self.inpainting_mask_invert = inpainting_mask_invert

def run_generation_task(generation_task):

    # Instead of using cli arguments, we are using the
    # Generation_task class to provide the parameters
    args = GenerateImagesWithInpaintingFromPromptListArguments(prompt_list_dataset_path=generation_task.prompt_list_dataset_path,
                                                               num_images=generation_task.num_images,
                                                               init_img=generation_task.init_img,
                                                               init_mask=generation_task.init_mask,
                                                               sampler_name=generation_task.sampler,
                                                               batch_size=1,
                                                               n_iter=generation_task.num_images,
                                                               steps=generation_task.steps,
                                                               cfg_scale=generation_task.cfg_strength,
                                                               width=generation_task.image_width,
                                                               height=generation_task.image_height,
                                                               outpath=generation_task.output_path,
                                                               mask_blur=generation_task.mask_blur,
                                                               inpainting_fill=generation_task.inpainting_fill,
                                                               styles=generation_task.styles,
                                                               resize_mode=generation_task.resize_mode,
                                                               denoising_strength=generation_task.denoising_strength,
                                                               image_cfg_scale=generation_task.image_cfg_scale,
                                                               inpaint_full_res_padding=generation_task.inpaint_full_res_padding,
                                                               inpainting_mask_invert=generation_task.inpainting_mask_invert)

    run_generate_images_with_inpainting_from_prompt_list(args)

# Get request to get an available job
def http_get_job():
    url = SERVER_ADRESS + "/get-job"
    response = requests.get(url)

    if response.status_code == 200:
        job_json = response.json()
        return job_json

    return None

# Used for debugging purpose
# The worker should not be adding jobs
def http_add_job(job):
    url = SERVER_ADRESS + "/add-job"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data
    response = requests.post(url, json=job, headers=headers)

    if response.status_code != 201 and response.status_code != 200:
        print(f"POST request failed with status code: {response.status_code}")


def http_update_job_completed(job):
    url = SERVER_ADRESS + "/update-job-completed"
    print(url)
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data

    response = requests.put(url, json=job, headers=headers)

    print(response)

    if response.status_code != 200:
        print(f"request failed with status code: {response.status_code}")

def http_update_job_failed(job):
    url = SERVER_ADRESS + "/update-job-failed"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data

    response = requests.put(url, json=job, headers=headers)

    if response.status_code != 200:
        print(f"request failed with status code: {response.status_code}")

def main():
    print("starting")

    # for debugging purpose only

    job = {
        "uuid": '1',
        "task_type": "icon_generation_task",
        "task_creation_time": "ignore",
        "model_name" : "sd",
        "model_file_name": "N/A",
        "model_file_path": "N/A",
        "sd_model_hash": "N/A",
        "task_creation_time": "N/A",
        "task_start_time": "N/A",
        "task_completion_time": "N/A",
        "task_error_str": "",
        "task_input_dict": {
            'prompt': "icon",
            'cfg_strength': 12,
            'iterations': 1,
            'denoiser': "",
            'seed': '',
            'output_path': "./output/inpainting/",
            'num_images': 2,
            'image_width': 512,
            'image_height': 512,
            'batch_size': 1,
            'checkpoint_path': 'input/model/sd/v1-5-pruned-emaonly/v1-5-pruned-emaonly.safetensors',
            'flash': False,
            'device': "cuda",
            'sampler': "ddim",
            'steps': 20,
            'prompt_list_dataset_path': './input/prompt_list_civitai_10000.zip',
            'init_img': './test/test_inpainting/white_512x512.jpg',
            'init_mask': './test/test_inpainting/icon_mask.png',

            'mask_blur' : 4,
            'inpainting_fill': 1,
            'styles': [],
            'resize_mode': 0,
            'denoising_strength': 0.75,
            'image_cfg_scale': 1.5,
            'inpaint_full_res_padding': 32,
            'inpainting_mask_invert': 0
        },
        "task_input_file_dict": {},
        "task_output_file_dict": {},
    }

    http_add_job(job)
    #http_add_job(job)


    while True:
        print("Looking for jobs ! ")
        job = http_get_job()
        if job != None:
            print("Found job ! ")
            job['task_start_time'] = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')

            # Convert the job into a dictionary
            # Then use the dictionary to create the generation task
            task = {
                'generation_task_type' : job['task_type'],
                'prompt': job['task_input_dict']['prompt'],
                'model_name': job['model_name'],
                'cfg_strength': job['task_input_dict']['cfg_strength'],
                'num_images': job['task_input_dict']['num_images'],
                'iterations': job['task_input_dict']['iterations'],
                'denoiser': job['task_input_dict']['denoiser'],
                'seed': job['task_input_dict']['seed'],
                'output_path': job['task_input_dict']['output_path'],
                'image_width': job['task_input_dict']['image_width'],
                'image_height': job['task_input_dict']['image_height'],
                'batch_size': job['task_input_dict']['batch_size'],
                'checkpoint_path': job['task_input_dict']['checkpoint_path'],
                'flash': job['task_input_dict']['flash'],
                'device': job['task_input_dict']['device'],
                'sampler': job['task_input_dict']['sampler'],
                'steps': job['task_input_dict']['steps'],
                'prompt_list_dataset_path': job['task_input_dict']['prompt_list_dataset_path'],
                'init_img': job['task_input_dict']['init_img'],
                'init_mask': job['task_input_dict']['init_mask'],

                'mask_blur': job['task_input_dict']['mask_blur'],
                'inpainting_fill': job['task_input_dict']['inpainting_fill'],
                'styles': job['task_input_dict']['styles'],
                'resize_mode': job['task_input_dict']['resize_mode'],
                'denoising_strength': job['task_input_dict']['denoising_strength'],
                'image_cfg_scale': job['task_input_dict']['image_cfg_scale'],
                'inpaint_full_res_padding': job['task_input_dict']['inpaint_full_res_padding'],
                'inpainting_mask_invert': job['task_input_dict']['inpainting_mask_invert']
            }

            # Switch on the task type
            # We have 2 for now
            # And they are identical
            task_type = task['generation_task_type']

            if task_type == 'icon_generation_task':
                generation_task = IconGenerationTask.from_dict(task)

                # Run inpainting task
                try:
                    run_generation_task(generation_task)
                    print("job completed !")
                    job['task_completion_time'] = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
                    http_update_job_completed(job)
                except Exception as e:
                    print(f"generation task failed: {e}")
                    job['task_error_str'] = e
                    http_update_job_failed(job)


            elif task_type == 'image_generation_task':
                generation_task = ImageGenerationTask.from_dict(task)
                # Run inpainting task
                try:
                    run_generation_task(generation_task)
                    print("job completed !")
                    job['task_completion_time'] = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
                    http_update_job_completed(job)
                except Exception as e:
                    print(f"generation task failed: {e}")
                    job['task_error_str'] = e
                    http_update_job_failed(job)

        else:
            # If there was no job, go to sleep for a while
            sleep_time_in_seconds = 10
            print("Did not find job, going to sleep for ", sleep_time_in_seconds, " seconds")
            time.sleep(sleep_time_in_seconds)




if __name__ == '__main__':
    main()
