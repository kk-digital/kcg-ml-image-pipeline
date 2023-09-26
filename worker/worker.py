
import sys
import time
import requests
import json
from datetime import datetime
import argparse
from PIL import Image

base_directory = "./"
sys.path.insert(0, base_directory)

from worker.image_generation.generation_task.icon_generation_task import IconGenerationTask
from worker.image_generation.generation_task.image_generation_task import ImageGenerationTask
from worker.image_generation.scripts.inpaint_A1111 import img2img, get_model
from stable_diffusion import StableDiffusion, CLIPTextEmbedder
from configs.model_config import ModelPathConfig
from stable_diffusion.model_paths import (SDconfigs, CLIPconfigs)

SERVER_ADRESS = 'http://192.168.3.1:8111'

class WorkerState:
    def __init__(self, device):

        self.device = device
        self.config = ModelPathConfig()
        self.stable_diffusion = None
        self.clip_text_embedder = None

    def load_models(self, model_path='input/model/sd/v1-5-pruned-emaonly/v1-5-pruned-emaonly.safetensors'):
        # NOTE: Initializing stable diffusion
        self.stable_diffusion = StableDiffusion(device=self.device)

        self.stable_diffusion.quick_initialize().load_autoencoder(self.config.get_model(SDconfigs.VAE)).load_decoder(
            self.config.get_model(SDconfigs.VAE_DECODER))
        self.stable_diffusion.model.load_unet(self.config.get_model(SDconfigs.UNET))
        self.stable_diffusion.initialize_latent_diffusion(path=model_path, force_submodels_init=True)

        self.clip_text_embedder = CLIPTextEmbedder(device=self.device)

        self.clip_text_embedder.load_submodels(
            tokenizer_path=self.config.get_model_folder_path(CLIPconfigs.TXT_EMB_TOKENIZER),
            transformer_path=self.config.get_model_folder_path(CLIPconfigs.TXT_EMB_TEXT_MODEL)
        )


def run_generation_task(worker_state, generation_task):

    # Make a cache for these images
    # Check if they changed on disk maybe and reload
    init_image = Image.open(generation_task.init_img)
    init_mask = Image.open(generation_task.init_mask)


    img2img(prompt=generation_task.positive_prompt,
            negative_prompt=generation_task.negative_prompt,
            sampler_name=generation_task.sampler,
            batch_size=1,
            n_iter=generation_task.num_images,
            steps=generation_task.steps,
            cfg_scale=generation_task.cfg_strength,
            width=generation_task.image_width,
            height=generation_task.image_height,
            mask_blur=generation_task.mask_blur,
            inpainting_fill=generation_task.inpainting_fill,
            outpath=generation_task.output_path,
            styles=generation_task.styles,
            init_images=[init_image],
            mask=init_mask,
            resize_mode=generation_task.resize_mode,
            denoising_strength=generation_task.denoising_strength,
            image_cfg_scale=generation_task.image_cfg_scale,
            inpaint_full_res_padding=generation_task.inpaint_full_res_padding,
            inpainting_mask_invert=generation_task.inpainting_mask_invert,
            sd=worker_state.stable_diffusion,
            model=worker_state.stable_diffusion.model,
            clip_text_embedder=worker_state.clip_text_embedder,
            device=worker_state.device
            )

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
    print(url)
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data

    response = requests.put(url, json=job, headers=headers)
    print(response)
    if response.status_code != 200:
        print(f"request failed with status code: {response.status_code}")

def parse_args():
    parser = argparse.ArgumentParser(description="Worker for image generation")

    # Required parameters
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()

def main():
    args = parse_args()


    # Initialize worker state
    worker_state = WorkerState(args.device)
    worker_state.load_models()

    print("starting")

    # for debugging purpose only

    job = {
        "uuid": '1',
        "task_type": "icon_generation_task",
        "model_name" : "v1-5-pruned-emaonly",
        "model_file_name": "v1-5-pruned-emaonly",
        "model_file_path": "input/model/sd/v1-5-pruned-emaonly/v1-5-pruned-emaonly.safetensors",
        "sd_model_hash": "N/A",
        "task_creation_time": "N/A",
        "task_start_time": "N/A",
        "task_completion_time": "N/A",
        "task_error_str": "",
        "task_input_dict": {
            'positive_prompt': "icon, game icon, crystal, high resolution, contour, game icon, jewels, minerals, stones, gems, flat, vector art, game art, stylized, cell shaded, 8bit, 16bit, retro, russian futurism",
            'negative_prompt' : "low resolution, mediocre style, normal resolution",
            'cfg_strength': 12,
            'seed': '',
            'output_path': "./output/inpainting/",
            'num_images': 1,
            'image_width': 512,
            'image_height': 512,
            'checkpoint_path': 'input/model/sd/v1-5-pruned-emaonly/v1-5-pruned-emaonly.safetensors',
            'device': "cuda",
            'sampler': "ddim",
            'sampler_steps': 20,
            'init_img': './test/test_inpainting/white_512x512.jpg',
            'init_mask': './test/test_inpainting/icon_mask.png',

            'mask_blur' : 0,
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
            job['task_start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Convert the job into a dictionary
            # Then use the dictionary to create the generation task
            task = {
                'generation_task_type' : job['task_type'],
                'positive_prompt': job['task_input_dict']['positive_prompt'],
                'negative_prompt': job['task_input_dict']['negative_prompt'],
                'model_name': job['model_name'],
                'cfg_strength': job['task_input_dict']['cfg_strength'],
                'num_images': job['task_input_dict']['num_images'],
                'seed': job['task_input_dict']['seed'],
                'output_path': job['task_input_dict']['output_path'],
                'image_width': job['task_input_dict']['image_width'],
                'image_height': job['task_input_dict']['image_height'],
                'batch_size': 1,
                'checkpoint_path': job['task_input_dict']['checkpoint_path'],
                'device': args.device,
                'sampler': job['task_input_dict']['sampler'],
                'steps': job['task_input_dict']['sampler_steps'],
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
                    run_generation_task(worker_state, generation_task)
                    print("job completed !")
                    job['task_completion_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    http_update_job_completed(job)
                except Exception as e:
                    print(f"generation task failed: {e}")
                    job['task_error_str'] = str(e)
                    http_update_job_failed(job)


            elif task_type == 'image_generation_task':
                generation_task = ImageGenerationTask.from_dict(task)
                # Run inpainting task
                try:
                    run_generation_task(worker_state, generation_task)
                    print("job completed !")
                    job['task_completion_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    http_update_job_completed(job)
                except Exception as e:
                    print(f"generation task failed: {e}")
                    job['task_error_str'] = str(e)
                    http_update_job_failed(job)

        else:
            # If there was no job, go to sleep for a while
            sleep_time_in_seconds = 10
            print("Did not find job, going to sleep for ", sleep_time_in_seconds, " seconds")
            time.sleep(sleep_time_in_seconds)




if __name__ == '__main__':
    main()
