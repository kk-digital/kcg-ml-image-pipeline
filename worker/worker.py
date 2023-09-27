
import sys
import time
import requests
import random
from datetime import datetime
import argparse
from PIL import Image
import hashlib
from termcolor import colored

base_directory = "./"
sys.path.insert(0, base_directory)

from worker.image_generation.generation_task.icon_generation_task import IconGenerationTask
from worker.image_generation.generation_task.image_generation_task import ImageGenerationTask
from worker.image_generation.scripts.inpaint_A1111 import img2img
from stable_diffusion import StableDiffusion, CLIPTextEmbedder
from configs.model_config import ModelPathConfig
from stable_diffusion.model_paths import (SDconfigs, CLIPconfigs)
from worker.image_generation.scripts.stable_diffusion_base_script import StableDiffusionBaseScript
from worker.image_generation.scripts.generate_image_from_text import generate_image_from_text

SERVER_ADRESS = 'http://192.168.3.1:8111'


def info(message):
    print(colored("[INFO] ", 'green') + message)

def error(message):
    print(colored("[ERROR] ", 'red') + message)

def warning(message):
    print(colored("[WARNING] ", 'yellow') + message)


class WorkerState:
    def __init__(self, device):

        self.device = device
        self.config = ModelPathConfig()
        self.stable_diffusion = None
        self.clip_text_embedder = None
        self.txt2img = None

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

        # Starts the text2img
        self.txt2img = StableDiffusionBaseScript(
            sampler_name="ddim",
            n_steps=20,
            force_cpu=False,
            cuda_device=self.device,
        )
        self.txt2img.initialize_latent_diffusion(autoencoder=None, clip_text_embedder=None, unet_model=None,
                                            path=model_path, force_submodels_init=True)


def compute_file_hash(file_path, hash_algorithm='sha256'):
    """Compute the hash of a file using the given algorithm (default: sha256)"""

    # Create a hash object
    h = hashlib.new(hash_algorithm)

    # Open the file in binary read mode
    with open(file_path, 'rb') as f:
        # Read the file in chunks (useful for large files)
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)

    # Return the hexadecimal representation of the hash
    return h.hexdigest()

def run_image_generation_task(worker_state, generation_task):

    # Random seed for now
    # Should we use the seed from job parameters ?
    random.seed(time.time())
    seed = random.randint(0, 2 ** 24 - 1)

    output_file_path = generate_image_from_text(worker_state.txt2img,
                             worker_state.clip_text_embedder,
                             generation_task.positive_prompt,
                             generation_task.negative_prompt,
                             generation_task.cfg_strength,
                             seed,
                             generation_task.image_width,
                             generation_task.image_height,
                             generation_task.output_path)

    output_file_hash = compute_file_hash(output_file_path)

    return output_file_path, output_file_hash



def run_inpainting_generation_task(worker_state, generation_task):

    # TODO(): Make a cache for these images
    # Check if they changed on disk maybe and reload
    init_image = Image.open(generation_task.init_img)
    init_mask = Image.open(generation_task.init_mask)

    output_file_path = img2img(
            prompt=generation_task.positive_prompt,
            negative_prompt=generation_task.negative_prompt,
            sampler_name=generation_task.sampler,
            batch_size=1,
            n_iter=1,
            steps=generation_task.steps,
            cfg_scale=generation_task.cfg_strength,
            width=generation_task.image_width,
            height=generation_task.image_height,
            mask_blur=generation_task.mask_blur,
            inpainting_fill=generation_task.inpainting_fill_mode,
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

    output_file_hash = compute_file_hash(output_file_path)

    return output_file_path, output_file_hash

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
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data

    response = requests.put(url, json=job, headers=headers)

    if response.status_code != 200:
        print(f"request failed with status code: {response.status_code}")

def http_update_job_failed(job):
    url = SERVER_ADRESS + "/update-job-failed"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data

    response = requests.put(url, json=job, headers=headers)
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
    # Loading models
    worker_state.load_models()

    info("starting worker ! ")

    # For debug purpose only
    # TODO(): delete
    inpainting_job = {
        "task_type": "inpainting_generation_task",
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
            'image_width': 512,
            'image_height': 512,
            'sampler': "ddim",
            'sampler_steps': 20,
            'init_img': './test/test_inpainting/white_512x512.jpg',
            'init_mask': './test/test_inpainting/icon_mask.png',

            'mask_blur' : 0,
            'inpainting_fill_mode': 1,
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

    # For debug purpose only
    # TODO(): delete
    image_generation_job = {
        "task_type": "image_generation_task",
        "model_name": "v1-5-pruned-emaonly",
        "model_file_name": "v1-5-pruned-emaonly",
        "model_file_path": "input/model/sd/v1-5-pruned-emaonly/v1-5-pruned-emaonly.safetensors",
        "sd_model_hash": "N/A",
        "task_creation_time": "N/A",
        "task_start_time": "N/A",
        "task_completion_time": "N/A",
        "task_error_str": "",
        "task_input_dict": {
            'positive_prompt': "icon, game icon, crystal, high resolution, contour, game icon, jewels, minerals, stones, gems, flat, vector art, game art, stylized, cell shaded, 8bit, 16bit, retro, russian futurism",
            'negative_prompt': "low resolution, mediocre style, normal resolution",
            'cfg_strength': 12,
            'seed': '',
            'output_path': "./output/",
            'num_images': 1,
            'image_width': 512,
            'image_height': 512,
            'sampler': "ddim",
            'sampler_steps': 20,
        },
        "task_input_file_dict": {},
        "task_output_file_dict": {},
    }

    invalid_job = {
        "task_type": "invalid_job_type",
    }

    # For debug purpose only
    # TODO(): delete
    http_add_job(image_generation_job)
    http_add_job(image_generation_job)
    http_add_job(inpainting_job)
    http_add_job(invalid_job)

    last_job_time = time.time()

    while True:
        info("Looking for jobs ! ")
        job = http_get_job()
        if job != None:
            task_type = job['task_type']

            info("Found job ! " + task_type)
            job_start_time = time.time()
            worker_idle_time = job_start_time - last_job_time
            info(f"worker idle time was {worker_idle_time} seconds.")

            job['task_start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            if task_type == 'inpainting_generation_task':

                # Convert the job into a dictionary
                # Then use the dictionary to create the generation task
                task = {
                    'generation_task_type': job['task_type'],
                    'positive_prompt': job['task_input_dict']['positive_prompt'],
                    'negative_prompt': job['task_input_dict']['negative_prompt'],
                    'model_name': job['model_name'],
                    'cfg_strength': job['task_input_dict']['cfg_strength'],
                    'seed': job['task_input_dict']['seed'],
                    'output_path': job['task_input_dict']['output_path'],
                    'image_width': job['task_input_dict']['image_width'],
                    'image_height': job['task_input_dict']['image_height'],
                    'batch_size': 1,
                    'sampler': job['task_input_dict']['sampler'],
                    'steps': job['task_input_dict']['sampler_steps'],
                    'init_img': job['task_input_dict']['init_img'],
                    'init_mask': job['task_input_dict']['init_mask'],

                    'mask_blur': job['task_input_dict']['mask_blur'],
                    'inpainting_fill_mode': job['task_input_dict']['inpainting_fill_mode'],
                    'styles': job['task_input_dict']['styles'],
                    'resize_mode': job['task_input_dict']['resize_mode'],
                    'denoising_strength': job['task_input_dict']['denoising_strength'],
                    'image_cfg_scale': job['task_input_dict']['image_cfg_scale'],
                    'inpaint_full_res_padding': job['task_input_dict']['inpaint_full_res_padding'],
                    'inpainting_mask_invert': job['task_input_dict']['inpainting_mask_invert']
                }

                generation_task = IconGenerationTask.from_dict(task)

                # Run inpainting task
                try:
                    output_file_path, output_file_hash = run_inpainting_generation_task(worker_state, generation_task)
                    info("job completed !")
                    job['task_completion_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    job['task_output_file_dict'] = {
                        'output_file_path': output_file_path,
                        'output_file_hash': output_file_hash
                    }
                    info("output file path : " + output_file_path)
                    info("output file hash : " + output_file_hash)
                    http_update_job_completed(job)
                except Exception as e:
                    error(f"generation task failed: {e}")
                    job['task_error_str'] = str(e)
                    http_update_job_failed(job)


            elif task_type == 'image_generation_task':

                # Convert the job into a dictionary
                # Then use the dictionary to create the generation task
                task = {
                    'generation_task_type': job['task_type'],
                    'positive_prompt': job['task_input_dict']['positive_prompt'],
                    'negative_prompt': job['task_input_dict']['negative_prompt'],
                    'model_name': job['model_name'],
                    'cfg_strength': job['task_input_dict']['cfg_strength'],
                    'seed': job['task_input_dict']['seed'],
                    'output_path': job['task_input_dict']['output_path'],
                    'image_width': job['task_input_dict']['image_width'],
                    'image_height': job['task_input_dict']['image_height'],
                    'batch_size': 1,
                    'sampler': job['task_input_dict']['sampler'],
                    'steps': job['task_input_dict']['sampler_steps'],
                }

                generation_task = ImageGenerationTask.from_dict(task)
                # Run inpainting task
                try:
                    output_file_path, output_file_hash = run_image_generation_task(worker_state, generation_task)
                    info("job completed !")
                    job['task_completion_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    job['task_output_file_dict'] = {
                        'output_file_path' : output_file_path,
                        'output_file_hash' : output_file_hash
                    }
                    info("output file path : " + output_file_path)
                    info("output file hash : " + output_file_hash)
                    info("job completed !")
                    http_update_job_completed(job)
                except Exception as e:
                    error(f"generation task failed: {e}")
                    job['task_error_str'] = str(e)
                    http_update_job_failed(job)

            else:
                e = "job with task type '" + task_type + "' is not supported"
                error(e)
                job['task_error_str'] = e
                http_update_job_failed(job)

            job_end_time = time.time()
            last_job_time = job_end_time
            job_elapsed_time = job_end_time - job_start_time
            info(f"job took {job_elapsed_time} seconds to execute.")

        else:
            # If there was no job, go to sleep for a while
            sleep_time_in_seconds = 10
            info("Did not find job, going to sleep for " + str(sleep_time_in_seconds) + " seconds")
            time.sleep(sleep_time_in_seconds)


if __name__ == '__main__':
    main()
