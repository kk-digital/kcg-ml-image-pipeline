import sys
import time
import random
from datetime import datetime
import argparse
from PIL import Image
import hashlib
from termcolor import colored
import os

base_directory = "./"
sys.path.insert(0, base_directory)

from worker.generation_task.generation_task import GenerationTask
from worker.image_generation.scripts.inpaint_A1111 import img2img
from stable_diffusion import StableDiffusion, CLIPTextEmbedder
from configs.model_config import ModelPathConfig
from stable_diffusion.model_paths import (SDconfigs, CLIPconfigs)
from worker.image_generation.scripts.stable_diffusion_base_script import StableDiffusionBaseScript
from worker.image_generation.scripts.generate_image_from_text import generate_image_from_text
from utility.minio import cmd
from worker.http import request
from worker.prompt_generation.prompt_generator import generate_image_generation_jobs_using_generated_prompts, generate_inpainting_generation_jobs_using_generated_prompts

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


def run_image_generation_task(worker_state, generation_task, minio_client):
    # Random seed for now
    # Should we use the seed from job parameters ?
    random.seed(time.time())
    seed = random.randint(0, 2 ** 24 - 1)

    output_file_path, output_file_hash = generate_image_from_text(minio_client,
                                                                  worker_state.txt2img,
                                                                  worker_state.clip_text_embedder,
                                                                  positive_prompts=generation_task.task_input_dict["positive_prompt"],
                                                                  negative_prompts=generation_task.task_input_dict["negative_prompt"],
                                                                  cfg_strength=generation_task.task_input_dict["cfg_strength"],
                                                                  # seed=generation_task.task_input_dict["seed"],
                                                                  seed=seed,
                                                                  image_width=generation_task.task_input_dict["image_width"],
                                                                  image_height=generation_task.task_input_dict["image_height"],
                                                                  output_path=os.path.join("datasets",
                                                                                                generation_task.task_input_dict[
                                                                                                    "dataset"],
                                                                                                generation_task.task_input_dict[
                                                                                                    "file_path"]))

    return output_file_path, output_file_hash


def run_inpainting_generation_task(worker_state, generation_task: GenerationTask, minio_client):
    # TODO(): Make a cache for these images
    # Check if they changed on disk maybe and reload
    init_image = Image.open(generation_task.task_input_dict["init_img"])
    init_mask = Image.open(generation_task.task_input_dict["init_mask"])

    output_file_path, output_file_hash = img2img(
        minio_client=minio_client,
        prompt=generation_task.task_input_dict["positive_prompt"],
        negative_prompt=generation_task.task_input_dict["negative_prompt"],
        sampler_name=generation_task.task_input_dict["sampler"],
        batch_size=1,
        n_iter=1,
        steps=generation_task.task_input_dict["sampler_steps"],
        cfg_scale=generation_task.task_input_dict["cfg_strength"],
        width=generation_task.task_input_dict["image_width"],
        height=generation_task.task_input_dict["image_height"],
        mask_blur=generation_task.task_input_dict["mask_blur"],
        inpainting_fill=generation_task.task_input_dict["inpainting_fill_mode"],
        outpath=os.path.join("datasets", generation_task.task_input_dict['dataset'],
                             generation_task.task_input_dict['file_path']),
        styles=generation_task.task_input_dict["styles"],
        init_images=[init_image],
        mask=init_mask,
        resize_mode=generation_task.task_input_dict["resize_mode"],
        denoising_strength=generation_task.task_input_dict["denoising_strength"],
        image_cfg_scale=generation_task.task_input_dict["image_cfg_scale"],
        inpaint_full_res_padding=generation_task.task_input_dict["inpaint_full_res_padding"],
        inpainting_mask_invert=generation_task.task_input_dict["inpainting_mask_invert"],
        sd=worker_state.stable_diffusion,
        model=worker_state.stable_diffusion.model,
        clip_text_embedder=worker_state.clip_text_embedder,
        device=worker_state.device
    )

    return output_file_path, output_file_hash


def run_generate_image_generation_task(generation_task):
    generate_image_generation_jobs_using_generated_prompts(
        csv_dataset_path=generation_task.task_input_dict["csv_dataset_path"],
        prompt_count=generation_task.task_input_dict["prompt_count"],
        dataset_name=generation_task.task_input_dict["dataset_name"],
        positive_prefix=generation_task.task_input_dict["positive_prefix"]
    )


def run_generate_inpainting_generation_task(generation_task):
    generate_inpainting_generation_jobs_using_generated_prompts(
        csv_dataset_path=generation_task.task_input_dict["csv_dataset_path"],
        prompt_count=generation_task.task_input_dict["prompt_count"],
        dataset_name=generation_task.task_input_dict["dataset_name"],
        positive_prefix=generation_task.task_input_dict["positive_prefix"],
        init_img_path=generation_task.task_input_dict["init_img_path"],
        mask_path=generation_task.task_input_dict["mask_path"],
    )


def get_minio_client(minio_access_key, minio_secret_key):
    # check first if minio client is available
    minio_client = None
    while minio_client is None:
        # check minio server
        if cmd.is_minio_server_accesssible():
            minio_client = cmd.connect_to_minio_client(minio_access_key, minio_secret_key)
            return minio_client


def parse_args():
    parser = argparse.ArgumentParser(description="Worker for image generation")

    # Required parameters
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--minio-access-key", type=str,
                        help="The minio access key to use so worker can upload files to minio server")
    parser.add_argument("--minio-secret-key", type=str,
                        help="The minio secret key to use so worker can upload files to minio server")
    parser.add_argument("--worker-type", type=str, default="",
                        help="The task types the worker will accept and do. If blank then worker will accept all task types.")

    return parser.parse_args()


def get_job_if_exist(worker_type_list):
    job = None
    for worker_type in worker_type_list:
        if worker_type == "":
            job = request.http_get_job()
        else:
            job = request.http_get_job(worker_type)

        if job is not None:
            break

    return job


def main():
    args = parse_args()

    # get worker type
    worker_type = args.worker_type
    worker_type = worker_type.strip()  # remove trailing and leading spaces
    worker_type = worker_type.replace(' ', '')  # remove spaces
    worker_type_list = worker_type.split(",")  # split by comma

    # Initialize worker state
    worker_state = WorkerState(args.device)
    # Loading models
    worker_state.load_models()

    # get minio client
    minio_client = get_minio_client(args.minio_access_key, args.minio_secret_key)

    info("starting worker ! ")
    info("Worker type: {} ".format(worker_type_list))

    last_job_time = time.time()

    while True:
        info("Looking for jobs")
        job = get_job_if_exist(worker_type_list)

        if job is not None:
            task_type = job['task_type']

            info("Found job: " + task_type)
            job_start_time = time.time()
            worker_idle_time = job_start_time - last_job_time
            info(f"worker idle time was {worker_idle_time:.4f} seconds.")

            job['task_start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            generation_task = GenerationTask.from_dict(job)
            if task_type == 'inpainting_generation_task':
                # Convert the job into a dictionary
                # Then use the dictionary to create the generation task
                try:
                    output_file_path, output_file_hash = run_inpainting_generation_task(worker_state, generation_task,
                                                                                        minio_client)
                    info("job completed !")
                    job['task_completion_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    job['task_output_file_dict'] = {
                        'output_file_path': output_file_path,
                        'output_file_hash': output_file_hash
                    }
                    info("output file path : " + output_file_path)
                    info("output file hash : " + output_file_hash)
                    request.http_update_job_completed(job)
                except Exception as e:
                    error(f"generation task failed: {e}")
                    job['task_error_str'] = str(e)
                    request.http_update_job_failed(job)

            elif task_type == 'image_generation_task':
                try:
                    # Run inpainting task
                    output_file_path, output_file_hash = run_image_generation_task(worker_state, generation_task,
                                                                                   minio_client)
                    info("job completed !")
                    job['task_completion_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    job['task_output_file_dict'] = {
                        'output_file_path': output_file_path,
                        'output_file_hash': output_file_hash
                    }
                    info("output file path : " + output_file_path)
                    info("output file hash : " + output_file_hash)
                    info("job completed")
                    request.http_update_job_completed(job)
                except Exception as e:
                    error(f"generation task failed: {e}")
                    job['task_error_str'] = str(e)
                    request.http_update_job_failed(job)

            elif task_type == "generate_image_generation_task":
                try:
                    # run generate image generation task
                    run_generate_image_generation_task(generation_task)
                    info("job completed !")
                    job['task_completion_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    info("job completed")
                    request.http_update_job_completed(job)

                except Exception as e:
                    error(f"generation task failed: {e}")
                    job['task_error_str'] = str(e)
                    request.http_update_job_failed(job)

            elif task_type == "generate_inpainting_generation_task":
                try:
                    # run generate inpainting generation task
                    run_generate_inpainting_generation_task(generation_task)
                    info("job completed !")
                    job['task_completion_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    info("job completed")
                    request.http_update_job_completed(job)

                except Exception as e:
                    error(f"generation task failed: {e}")
                    job['task_error_str'] = str(e)
                    request.http_update_job_failed(job)

            else:
                e = "job with task type '" + task_type + "' is not supported"
                error(e)
                job['task_error_str'] = e
                request.http_update_job_failed(job)

            job_end_time = time.time()
            last_job_time = job_end_time
            job_elapsed_time = job_end_time - job_start_time
            info(f"job took {job_elapsed_time:.4f} seconds to execute.")

        else:
            # If there was no job, go to sleep for a while
            sleep_time_in_seconds = 10
            info("Did not find job, going to sleep for " + f"{sleep_time_in_seconds:.4f}" + " seconds")
            time.sleep(sleep_time_in_seconds)


if __name__ == '__main__':
    main()