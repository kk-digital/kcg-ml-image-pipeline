import hashlib
import io
import sys
import time
import random
from datetime import datetime
import argparse
from PIL import Image
import numpy as np
from termcolor import colored
import os
import threading
import traceback

base_directory = "./"
sys.path.insert(0, base_directory)

from kandinsky_worker.worker_state import WorkerState
from utility.http import generation_request
from utility.path import separate_bucket_and_file_path
from utility.minio import cmd
from kandinsky.utils_image import  save_image_data_to_minio, save_latent_to_minio, save_image_embedding_to_minio, \
    get_embeddings, save_img2img_data_to_minio
from worker.clip_calculation.clip_calculator import run_clip_calculation_task
from worker.generation_task.generation_task import GenerationTask
from kandinsky.models.kandisky import KandinskyPipeline
from utility.utils_logger import logger
from data_loader.utils import get_object
from kandinsky_worker.dataloaders.image_embedding import ImageEmbedding

class ThreadState:
    def __init__(self, thread_id, thread_name):
        self.thread_id = thread_id
        self.thread_name = thread_name


def info(thread_state, message):
    print(colored(f"Thread [{thread_state.thread_id}] {thread_state.thread_name}", 'green') + " " + colored("[INFO] ",
                                                                                                            'green') + message)


def info_v2(message):
    print(colored("[INFO] ", 'green') + message)


def error(thread_state, message):
    print(colored(f"Thread [{thread_state.thread_id}] {thread_state.thread_name}", 'green') + " " + colored("[ERROR] ",
                                                                                                            'red') + message)


def warning(thread_state, message):
    print(
        colored(f"Thread [{thread_state.thread_id}] {thread_state.thread_name}", 'green') + " " + colored("[WARNING] ",
                                                                                                          'yellow') + message)


def run_image_generation_task(worker_state, generation_task):
    # Random seed for now
    # Should we use the seed from job parameters ?
    random.seed(time.time())
    seed = random.randint(0, 2 ** 24 - 1)

    generation_task.task_input_dict["seed"] = seed
    positive_prompt = generation_task.task_input_dict["positive_prompt"]
    negative_decoder_prompt = generation_task.task_input_dict["negative_decoder_prompt"]
    negative_prior_prompt = generation_task.task_input_dict["negative_prior_prompt"]

    image_width = generation_task.task_input_dict["image_width"]
    image_height = generation_task.task_input_dict["image_height"]
    strength = generation_task.task_input_dict["strength"]
    decoder_steps = generation_task.task_input_dict["decoder_steps"]
    prior_steps = generation_task.task_input_dict["prior_steps"]
    prior_guidance_scale = generation_task.task_input_dict["prior_guidance_scale"]
    decoder_guidance_scale = generation_task.task_input_dict["decoder_guidance_scale"]
    dataset = generation_task.task_input_dict["dataset"]

    image_encoder=worker_state.clip.vision_model
    unet = worker_state.unet
    prior_model = worker_state.prior_model
    decoder_model = worker_state.decoder_model

    text2img_processor = KandinskyPipeline(
        device=worker_state.device,
        width= image_width,
        height= image_height,
        batch_size=1,
        decoder_steps= decoder_steps,
        prior_steps= prior_steps,
        strength= strength,
        prior_guidance_scale= prior_guidance_scale,
        decoder_guidance_scale= decoder_guidance_scale
    )

    text2img_processor.set_models(
        image_encoder= image_encoder,
        unet=unet,
        prior_model= prior_model,
        decoder_model= decoder_model
    )
    
    # generate image
    image, latents = text2img_processor.generate_text2img(prompt=positive_prompt, 
                                               negative_prior_prompt=negative_prior_prompt, 
                                               negative_decoder_prompt=negative_decoder_prompt,
                                               seed=seed)

    # convert image to png from RGB
    output_file_hash, img_byte_arr = text2img_processor.convert_image_to_png(image)
    
    output_file_path = os.path.join("datasets", dataset, generation_task.task_input_dict['file_path'])

    # Return the latent vector along with other values
    return output_file_path, output_file_hash, img_byte_arr, latents, seed


def run_inpainting_generation_task(worker_state, generation_task: GenerationTask):
    # TODO(): Make a cache for these images
    # Check if they changed on disk maybe and reload

    random.seed(time.time())
    seed = random.randint(0, 2 ** 24 - 1)

    generation_task.task_input_dict["seed"] = seed
    init_image = Image.open(generation_task.task_input_dict["init_img"])
    init_mask = Image.open(generation_task.task_input_dict["init_mask"])

    positive_prompt = generation_task.task_input_dict["positive_prompt"]
    negative_decoder_prompt = generation_task.task_input_dict["negative_decoder_prompt"]
    negative_prior_prompt = generation_task.task_input_dict["negative_prior_prompt"]

    image_width = generation_task.task_input_dict["image_width"]
    image_height = generation_task.task_input_dict["image_height"]
    strength = generation_task.task_input_dict["strength"]
    decoder_steps = generation_task.task_input_dict["decoder_steps"]
    prior_steps = generation_task.task_input_dict["prior_steps"]
    prior_guidance_scale = generation_task.task_input_dict["prior_guidance_scale"]
    decoder_guidance_scale = generation_task.task_input_dict["decoder_guidance_scale"]
    dataset = generation_task.task_input_dict["dataset"]

    image_encoder=worker_state.clip.vision_model
    unet = worker_state.unet
    prior_model = worker_state.prior_model
    decoder_model = worker_state.inpainting_decoder_model

    inpainting_processor = KandinskyPipeline(
        device=worker_state.device,
        width= image_width,
        height= image_height,
        batch_size=1,
        decoder_steps= decoder_steps,
        prior_steps= prior_steps,
        strength= strength,
        prior_guidance_scale= prior_guidance_scale,
        decoder_guidance_scale= decoder_guidance_scale
    )

    inpainting_processor.set_models(
        image_encoder= image_encoder,
        unet=unet,
        prior_model= prior_model,
        decoder_model= decoder_model
    )
    
    # generate image
    image, latents = inpainting_processor.generate_inpainting(prompt=positive_prompt, 
                                               negative_prior_prompt=negative_prior_prompt, 
                                               negative_decoder_prompt=negative_decoder_prompt, 
                                               initial_image=init_image,
                                               img_mask=init_mask,
                                               seed=seed)

    # convert image to png from RGB
    output_file_hash, img_byte_arr = inpainting_processor.convert_image_to_png(image)
    
    output_file_path = os.path.join("datasets", dataset, generation_task.task_input_dict['file_path'])

    # Return the latent vector along with other values
    return output_file_path, output_file_hash, img_byte_arr, latents, seed

def run_img2img_generation_task(worker_state, generation_task: GenerationTask):
    # TODO(): Make a cache for these images
    # Check if they changed on disk maybe and reload
    random.seed(time.time())
    seed = random.randint(0, 2 ** 24 - 1)

    generation_task.task_input_dict["seed"] = seed
    init_image = Image.open(generation_task.task_input_dict["init_img"])

    image_width = generation_task.task_input_dict["image_width"]
    image_height = generation_task.task_input_dict["image_height"]
    strength = generation_task.task_input_dict["strength"]
    decoder_steps = generation_task.task_input_dict["decoder_steps"]
    decoder_guidance_scale = generation_task.task_input_dict["decoder_guidance_scale"]
    dataset = generation_task.task_input_dict["dataset"]

    image_encoder=worker_state.clip.vision_model
    decoder_model = worker_state.img2img_decoder

    img2img_processor = KandinskyPipeline(
        device=worker_state.device,
        width= image_width,
        height= image_height,
        batch_size=1,
        decoder_steps= decoder_steps,
        strength= strength,
        decoder_guidance_scale= decoder_guidance_scale
    )

    img2img_processor.set_models(
        unet=None,
        prior_model=None,
        image_encoder= image_encoder,
        decoder_model= decoder_model
    )
    
    # get the input image embeddings from minIO
    output_file_path = os.path.join(dataset, generation_task.task_input_dict['file_path'])
    image_embeddings_path = output_file_path.replace(".jpg", "_embedding.msgpack")    
    embedding_data = get_object(worker_state.minio_client, image_embeddings_path)
    embedding_dict = ImageEmbedding.from_msgpack_bytes(embedding_data)
    image_embedding= embedding_dict.image_embedding.to(worker_state.device)
    negative_image_embedding= embedding_dict.negative_image_embedding
    if negative_image_embedding is not None:
        negative_image_embedding= negative_image_embedding.to(worker_state.device)

    # generate image
    image, latents = img2img_processor.generate_img2img(init_img=init_image,
                                                        image_embeds= image_embedding,
                                                        negative_image_embeds= negative_image_embedding,
                                                        seed=seed)

    output_file_path = os.path.join("datasets", output_file_path)
    # convert image to png from RGB
    output_file_hash, img_byte_arr = img2img_processor.convert_image_to_png(image)

    # Return the latent vector along with other values
    return output_file_path, output_file_hash, img_byte_arr, latents, seed

def parse_args():
    parser = argparse.ArgumentParser(description="Worker for image generation")

    # Required parameters
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--queue_size", type=int, default=8)
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
            job = generation_request.http_get_job(model_type="kandinsky")
        else:
            job = generation_request.http_get_job(worker_type, model_type="kandinsky")

        if job is not None:
            break

    return job


def upload_data_and_update_job_status(job, output_file_path, output_file_hash, data, minio_client):
    start_time = time.time()
    bucket_name, file_path = separate_bucket_and_file_path(output_file_path)

    cmd.upload_data(minio_client, bucket_name, file_path, data)

    info_v2("Upload for job {} completed".format(job["uuid"]))
    info_v2("Upload time elapsed: {:.4f}s".format(time.time() - start_time))

    # update job info
    job['task_completion_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    job['task_output_file_dict'] = {
        'output_file_path': output_file_path,
        'output_file_hash': output_file_hash
    }
    info_v2("output file path: " + output_file_path)
    info_v2("output file hash: " + output_file_hash)
    info_v2("job completed: " + job["uuid"])

    # update status
    generation_request.http_update_job_completed(job)


def upload_image_data_and_update_job_status_img2img(worker_state,
                                            job,
                                            generation_task,
                                            seed,
                                            latent,
                                            output_file_path,
                                            output_file_hash,
                                            job_completion_time,
                                            data):
    start_time = time.time()
    bucket_name, file_path = separate_bucket_and_file_path(output_file_path)

    minio_client = worker_state.minio_client

    image_width = generation_task.task_input_dict["image_width"]
    image_height = generation_task.task_input_dict["image_height"]
    strength = generation_task.task_input_dict["strength"]
    decoder_steps = generation_task.task_input_dict["decoder_steps"]
    decoder_guidance_scale = generation_task.task_input_dict["decoder_guidance_scale"]
    dataset = generation_task.task_input_dict["dataset"]

    cmd.upload_data(minio_client, bucket_name, file_path, data)

    # save image meta data
    save_img2img_data_to_minio(minio_client,
                             generation_task.uuid,
                             job_completion_time,
                             dataset,
                             output_file_path,
                             output_file_hash,
                             seed,
                             image_width,
                             image_height,
                             strength,
                             decoder_steps,
                             decoder_guidance_scale)

    save_latent_to_minio(minio_client, 
                         bucket_name, 
                         generation_task.uuid, 
                         output_file_hash, 
                         latent, 
                         output_file_path)

    info_v2("Upload for job {} completed".format(generation_task.uuid))
    info_v2("Upload time elapsed: {:.4f}s".format(time.time() - start_time))

    # update job info
    job['task_completion_time'] = job_completion_time
    job['task_output_file_dict'] = {
        'output_file_path': output_file_path,
        'output_file_hash': output_file_hash
    }
    info_v2("output file path: " + output_file_path)
    info_v2("output file hash: " + output_file_hash)
    info_v2("job completed: " + generation_task.uuid)

    # update status
    generation_request.http_update_job_completed(job)

    # add clip calculation tasks
    kandinsky_clip_calculation_job = {"uuid": "",
                            "task_type": "clip_calculation_task_kandinsky",
                            "task_input_dict": {
                                "input_file_path": output_file_path,
                                "input_file_hash": output_file_hash
                            },
                            }
    
    sd_clip_calculation_job = {"uuid": "",
                            "task_type": "clip_calculation_task_sd_1_5",
                            "task_input_dict": {
                                "input_file_path": output_file_path,
                                "input_file_hash": output_file_hash
                            },
                            }

    generation_request.http_add_job(kandinsky_clip_calculation_job)
    generation_request.http_add_job(sd_clip_calculation_job)

def upload_image_data_and_update_job_status(worker_state,
                                            job,
                                            generation_task,
                                            seed,
                                            latent,
                                            output_file_path,
                                            output_file_hash,
                                            job_completion_time,
                                            data,
                                            prompt_embedding,
                                            prompt_embedding_average_pooled,
                                            prompt_embedding_max_pooled,
                                            prompt_embedding_signed_max_pooled):
    start_time = time.time()
    bucket_name, file_path = separate_bucket_and_file_path(output_file_path)

    minio_client = worker_state.minio_client

    positive_prompts = generation_task.task_input_dict["positive_prompt"]
    negative_prior_prompt = generation_task.task_input_dict["negative_prior_prompt"]
    negative_decoder_prompt = generation_task.task_input_dict["negative_decoder_prompt"]

    image_width = generation_task.task_input_dict["image_width"]
    image_height = generation_task.task_input_dict["image_height"]
    strength = generation_task.task_input_dict["strength"]
    decoder_steps = generation_task.task_input_dict["decoder_steps"]
    prior_steps = generation_task.task_input_dict["prior_steps"]
    prior_guidance_scale = generation_task.task_input_dict["prior_guidance_scale"]
    decoder_guidance_scale = generation_task.task_input_dict["decoder_guidance_scale"]
    dataset = generation_task.task_input_dict["dataset"]

    prompt_scoring_model = generation_task.prompt_generation_data["prompt_scoring_model"]
    prompt_score = generation_task.prompt_generation_data["prompt_score"]
    prompt_generation_policy = generation_task.prompt_generation_data["prompt_generation_policy"]
    top_k = generation_task.prompt_generation_data["top_k"]

    cmd.upload_data(minio_client, bucket_name, file_path, data)

    # save image meta data
    save_image_data_to_minio(minio_client,
                             generation_task.uuid,
                             job_completion_time,
                             dataset,
                             output_file_path,
                             output_file_hash,
                             positive_prompts,
                             negative_prior_prompt,
                             negative_decoder_prompt,
                             seed,
                             image_width,
                             image_height,
                             strength,
                             decoder_steps,
                             prior_steps,
                             prior_guidance_scale,
                             decoder_guidance_scale,
                             prompt_scoring_model,
                             prompt_score,
                             prompt_generation_policy,
                             top_k)

    save_latent_to_minio(minio_client, 
                         bucket_name, 
                         generation_task.uuid, 
                         output_file_hash, 
                         latent, 
                         output_file_path)
    
    # save image embedding data
    save_image_embedding_to_minio(minio_client,
                                  dataset,
                                  output_file_path,
                                  prompt_embedding,
                                  prompt_embedding_average_pooled,
                                  prompt_embedding_max_pooled,
                                  prompt_embedding_signed_max_pooled)

    info_v2("Upload for job {} completed".format(generation_task.uuid))
    info_v2("Upload time elapsed: {:.4f}s".format(time.time() - start_time))

    # update job info
    job['task_completion_time'] = job_completion_time
    job['task_output_file_dict'] = {
        'output_file_path': output_file_path,
        'output_file_hash': output_file_hash
    }
    info_v2("output file path: " + output_file_path)
    info_v2("output file hash: " + output_file_hash)
    info_v2("job completed: " + generation_task.uuid)

    # update status
    generation_request.http_update_job_completed(job)

    # add clip calculation tasks
    kandinsky_clip_calculation_job = {"uuid": "",
                            "task_type": "clip_calculation_task_kandinsky",
                            "task_input_dict": {
                                "input_file_path": output_file_path,
                                "input_file_hash": output_file_hash
                            },
                            }
    
    sd_clip_calculation_job = {"uuid": "",
                            "task_type": "clip_calculation_task_sd_1_5",
                            "task_input_dict": {
                                "input_file_path": output_file_path,
                                "input_file_hash": output_file_hash
                            },
                            }

    generation_request.http_add_job(kandinsky_clip_calculation_job)
    generation_request.http_add_job(sd_clip_calculation_job)


def process_jobs(worker_state):
    thread_state = ThreadState(1, "Job Processor")
    last_job_time = time.time()

    while True:
        job = worker_state.job_queue.get()

        if job is not None:
            task_type = job['task_type']

            print('\n\n')
            info(thread_state, "Processing job: " + task_type)
            info(thread_state, 'Queue size ' + str(worker_state.job_queue.qsize()))
            job_start_time = time.time()
            worker_idle_time = job_start_time - last_job_time
            info(thread_state, f"worker idle time was {worker_idle_time:.4f} seconds.")

            job['task_start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            generation_task = GenerationTask.from_dict(job)

            try:
                if task_type == 'img2img_generation_kandinsky':
                    output_file_path, output_file_hash, img_data, latent, seed = run_img2img_generation_task(worker_state,
                                                                                                   generation_task)
                    
                    job_completion_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    # spawn upload data and update job thread
                    thread = threading.Thread(target=upload_image_data_and_update_job_status_img2img, args=(
                        worker_state, job, generation_task, seed, latent, output_file_path, output_file_hash, job_completion_time, img_data))
                    thread.start()

                elif task_type == 'inpainting_kandinsky':
                    output_file_path, output_file_hash, img_data, inpainting_latent, seed = run_inpainting_generation_task(worker_state,
                                                                                                  generation_task)

                    job_completion_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    (prompt_embedding,
                     prompt_embedding_average_pooled,
                     prompt_embedding_max_pooled,
                     prompt_embedding_signed_max_pooled) = get_embeddings(generation_task.uuid,
                                                                          job_completion_time,
                                                                          generation_task.task_input_dict["dataset"],
                                                                          output_file_path,
                                                                          output_file_hash,
                                                                          generation_task.task_input_dict[
                                                                              "positive_prompt"],
                                                                          generation_task.task_input_dict[
                                                                              "negative_prior_prompt"],
                                                                          generation_task.task_input_dict[
                                                                              "negative_decoder_prompt"],
                                                                          worker_state.clip_text_embedder)
                    # spawn upload data and update job thread
                    thread = threading.Thread(target=upload_image_data_and_update_job_status, args=(
                        worker_state, job, generation_task, seed, inpainting_latent, output_file_path, output_file_hash, job_completion_time,
                        img_data, prompt_embedding, prompt_embedding_average_pooled, prompt_embedding_max_pooled,
                        prompt_embedding_signed_max_pooled,))
                    thread.start()

                elif task_type == 'image_generation_kandinsky':
                    output_file_path, output_file_hash, img_data, latent, seed = run_image_generation_task(worker_state,
                                                                                                   generation_task)

                    job_completion_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    (prompt_embedding,
                     prompt_embedding_average_pooled,
                     prompt_embedding_max_pooled,
                     prompt_embedding_signed_max_pooled) = get_embeddings(generation_task.uuid,
                                                                          job_completion_time,
                                                                          generation_task.task_input_dict["dataset"],
                                                                          output_file_path,
                                                                          output_file_hash,
                                                                          generation_task.task_input_dict[
                                                                              "positive_prompt"],
                                                                          generation_task.task_input_dict[
                                                                              "negative_prior_prompt"],
                                                                          generation_task.task_input_dict[
                                                                              "negative_decoder_prompt"],
                                                                          worker_state.clip_text_embedder)

                    # spawn upload data and update job thread
                    thread = threading.Thread(target=upload_image_data_and_update_job_status, args=(
                        worker_state, job, generation_task, seed, latent, output_file_path, output_file_hash,
                        job_completion_time, img_data, prompt_embedding, prompt_embedding_average_pooled,
                        prompt_embedding_max_pooled, prompt_embedding_signed_max_pooled,))
                    thread.start()

                elif task_type == 'clip_calculation_task_kandinsky':
                    output_file_path, output_file_hash, clip_data = run_clip_calculation_task(worker_state,
                                                                                              generation_task,
                                                                                              model_type="kandinsky")

                    # spawn upload data and update job thread
                    thread = threading.Thread(target=upload_data_and_update_job_status, args=(
                        job, output_file_path, output_file_hash, clip_data, worker_state.minio_client,))
                    thread.start()

                else:
                    e = "job with task type '" + task_type + "' is not supported"
                    error(thread_state, e)
                    job['task_error_str'] = e
                    generation_request.http_update_job_failed(job)
            except Exception as e:
                error(thread_state, f"generation task failed: {traceback.format_exc()}")
                job['task_error_str'] = str(e)
                generation_request.http_update_job_failed(job)

            job_end_time = time.time()
            last_job_time = job_end_time
            job_elapsed_time = job_end_time - job_start_time
            info(thread_state, f"job took {job_elapsed_time:.4f} seconds to execute.")

        else:
            # If there was no job, go to sleep for a while
            sleep_time_in_seconds = 0.001
            time.sleep(sleep_time_in_seconds)


def get_worker_type_list(worker_type: str):
    worker_type = worker_type.strip()  # remove trailing and leading spaces
    worker_type = worker_type.replace(' ', '')  # remove spaces
    worker_type_list = worker_type.split(",")  # split by comma
    return worker_type_list


def main():
    args = parse_args()

    thread_state = ThreadState(0, "Job Fetcher")
    queue_size = args.queue_size

    # get worker type
    worker_type_list = get_worker_type_list(args.worker_type)

    # Initialize worker state
    worker_state = WorkerState(args.device, args.minio_access_key, args.minio_secret_key, queue_size)
    # Loading models
    worker_state.load_models()

    info(thread_state, "starting worker ! ")
    info(thread_state, "Worker type: {} ".format(worker_type_list))

    # spawning worker thread
    thread = threading.Thread(target=process_jobs, args=(worker_state,))
    thread.start()

    while True:
        # if we have more than n jobs in queue
        # sleep for a while
        if worker_state.job_queue.qsize() >= worker_state.queue_size:
            sleep_time_in_seconds = 0.001
            time.sleep(sleep_time_in_seconds)
            continue

        # try to find a job
        # if job exists add it to job queue
        # if not sleep for a while
        job = get_job_if_exist(worker_type_list)
        if job != None:
            info(thread_state, 'Found job ! ')
            worker_state.job_queue.put(job)
            info(thread_state, 'Queue size ' + str(worker_state.job_queue.qsize()))

        else:
            sleep_time_in_seconds = 5
            info(thread_state, "Did not find job, going to sleep for " + f"{sleep_time_in_seconds:.4f}" + " seconds")
            time.sleep(sleep_time_in_seconds)


if __name__ == '__main__':
    main()
