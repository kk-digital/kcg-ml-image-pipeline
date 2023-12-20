import sys
import time
import random
from datetime import datetime
import argparse
from PIL import Image
from termcolor import colored
import os
import threading
import traceback

import torch

base_directory = "./"
sys.path.insert(0, base_directory)

from worker.prompt_generation.prompt_generator import run_generate_inpainting_generation_task, \
    run_generate_image_generation_task
from worker.image_generation.scripts.inpaint_A1111 import img2img
from worker.image_generation.scripts.generate_image_from_text import generate_image_from_text
from worker.worker_state import WorkerState
from utility.http import generation_request
from utility.path import separate_bucket_and_file_path
from utility.minio import cmd
from stable_diffusion.utils_image import save_images_to_minio, save_image_data_to_minio, save_image_embedding_to_minio, \
    get_image_data, get_embeddings
from worker.clip_calculation.clip_calculator import run_clip_calculation_task
from worker.generation_task.generation_task import GenerationTask


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

    output_file_path, output_file_hash, img_data = generate_image_from_text(
        worker_state.minio_client,
        worker_state.txt2img,
        worker_state.clip_text_embedder,
        dataset=generation_task.task_input_dict["dataset"],
        job_uuid=generation_task.uuid,
        sampler=generation_task.task_input_dict["sampler"],
        sampler_steps=generation_task.task_input_dict["sampler_steps"],
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

    return output_file_path, output_file_hash, img_data, seed


def run_inpainting_generation_task(worker_state, generation_task: GenerationTask):
    # TODO(): Make a cache for these images
    # Check if they changed on disk maybe and reload
    init_image = Image.open(generation_task.task_input_dict["init_img"])
    init_mask = Image.open(generation_task.task_input_dict["init_mask"])

    positive_prompts = generation_task.task_input_dict["positive_prompt"]
    negative_prompts = generation_task.task_input_dict["negative_prompt"]

    image_width = generation_task.task_input_dict["image_width"]
    image_height = generation_task.task_input_dict["image_height"]
    cfg_strength = generation_task.task_input_dict["cfg_strength"]
    sampler = generation_task.task_input_dict["sampler"]
    sampler_steps = generation_task.task_input_dict["sampler_steps"]
    dataset = generation_task.task_input_dict["dataset"]

    prompt_scoring_model = generation_task.task_input_dict["prompt_scoring_model"]
    prompt_score = generation_task.task_input_dict["prompt_score"]
    prompt_generation_policy = generation_task.task_input_dict["prompt_generation_policy"]
    top_k = generation_task.task_input_dict["top_k"]

    output_file_path, output_file_hash, img_byte_arr, seed, subseed = img2img(
        prompt=positive_prompts,
        negative_prompt=negative_prompts,
        sampler_name=sampler,
        batch_size=1,
        n_iter=1,
        steps=sampler_steps,
        cfg_scale=cfg_strength,
        width=image_width,
        height=image_height,
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

    generation_task.task_input_dict["seed"] = seed
    generation_task.task_input_dict["subseed"] = subseed

    return output_file_path, output_file_hash, img_byte_arr


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
            job = generation_request.http_get_job()
        else:
            job = generation_request.http_get_job(worker_type)

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


def upload_image_data_and_update_job_status(worker_state,
                                            job,
                                            generation_task,
                                            seed,
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
    negative_prompts = generation_task.task_input_dict["negative_prompt"]

    image_width = generation_task.task_input_dict["image_width"]
    image_height = generation_task.task_input_dict["image_height"]
    cfg_strength = generation_task.task_input_dict["cfg_strength"]
    sampler = generation_task.task_input_dict["sampler"]
    sampler_steps = generation_task.task_input_dict["sampler_steps"]
    dataset = generation_task.task_input_dict["dataset"]

    prompt_scoring_model = generation_task.task_input_dict["prompt_scoring_model"]
    prompt_score = generation_task.task_input_dict["prompt_score"]
    prompt_generation_policy = generation_task.task_input_dict["prompt_generation_policy"]
    top_k = generation_task.task_input_dict["top_k"]

    cmd.upload_data(minio_client, bucket_name, file_path, data)

    # save image meta data
    save_image_data_to_minio(minio_client,
                             generation_task.uuid,
                             job_completion_time,
                             dataset,
                             output_file_path,
                             output_file_hash,
                             positive_prompts,
                             negative_prompts,
                             cfg_strength,
                             seed,
                             image_width,
                             image_height,
                             sampler,
                             sampler_steps,
                             prompt_scoring_model,
                             prompt_score,
                             prompt_generation_policy,
                             top_k)
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

    # add clip calculation task
    clip_calculation_job = {"uuid": "",
                            "task_type": "clip_calculation_task",
                            "task_input_dict": {
                                "input_file_path": output_file_path,
                                "input_file_hash": output_file_hash
                            },
                            }

    generation_request.http_add_job(clip_calculation_job)


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
                if task_type == 'inpainting_generation_task':
                    output_file_path, output_file_hash, img_data = run_inpainting_generation_task(worker_state,
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
                                                                              "negative_prompt"],
                                                                          worker_state.clip_text_embedder)
                    # spawn upload data and update job thread
                    thread = threading.Thread(target=upload_image_data_and_update_job_status, args=(
                        worker_state, job, generation_task, -1, output_file_path, output_file_hash, job_completion_time,
                        img_data, prompt_embedding, prompt_embedding_average_pooled, prompt_embedding_max_pooled,
                        prompt_embedding_signed_max_pooled,))
                    thread.start()

                elif task_type == 'image_generation_task':
                    output_file_path, output_file_hash, img_data, seed = run_image_generation_task(worker_state,
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
                                                                              "negative_prompt"],
                                                                          worker_state.clip_text_embedder)

                    # spawn upload data and update job thread
                    thread = threading.Thread(target=upload_image_data_and_update_job_status, args=(
                        worker_state, job, generation_task, seed, output_file_path, output_file_hash,
                        job_completion_time, img_data, prompt_embedding, prompt_embedding_average_pooled,
                        prompt_embedding_max_pooled, prompt_embedding_signed_max_pooled,))
                    thread.start()

                elif task_type == 'clip_calculation_task':
                    output_file_path, output_file_hash, clip_data = run_clip_calculation_task(worker_state,
                                                                                              generation_task)

                    # spawn upload data and update job thread
                    thread = threading.Thread(target=upload_data_and_update_job_status, args=(
                        job, output_file_path, output_file_hash, clip_data, worker_state.minio_client,))
                    thread.start()

                elif task_type == "generate_image_generation_task":
                    # run generate image generation task
                    run_generate_image_generation_task(generation_task)
                    job['task_completion_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    info(thread_state, "job completed: " + job["uuid"])
                    generation_request.http_update_job_completed(job)

                elif task_type == "generate_inpainting_generation_task":
                    # run generate inpainting generation task
                    run_generate_inpainting_generation_task(generation_task)
                    job['task_completion_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    info(thread_state, "job completed: " + job["uuid"])
                    generation_request.http_update_job_completed(job)
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

    load_clip = False
    if 'clip_calculation_task' in worker_type_list or len(worker_type_list) == 0:
        load_clip = True

    # Initialize worker state
    worker_state = WorkerState(args.device, args.minio_access_key, args.minio_secret_key, queue_size, load_clip)
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
