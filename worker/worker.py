import sys
import time
import random
from datetime import datetime
import argparse
from PIL import Image
from termcolor import colored
import os
import threading

base_directory = "./"
sys.path.insert(0, base_directory)

from worker.generation_task.generation_task import GenerationTask
from worker.image_generation.scripts.inpaint_A1111 import img2img
from worker.image_generation.scripts.generate_image_from_text import generate_image_from_text
from worker.worker_state import WorkerState
from worker.http import request
from worker.prompt_generation.prompt_generator import generate_image_generation_jobs_using_generated_prompts, generate_inpainting_generation_jobs_using_generated_prompts


class ThreadState:
    def __init__(self, thread_id, thread_name):
        self.thread_id = thread_id
        self.thread_name = thread_name
def info(thread_state, message):
    print(colored(f"Thread [{thread_state.thread_id}] {thread_state.thread_name}", 'green') + " " + colored("[INFO] ", 'green') + message)


def error(thread_state, message):
    print(colored(f"Thread [{thread_state.thread_id}] {thread_state.thread_name}", 'green') + " " + colored("[ERROR] ", 'red') + message)


def warning(thread_state, message):
    print(colored(f"Thread [{thread_state.thread_id}] {thread_state.thread_name}", 'green') + " " + colored("[WARNING] ", 'yellow') + message)


def run_image_generation_task(worker_state, generation_task):
    # Random seed for now
    # Should we use the seed from job parameters ?
    random.seed(time.time())
    seed = random.randint(0, 2 ** 24 - 1)

    output_file_path, output_file_hash = generate_image_from_text(worker_state.minio_client,
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


def run_inpainting_generation_task(worker_state, generation_task: GenerationTask):
    # TODO(): Make a cache for these images
    # Check if they changed on disk maybe and reload
    init_image = Image.open(generation_task.task_input_dict["init_img"])
    init_mask = Image.open(generation_task.task_input_dict["init_mask"])

    output_file_path, output_file_hash = img2img(
        minio_client=worker_state.minio_client,
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
            job = request.http_get_job()
        else:
            job = request.http_get_job(worker_type)

        if job is not None:
            break

    return job


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
            if task_type == 'inpainting_generation_task':
                # Convert the job into a dictionary
                # Then use the dictionary to create the generation task
                try:
                    output_file_path, output_file_hash = run_inpainting_generation_task(worker_state, generation_task)
                    info(thread_state, "job completed !")
                    job['task_completion_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    job['task_output_file_dict'] = {
                        'output_file_path': output_file_path,
                        'output_file_hash': output_file_hash
                    }
                    info(thread_state, "output file path : " + output_file_path)
                    info(thread_state, "output file hash : " + output_file_hash)
                    request.http_update_job_completed(job)
                except Exception as e:
                    error(thread_state, f"generation task failed: {e}")
                    job['task_error_str'] = str(e)
                    request.http_update_job_failed(job)

            elif task_type == 'image_generation_task':
                try:
                    # Run inpainting task
                    output_file_path, output_file_hash = run_image_generation_task(worker_state, generation_task)
                    info(thread_state, "job completed !")
                    job['task_completion_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    job['task_output_file_dict'] = {
                        'output_file_path': output_file_path,
                        'output_file_hash': output_file_hash
                    }
                    info(thread_state, "output file path : " + output_file_path)
                    info(thread_state, "output file hash : " + output_file_hash)
                    info(thread_state, "job completed")
                    request.http_update_job_completed(job)
                except Exception as e:
                    error(thread_state, f"generation task failed: {e}")
                    job['task_error_str'] = str(e)
                    request.http_update_job_failed(job)

            elif task_type == "generate_image_generation_task":
                try:
                    # run generate image generation task
                    run_generate_image_generation_task(generation_task)
                    info(thread_state, "job completed !")
                    job['task_completion_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    info(thread_state, "job completed")
                    request.http_update_job_completed(job)

                except Exception as e:
                    error(thread_state, f"generation task failed: {e}")
                    job['task_error_str'] = str(e)
                    request.http_update_job_failed(job)

            elif task_type == "generate_inpainting_generation_task":
                try:
                    # run generate inpainting generation task
                    run_generate_inpainting_generation_task(generation_task)
                    info(thread_state, "job completed !")
                    job['task_completion_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    info(thread_state, "job completed")
                    request.http_update_job_completed(job)

                except Exception as e:
                    error(thread_state, f"generation task failed: {e}")
                    job['task_error_str'] = str(e)
                    request.http_update_job_failed(job)

            else:
                e = "job with task type '" + task_type + "' is not supported"
                error(thread_state, e)
                job['task_error_str'] = e
                request.http_update_job_failed(job)

            job_end_time = time.time()
            last_job_time = job_end_time
            job_elapsed_time = job_end_time - job_start_time
            info(thread_state, f"job took {job_elapsed_time:.4f} seconds to execute.")

        else:
            # If there was no job, go to sleep for a while
            sleep_time_in_seconds = 1
            time.sleep(sleep_time_in_seconds)

def main():
    args = parse_args()

    thread_state = ThreadState(0, "Job Fetcher")

    queue_size = args.queue_size
    # get worker type
    worker_type = args.worker_type
    worker_type = worker_type.strip()  # remove trailing and leading spaces
    worker_type = worker_type.replace(' ', '')  # remove spaces
    worker_type_list = worker_type.split(",")  # split by comma

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
            sleep_time_in_seconds = 2
            info(thread_state, "Queue is full, going to sleep for " + f"{sleep_time_in_seconds:.4f}" + " seconds")
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
            sleep_time_in_seconds = 10
            info(thread_state, "Did not find job, going to sleep for " + f"{sleep_time_in_seconds:.4f}" + " seconds")
            time.sleep(sleep_time_in_seconds)



if __name__ == '__main__':
    main()