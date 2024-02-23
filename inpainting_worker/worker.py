import sys
import time
from datetime import datetime
import argparse
from termcolor import colored
import threading
import traceback

base_directory = "./"
sys.path.insert(0, base_directory)

from inpainting_worker.worker_state import WorkerState
from utility.http import inpainting_request


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
    

def parse_args():
    parser = argparse.ArgumentParser(description="Worker for inpainting")

    # Required parameters
    parser.add_argument("--queue_size", type=int, default=8)

    return parser.parse_args()


def get_job_if_exist():
    job = inpainting_request.http_get_job()

    return job


def process_jobs(worker_state):
    thread_state = ThreadState(1, "Inpainting Job Processor")
    last_job_time = time.time()

    while True:
        job = worker_state.job_queue.get()

        if job is not None:
            try:
                print('\n\n')
                info(thread_state, 'Job:') # print the job for test
                print(job)
                job['task_start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                job['task_completion_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                info(thread_state, "job completed: " + job["uuid"])

                inpainting_request.http_update_job_completed(job)
            except Exception as e:
                error(thread_state, f"generation task failed: {traceback.format_exc()}")
                job['task_error_str'] = str(e)
                inpainting_request.http_update_job_failed(job)
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

    thread_state = ThreadState(0, "Inpainting Job Fetcher")
    queue_size = args.queue_size

    # Initialize worker state
    worker_state = WorkerState(queue_size)

    info(thread_state, "Starting worker!")

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
        job = get_job_if_exist()
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
