import sys
import time
from datetime import datetime
import argparse
from termcolor import colored

base_directory = "./"
sys.path.insert(0, base_directory)

from training_worker.http import request
from training_worker.ab_ranking_linear.script.ab_ranking_linear import run_ab_ranking_linear_task


def info(message):
    print(colored("[INFO] ", 'green') + message)


def error(message):
    print(colored("[ERROR] ", 'red') + message)


def warning(thread_state, message):
    print(colored("[WARNING] ", 'yellow') + message)


def parse_args():
    parser = argparse.ArgumentParser(description="Worker for training models")

    # Required parameters
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


def get_worker_type_list(worker_type: str):
    worker_type = worker_type.strip()  # remove trailing and leading spaces
    worker_type = worker_type.replace(' ', '')  # remove spaces
    worker_type_list = worker_type.split(",")  # split by comma
    return worker_type_list


def main():
    args = parse_args()
    minio_access_key = args.minio_access_key
    minio_secret_key = args.minio_secret_key

    # get worker type
    worker_type_list = get_worker_type_list(args.worker_type)
    sleep_time_in_seconds = 5

    while True:
        # try to find a job
        job = get_job_if_exist(worker_type_list)
        if job is not None:
            job_start_time = time.time()
            task_type = job['task_type']
            try:
                if task_type == 'ab_ranking_linear_task':
                    model_output_path, \
                        report_output_path, \
                        graph_output_path = run_ab_ranking_linear_task(training_task=job,
                                                                       minio_access_key=minio_access_key,
                                                                       minio_secret_key=minio_secret_key)

                    # update job info after completion
                    job['task_completion_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    job['task_output_file_dict'] = {
                        'output_model_path': model_output_path,
                        'output_report_path': report_output_path,
                        'output_graph_path': graph_output_path
                    }
                    info("output_model_path: " + model_output_path)
                    info("job completed: " + job["uuid"])

                    # update status
                    request.http_update_job_completed(job)
                else:
                    e = "job with task type '" + task_type + "' is not supported"
                    error(e)
                    job['task_error_str'] = e
                    request.http_update_job_failed(job)
            except Exception as e:
                error(f"generation task failed: {e}")
                job['task_error_str'] = str(e)
                request.http_update_job_failed(job)

            job_end_time = time.time()
            job_elapsed_time = job_end_time - job_start_time
            info(f"job took {job_elapsed_time:.4f} seconds to execute.")
        else:
            info("Did not find job, going to sleep for " + f"{sleep_time_in_seconds:.4f}" + " seconds")
            time.sleep(sleep_time_in_seconds)


if __name__ == '__main__':
    main()
