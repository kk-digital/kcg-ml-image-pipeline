import sys
from datetime import datetime

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.http import inpainting_request
from inpainting_worker import worker

def add_jobs():
    inpainting_request.http_add_job({
        "task_type": "image_generation_sd_1_5",
        "uuid": "",
        "model_name": "sd_1_5",
        "model_file_name": "test_mode",
        "model_file_path": "test_mode",
        "model_hash": "test_mode",
        "task_creation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "task_start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "task_completion_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "task_error_str": "",
        "task_input_dict": {},
        "task_input_file_dict": {},
        "task_output_file_dict": {},
        "task_attributes_dict": {},
        "prompt_generation_data": {}
    })

for i in range(10):
    add_jobs()

worker.main()
