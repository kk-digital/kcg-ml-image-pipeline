import sys
from datetime import datetime

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.http import inpainting_request
from inpainting_worker import worker

def add_jobs():
    inpainting_request.http_add_job({
        "task_type": "kandinsky-2-txt-to-img-inpainting",
        "uuid": "400a0cdf-fb83-412b-8c02-2027146490ad",
        "model_name": "kandinsky_2_2",
        "model_file_name": "kandinsky-2-2-decoder-inpaint",
        "model_file_path": "input/model/kandinsky/kandinsky-2-2-decoder",
        "model_hash": None,
        "task_creation_time": "2024-02-26T20:08:27.502000",
        "task_start_time": "2024-02-26 20:08:58",
        "task_completion_time": "2024-02-26 20:09:03",
        "task_error_str": None,
        "task_input_dict": {
            "strength": 0.4,
            "seed": "",
            "dataset": "test-generations",
            "file_path": "datasets/environmental/output/test_inpainting/022912.jpg",
            "init_img":"datasets/environmental/output/test_inpainting/test_inpainting_init_img_001.jpg",
            "init_mask":"datasets/environmental/output/test_inpainting/test_inpainting_init_mask_001.png",
            "num_images": 1,
            "image_width": 512,
            "image_height": 512,
            "prior_steps": 25,
            "decoder_steps": 50,
            "prior_guidance_scale": 4,
            "decoder_guidance_scale": 4,
            "positive_prompt":"a tiger sitting on a park bench",
            "negative_prior_prompt":"",
            "negative_decoder_prompt":""
        },
        "task_input_file_dict": None,
        "task_output_file_dict": None,
        "task_attributes_dict": None,
        "prompt_generation_data": {
            "prompt_scoring_model": None,
            "prompt_score":None,
            "top_k": None,
            "prompt_generation_policy": "pez_optimization"
        },
    })

# for i in range(10):
#     add_jobs()

add_jobs()
worker.main()
