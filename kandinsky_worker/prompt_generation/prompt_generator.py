import sys
import os
import uuid

base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from utility.http import generation_request
from utility.http import request
from worker.generation_task.generation_task import GenerationTask


def generate_image_generation_jobs_with_kandinsky(positive_prompt,
                                   negative_prior_prompt,
                                   negative_decoder_prompt,
                                   prompt_scoring_model,
                                   prompt_score,
                                   prompt_generation_policy,
                                   top_k,
                                   dataset_name):

    # get sequential ids
    sequential_ids = request.http_get_sequential_id(dataset_name, 1)

    count = 0
    # generate UUID
    task_uuid = str(uuid.uuid4())
    task_type = "image_generation_kandinsky"
    model_name = "kandinsky"
    model_file_name = "kandinsky-2-2-decoder"
    model_file_path = "input/model/kandinsky/kandinsky-2-2-decoder"
    task_input_dict = {
        "positive_prompt": positive_prompt,
        "negative_prior_prompt": negative_prior_prompt,
        "negative_decoder_prompt": negative_decoder_prompt,
        "strength": 0.4,
        "seed": "",
        "dataset": dataset_name,
        "file_path": sequential_ids[count]+".jpg",
        "num_images": 1,
        "image_width": 512,
        "image_height": 512,
        "decoder_steps": 50,
        "prior_steps": 25,
        "prior_guidance_scale": 4,
        "decoder_guidance_scale": 4
    }
    prompt_generation_data={
        "prompt_scoring_model": prompt_scoring_model,
        "prompt_score": prompt_score,
        "prompt_generation_policy": prompt_generation_policy,
        "top_k": top_k
    }

    generation_task = GenerationTask(uuid=task_uuid,
                                     task_type=task_type,
                                     model_name=model_name,
                                     model_file_name=model_file_name,
                                     model_file_path=model_file_path,
                                     task_input_dict=task_input_dict,
                                     prompt_generation_data=prompt_generation_data)
    generation_task_json = generation_task.to_dict()

    # add job
    response = generation_request.http_add_job(generation_task_json)

    return response

def generate_inpainting_job_with_kandinsky(positive_prompt,
                            negative_prior_prompt,
                            negative_decoder_prompt,
                            prompt_scoring_model,
                            prompt_score,
                            prompt_generation_policy,
                            top_k,
                            dataset_name,
                            init_img_path="./test/test_inpainting/white_512x512.jpg",
                            mask_path="./test/test_inpainting/icon_mask.png"):

    # get sequential ids
    sequential_ids = request.http_get_sequential_id(dataset_name, 1)

    task_uuid = str(uuid.uuid4())
    task_type = "inpainting_kandinsky"
    model_name = "kandinsky"
    model_file_name = "kandinsky-2-2-decoder-inpaint"
    model_file_path = "input/model/kandinsky/kandinsky-2-2-decoder-inpaint"
    task_input_dict = {
        "positive_prompt": positive_prompt,
        "negative_prior_prompt": negative_prior_prompt,
        "negative_decoder_prompt": negative_decoder_prompt,
        "strength": 0.4,
        "seed": "",
        "dataset": dataset_name,
        "file_path": sequential_ids[0] + ".jpg",
        "init_img": init_img_path,
        "init_mask": mask_path,
        "image_width": 512,
        "image_height": 512,
        "decoder_steps": 50,
        "prior_steps": 25,
        "prior_guidance_scale": 4,
        "decoder_guidance_scale": 4
    }
    prompt_generation_data={
        "prompt_scoring_model": prompt_scoring_model,
        "prompt_score": prompt_score,
        "prompt_generation_policy": prompt_generation_policy,
        "top_k": top_k
    }

    generation_task = GenerationTask(uuid=task_uuid,
                                     task_type=task_type,
                                     model_name=model_name,
                                     model_file_name=model_file_name,
                                     model_file_path=model_file_path,
                                     task_input_dict=task_input_dict,
                                     prompt_generation_data=prompt_generation_data)
    generation_task_json = generation_task.to_dict()

    # add job
    response = generation_request.http_add_job(generation_task_json)

    return response