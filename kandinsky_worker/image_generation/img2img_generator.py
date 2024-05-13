import io
import sys
import os
import uuid

base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from utility.http import generation_request
from utility.http import request
from utility.minio.cmd import connect_to_minio_client, upload_data
from worker.generation_task.generation_task import GenerationTask
from kandinsky_worker.dataloaders.image_embedding import ImageEmbedding


def generate_img2img_generation_jobs_with_kandinsky(image_embedding,
                                                    negative_image_embedding,
                                                    dataset_name,
                                                    prompt_generation_policy,
                                                    self_training= False,
                                                    init_img_path="./test/test_inpainting/white_512x512.jpg",
                                                    decoder_guidance_scale=12,
                                                    prior_guidance_scale=12):

    # get sequential ids
    sequential_ids = request.http_get_sequential_id(dataset_name, 1)

    count = 0
    # generate UUID
    task_uuid = str(uuid.uuid4())
    task_type = "img2img_generation_kandinsky"
    model_name = "kandinsky_2_2"
    model_file_name = "kandinsky-2-2-decoder"
    model_file_path = "input/model/kandinsky/kandinsky-2-2-decoder"
    task_input_dict = {
        "strength": 0.75,
        "seed": "",
        "dataset": dataset_name,
        "file_path": sequential_ids[count]+".jpg",
        "init_img": init_img_path,
        "num_images": 1,
        "image_width": 512,
        "image_height": 512,
        "decoder_steps": 100,
        "prior_guidance_scale": prior_guidance_scale,
        "decoder_guidance_scale": decoder_guidance_scale,
        "self_training": self_training
    }

    prompt_generation_data={
        "prompt_generation_policy": prompt_generation_policy
    }

    image_embedding= image_embedding.detach().cpu().numpy().tolist()[0]
    negative_image_embedding= image_embedding.detach().cpu().numpy().tolist()[0] if negative_image_embedding is not None else None

    # create the job
    generation_task = GenerationTask(uuid=task_uuid,
                                     task_type=task_type,
                                     model_name=model_name,
                                     model_file_name=model_file_name,
                                     model_file_path=model_file_path,
                                     task_input_dict=task_input_dict,
                                     prompt_generation_data=prompt_generation_data)
    generation_task_json = generation_task.to_dict()

    # add job
    response = generation_request.http_add_kandinsky_job(job=generation_task_json,
                                                         positive_embedding=image_embedding,
                                                         negative_embedding=negative_image_embedding)

    return response