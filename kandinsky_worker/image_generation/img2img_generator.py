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
                                                    init_img_path="./test/test_inpainting/white_512x512.jpg",
                                                    minio_client=None):

    # get sequential ids
    sequential_ids = request.http_get_sequential_id(dataset_name, 1)

    count = 0
    # generate UUID
    task_uuid = str(uuid.uuid4())
    task_type = "image_generation_kandinsky"
    model_name = "kandinsky_2_2"
    model_file_name = "kandinsky-2-2-decoder"
    model_file_path = "input/model/kandinsky/kandinsky-2-2-decoder"
    task_input_dict = {
        "strength": 0.2,
        "seed": "",
        "dataset": dataset_name,
        "file_path": sequential_ids[count]+".jpg",
        "init_img": init_img_path,
        "num_images": 1,
        "image_width": 512,
        "image_height": 512,
        "decoder_steps": 50,
        "decoder_guidance_scale": 8
    }

    # upload the image embeddings to minIO
    image_embedding_data= ImageEmbedding(job_uuid= task_uuid,
                                         dataset= dataset_name,
                                         image_embedding= image_embedding,
                                         negative_image_embedding= negative_image_embedding)
    
    output_file_path = os.path.join("datasets", dataset_name, task_input_dict['file_path'])
    image_embeddings_path = output_file_path.replace(".jpg", "_embedding.msgpack")

    msgpack_string = image_embedding_data.get_msgpack_string()

    buffer = io.BytesIO()
    buffer.write(msgpack_string)
    buffer.seek(0)

    if minio_client is None:
        minio_client= connect_to_minio_client()

    upload_data(minio_client, "datasets", image_embeddings_path, buffer) 

    # create the job
    generation_task = GenerationTask(uuid=task_uuid,
                                     task_type=task_type,
                                     model_name=model_name,
                                     model_file_name=model_file_name,
                                     model_file_path=model_file_path,
                                     task_input_dict=task_input_dict)
    generation_task_json = generation_task.to_dict()

    # add job
    response = generation_request.http_add_job(generation_task_json)

    return response