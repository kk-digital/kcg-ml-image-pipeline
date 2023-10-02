import sys
from PIL import Image
from io import BytesIO
import os
import msgpack
import numpy as np

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.path import separate_bucket_and_file_path
from worker.worker_state import WorkerState
from worker.generation_task.generation_task import GenerationTask


def calculate_image_feature_vector(worker_state: WorkerState, input_file_path: str, input_file_hash: str):
    # get image from minio server
    # Get data of an object.
    bucket_name, file_path = separate_bucket_and_file_path(input_file_path)
    try:
        response = worker_state.minio_client.get_object(bucket_name, file_path)
        image_data = BytesIO(response.data)
        img = Image.open(image_data)
        img = img.convert("RGB")
    except Exception as e:
        raise e
    finally:
        response.close()
        response.release_conn()

    # get feature
    clip_feature_vector = worker_state.clip.get_image_features(img)

    # put to cpu
    clip_feature_vector = clip_feature_vector.cpu().detach()

    # convert to np array
    clip_feature_vector_np_arr = np.array(clip_feature_vector, dtype=np.float32)

    # convert to normal list
    clip_feature_vector_arr = clip_feature_vector_np_arr.tolist()

    return input_file_hash, clip_feature_vector_arr


def run_clip_calculation_task(worker_state: WorkerState, generation_task: GenerationTask):
    input_file_path = generation_task.task_input_dict["input_file_path"]
    input_file_hash, clip_feature_vector = calculate_image_feature_vector(worker_state=worker_state,
                                                                          input_file_path=input_file_path,
                                                                          input_file_hash=
                                                                          generation_task.task_input_dict[
                                                                              "input_file_hash"])
    output_path = os.path.splitext(input_file_path)[0]
    output_path = output_path + "_clip.msgpack"

    clip_feature_dict = {"clip-feature-vector": clip_feature_vector}
    clip_feature_msgpack = msgpack.packb(clip_feature_dict)

    clip_feature_msgpack_buffer = BytesIO()
    clip_feature_msgpack_buffer.write(clip_feature_msgpack)
    clip_feature_msgpack_buffer.seek(0)

    return output_path, input_file_hash, clip_feature_msgpack_buffer
