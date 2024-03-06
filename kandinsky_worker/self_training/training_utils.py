import io
import os
import sys
import msgpack
import torch
from torch.nn.functional import cosine_similarity

from utility.path import separate_bucket_and_file_path
base_directory = "./"
sys.path.insert(0, base_directory)

from kandinsky_worker.dataloaders.image_embedding import ImageEmbedding 
from utility.minio import cmd
from kandinsky_worker.worker_state import WorkerState
from worker.generation_task.generation_task import GenerationTask
from data_loader.utils import get_object

def get_input_clip_vector(minio_client, dataset, output_file_path):
    image_embeddings_path = output_file_path.replace("'_clip_kandinsky.msgpack", "_embedding.msgpack")    
    embedding_data = get_object(minio_client, image_embeddings_path)
    embedding_dict = ImageEmbedding.from_msgpack_bytes(embedding_data)
    features_vector= embedding_dict.image_embedding

    return features_vector

def get_output_clip_vector(minio_client, dataset, output_file_path):
    features_data = get_object(minio_client, output_file_path)
    features_vector = msgpack.unpackb(features_data)["clip-feature-vector"]
    features_vector= torch.tensor(features_vector)

    return features_vector

def store_self_training_data(worker_state: WorkerState, job: dict):
    minio_client= worker_state.minio_client
    output_file_path= job['task_output_file_dict']['output_file_path']
    bucket_name, file_path = separate_bucket_and_file_path(output_file_path)
    dataset= file_path.split('/')[0]
    scoring_model= worker_state.scoring_models[dataset]

    score_mean= float(scoring_model.mean)
    score_std= float(scoring_model.standard_deviation)

    if scoring_model is None:
        raise Exception("No scoring model has been loaded for this dataset.")

    input_clip_vector= get_input_clip_vector(minio_client, dataset, file_path)
    output_clip_vector= get_output_clip_vector(minio_client, dataset, file_path)

    input_clip_score = scoring_model.predict_clip(input_clip_vector).item()
    input_clip_score = (input_clip_score - score_mean) / score_std 
    output_clip_score = scoring_model.predict_clip(output_clip_vector).item()
    output_clip_score = (output_clip_score - score_mean) / score_std 

    cosine_sim = cosine_similarity(input_clip_vector, output_clip_vector).item()

    data = {
        'input_clip': input_clip_vector.detach().cpu().numpy().tolist(),
        'output_clip': output_clip_vector.detach().cpu().numpy().tolist(),
        'input_clip_score': input_clip_score,
        'output_clip_score': output_clip_score,
        'cosine_sim': cosine_sim
    }
    
    batch_size = 10000
    dataset_path = f"{dataset}/data/latent-generator" + "/self_training/"
    dataset_files = minio_client.list_objects('datasets', prefix=dataset_path, recursive=True)
    dataset_files = [file.object_name for file in dataset_files]
    last_file_path= dataset_files[-1]

    if last_file_path.endswith("_incomplete.msgpack"):
        data = minio_client.get_object('datasets', last_file_path)
        content = data.read()
        batch = msgpack.loads(content)
        index = len(dataset_files)
        batch= batch.append(data)

        if len(batch) == batch_size:
            minio_client.remove_object('datasets', last_file_path)
            store_batch_in_msgpack_file(minio_client, dataset, batch, index)
        else:
            store_batch_in_msgpack_file(minio_client, dataset, batch, index, incomplete=True)
    else:
        index = len(dataset_files) + 1
        store_batch_in_msgpack_file(minio_client, dataset, [data], index, incomplete=True)


# function for storing self training data in a msgpack file
def store_batch_in_msgpack_file(minio_client, dataset, batch, index, incomplete=False):
    if incomplete:
        file_path=f"{str(index).zfill(4)}_incomplete.msgpack"
    else:
        file_path=f"{str(index).zfill(4)}.msgpack"
    packed_data = msgpack.packb(batch, use_single_float=True)

    local_file_path = f"output/temporary_file.msgpack"
    with open(local_file_path, 'wb') as local_file:
        local_file.write(packed_data)

    with open(local_file_path, 'rb') as file:
        content = file.read()

    buffer = io.BytesIO(content)
    buffer.seek(0)

    minio_path = f"{dataset}/data/latent-generator" + f"/self_training/{file_path}"
    cmd.upload_data(minio_client, 'datasets', minio_path, buffer)
    


