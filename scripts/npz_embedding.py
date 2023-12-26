'''
import pandas as pd
import numpy as np
import msgpack
import os
from minio import Minio
from autofaiss import build_index

# MinIO Client connection setup
def connect_to_minio_client(minio_ip_addr, access_key, secret_key):
    print("Connecting to minio client...")
    client = Minio(minio_ip_addr, access_key=access_key, secret_key=secret_key, secure=False)
    print("Successfully connected to minio client...")
    return client

# Process embedding msgpack files and collect data
def process_files(minio_client, bucket_name, embeddings_dir):
    metadata = []

    objects = minio_client.list_objects(bucket_name, recursive=True)

    os.makedirs(embeddings_dir, exist_ok=True)
    for obj in objects:
        if obj.object_name.endswith('_embedding.msgpack'):
            # Extract embedding data
            data = minio_client.get_object(bucket_name, obj.object_name).read()
            embedding_data = msgpack.unpackb(data, raw=False)
            
            # Extract metadata and embeddings
            job_uuid = embedding_data['job_uuid']
            file_hash = embedding_data['file_hash']
            dataset = embedding_data['dataset']
            positive_embedding = np.array(embedding_data['positive_embedding']['__ndarray__'], dtype=np.float32)
            negative_embedding = np.array(embedding_data['negative_embedding']['__ndarray__'], dtype=np.float32)
            
            # Save embeddings as .npy files
            pos_filename = f"{job_uuid}_positive.npy"
            neg_filename = f"{job_uuid}_negative.npy"
            np.save(f"{embeddings_dir}/{pos_filename}", positive_embedding)
            np.save(f"{embeddings_dir}/{neg_filename}", negative_embedding)
            
            # Save metadata
            metadata.append((file_hash, job_uuid, dataset, 'positive', pos_filename))
            metadata.append((file_hash, job_uuid, dataset, 'negative', neg_filename))
    
    return metadata

def build_autofaiss_index(embeddings_dir, index_path, index_infos_path, memory_available="4G"):
    # Build index using autofaiss
    build_index(embeddings_path=embeddings_dir, index_path=index_path, 
                index_infos_path=index_infos_path, max_index_memory_usage=memory_available)

def save_csv(metadata, csv_file_path):
    # Save metadata to .csv file
    df = pd.DataFrame(metadata, columns=['file_hash', 'job_uuid', 'dataset', 'embedding_type', 'npy_filename'])
    df.to_csv(csv_file_path, index=False)

def main():
    minio_client = connect_to_minio_client('123.176.98.90:9000', '3lUCPCfLMgQoxrYaxgoz', 'MXszqU6KFV6X95Lo5jhMeuu5Xm85R79YImgI3Xmp')
    embeddings_dir = 'path_to_embeddings_dir'
    metadata = process_files(minio_client, 'datasets', embeddings_dir)
    csv_file_path = 'metadata.csv'
    save_csv(metadata, csv_file_path)
    
    # Path where the KNN index and index info will be saved
    index_path = 'my_index_folder/knn.index'
    index_infos_path = 'my_index_folder/index_infos.json'
    
    # Build and save autofaiss index
    build_autofaiss_index(embeddings_dir, index_path, index_infos_path)

if __name__ == '__main__':
    main()

'''



## script for one npy file
import pandas as pd
import numpy as np
import msgpack
from minio import Minio
from autofaiss import build_index
import os
from tqdm import tqdm

# MinIO Client connection setup
def connect_to_minio_client(minio_ip_addr, access_key, secret_key):
    print("Connecting to minio client...")
    client = Minio(minio_ip_addr, access_key=access_key, secret_key=secret_key, secure=False)
    print("Successfully connected to minio client...")
    return client

# Process embedding msgpack files and collect data
def process_files(minio_client, bucket_name, dataset_folder):
    embeddings = []
    metadata = []

    # Ensure the embeddings folder exists
    os.makedirs('embeddings', exist_ok=True)

    objects = minio_client.list_objects(bucket_name, prefix=dataset_folder, recursive=True)
    object_list = list(objects)  # Convert to list for tqdm
    for obj in tqdm(object_list, desc="Processing files"):
        if obj.object_name.endswith('_embedding.msgpack'):
            # Extract embedding data
            data = minio_client.get_object(bucket_name, obj.object_name).read()
            embedding_data = msgpack.unpackb(data, raw=False)
            
            # Extract metadata and embeddings
            job_uuid = embedding_data['job_uuid']
            file_hash = embedding_data['file_hash']
            dataset = embedding_data['dataset']
            positive_embedding = np.array(embedding_data['positive_embedding']['__ndarray__'], dtype=np.float32)
            negative_embedding = np.array(embedding_data['negative_embedding']['__ndarray__'], dtype=np.float32)
            
            # Add embeddings to the list
            embeddings.append(positive_embedding)
            embeddings.append(negative_embedding)
            
            # Save metadata
            pos_index = len(embeddings) - 2  # Index of the positive embedding
            neg_index = len(embeddings) - 1  # Index of the negative embedding
            metadata.append((file_hash, job_uuid, dataset, 'positive', pos_index))
            metadata.append((file_hash, job_uuid, dataset, 'negative', neg_index))
    
    # Convert embeddings list to a numpy array and save as a single .npy file
    embeddings_array = np.array(embeddings, dtype=np.float32)
    npy_file_path = 'embeddings/embeddings.npy'
    np.save(npy_file_path, embeddings_array)

    return metadata, npy_file_path

def build_autofaiss_index(embeddings_npy_path, index_path, index_infos_path, memory_available="4G"):
    # Ensure the index folder exists
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    # Load embeddings from .npy file
    embeddings = np.load(embeddings_npy_path)

    # Build index using autofaiss
    build_index(embeddings=embeddings, index_path=index_path, 
                index_infos_path=index_infos_path, max_index_memory_usage=memory_available)


def save_csv(metadata, csv_file_path):
    # Ensure the CSV file directory exists
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    # Save metadata to .csv file
    df = pd.DataFrame(metadata, columns=['file_hash', 'job_uuid', 'dataset', 'embedding_type', 'npy_index'])
    df.to_csv(csv_file_path, index=False)

def main():
    minio_client = connect_to_minio_client('192.168.3.5:9000', 'v048BpXpWrsVIHUfdAix', '4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu')
    
    # The directory within the bucket where the embeddings are stored
    dataset_folder = 'test-generations'
    
    metadata, embeddings_npy_path = process_files(minio_client, 'datasets', dataset_folder)
    csv_file_path = 'metadata/metadata.csv'
    save_csv(metadata, csv_file_path)
    
    # Path where the KNN index and index info will be saved
    index_path = 'my_index_folder/knn.index'
    index_infos_path = 'my_index_folder/index_infos.json'
    
    # Build and save autofaiss index
    build_autofaiss_index(embeddings_npy_path, index_path, index_infos_path)

if __name__ == '__main__':
    main()

