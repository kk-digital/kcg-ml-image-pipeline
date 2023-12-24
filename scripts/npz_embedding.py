import pandas as pd
import numpy as np
import msgpack
from minio import Minio
from autofaiss import build_index

# MinIO Client connection setup
def connect_to_minio_client(minio_ip_addr, access_key, secret_key):
    print("Connecting to minio client...")
    client = Minio(minio_ip_addr, access_key, secret_key, secure=False)
    print("Successfully connected to minio client...")
    return client

# Process embedding msgpack files and collect data
def process_files(minio_client, bucket_name, embeddings_dir):
    metadata = []

    objects = minio_client.list_objects(bucket_name, recursive=True)
    for obj in objects:
        if obj.object_name.endswith('_embedding.msgpack'):
            # Extract embedding data
            embedding_data = msgpack.unpackb(minio_client.get_object(bucket_name, obj.object_name).read(), raw=False)
            
            # Extract metadata and embeddings
            uuid = embedding_data['uuid']
            image_hash = embedding_data['image_hash']
            dataset = embedding_data['dataset']
            positive_embedding = np.array(embedding_data['positive_embedding'], dtype=np.float32)
            negative_embedding = np.array(embedding_data['negative_embedding'], dtype=np.float32)
            
            # Save embeddings as .npy files
            pos_filename = f"{uuid}_positive.npy"
            neg_filename = f"{uuid}_negative.npy"
            np.save(f"{embeddings_dir}/{pos_filename}", positive_embedding)
            np.save(f"{embeddings_dir}/{neg_filename}", negative_embedding)
            
            # Save metadata
            metadata.append((image_hash, uuid, dataset, 'positive', pos_filename))
            metadata.append((image_hash, uuid, dataset, 'negative', neg_filename))
    
    return metadata

def build_autofaiss_index(embeddings_dir, index_path, index_infos_path, memory_available="4G"):
    # Build index using autofaiss
    build_index(embeddings_path=embeddings_dir, index_path=index_path, 
                index_infos_path=index_infos_path, max_index_memory_usage=memory_available)

def save_csv(metadata, csv_file_path):
    # Save metadata to .csv file
    df = pd.DataFrame(metadata, columns=['image_hash', 'uuid', 'dataset', 'embedding_type', 'npy_filename'])
    df.to_csv(csv_file_path, index=False)

def main():
    minio_client = connect_to_minio_client('minio_ip_address', 'access_key', 'secret_key')
    embeddings_dir = 'path_to_embeddings_dir'
    metadata = process_files(minio_client, 'bucket_name', embeddings_dir)
    csv_file_path = 'metadata.csv'
    save_csv(metadata, csv_file_path)
    
    # Path where the KNN index and index info will be saved
    index_path = 'my_index_folder/knn.index'
    index_infos_path = 'my_index_folder/index_infos.json'
    
    # Build and save autofaiss index
    build_autofaiss_index(embeddings_dir, index_path, index_infos_path)

if __name__ == '__main__':
    main()


"""
## script for one npy file

import pandas as pd
import numpy as np
import msgpack
from minio import Minio
from autofaiss import build_index

# MinIO Client connection setup
def connect_to_minio_client(minio_ip_addr, access_key, secret_key):
    print("Connecting to minio client...")
    client = Minio(minio_ip_addr, access_key, secret_key, secure=False)
    print("Successfully connected to minio client...")
    return client

# Process embedding msgpack files and collect data
def process_files(minio_client, bucket_name):
    embeddings = []
    metadata = []

    objects = minio_client.list_objects(bucket_name, recursive=True)
    for obj in objects:
        if obj.object_name.endswith('_embedding.msgpack'):
            # Extract embedding data
            embedding_data = msgpack.unpackb(minio_client.get_object(bucket_name, obj.object_name).read(), raw=False)
            
            # Extract metadata and embeddings
            uuid = embedding_data['uuid']
            image_hash = embedding_data['image_hash']
            dataset = embedding_data['dataset']
            positive_embedding = np.array(embedding_data['positive_embedding'], dtype=np.float32)
            negative_embedding = np.array(embedding_data['negative_embedding'], dtype=np.float32)
            
            # Add embeddings to the list
            embeddings.append(positive_embedding)
            embeddings.append(negative_embedding)
            
            # Save metadata
            pos_index = len(embeddings) - 2  # Index of the positive embedding
            neg_index = len(embeddings) - 1  # Index of the negative embedding
            metadata.append((image_hash, uuid, dataset, 'positive', pos_index))
            metadata.append((image_hash, uuid, dataset, 'negative', neg_index))
    
    # Convert embeddings list to a numpy array and save as a single .npy file
    embeddings_array = np.array(embeddings, dtype=np.float32)
    np.save('embeddings.npy', embeddings_array)

    return metadata

def build_autofaiss_index(embeddings_npy_path, index_path, index_infos_path, memory_available="4G"):
    # Build index using autofaiss
    build_index(embeddings_path=embeddings_npy_path, index_path=index_path, 
                index_infos_path=index_infos_path, max_index_memory_usage=memory_available)

def save_csv(metadata, csv_file_path):
    # Save metadata to .csv file
    df = pd.DataFrame(metadata, columns=['image_hash', 'uuid', 'dataset', 'embedding_type', 'npy_index'])
    df.to_csv(csv_file_path, index=False)

def main():
    minio_client = connect_to_minio_client('minio_ip_address', 'access_key', 'secret_key')
    metadata = process_files(minio_client, 'bucket_name')
    csv_file_path = 'metadata.csv'
    save_csv(metadata, csv_file_path)
    
    # Path where the .npy file is saved
    embeddings_npy_path = 'embeddings.npy'
    
    # Path where the KNN index and index info will be saved
    index_path = 'my_index_folder/knn.index'
    index_infos_path = 'my_index_folder/index_infos.json'
    
    # Build and save autofaiss index
    build_autofaiss_index(embeddings_npy_path, index_path, index_infos_path)

if __name__ == '__main__':
    main()

    """