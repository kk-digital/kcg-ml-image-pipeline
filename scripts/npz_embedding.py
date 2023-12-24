import pandas as pd
import numpy as np
import msgpack
from minio import Minio

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
            positive_embedding = embedding_data['positive_embedding']
            negative_embedding = embedding_data['negative_embedding']
            
            # Save embeddings and metadata
            pos_index = len(embeddings)
            embeddings.append(positive_embedding)
            metadata.append((image_hash, uuid, dataset, 'positive', pos_index))
            
            neg_index = len(embeddings)
            embeddings.append(negative_embedding)
            metadata.append((image_hash, uuid, dataset, 'negative', neg_index))
    
    return embeddings, metadata

def save_npz_and_csv(embeddings, metadata):
    # Save embeddings to .npz file
    npz_file_path = 'embeddings.npz'
    np.savez_compressed(npz_file_path, *embeddings)

    # Save metadata to .csv file
    csv_file_path = 'metadata.csv'
    df = pd.DataFrame(metadata, columns=['image_hash', 'uuid', 'dataset', 'embedding_type', 'npz_index'])
    df.to_csv(csv_file_path, index=False)

def main():
    minio_client = connect_to_minio_client('minio_ip_address', 'access_key', 'secret_key')
    embeddings, metadata = process_files(minio_client, 'bucket_name')
    save_npz_and_csv(embeddings, metadata)

if __name__ == '__main__':
    main()
