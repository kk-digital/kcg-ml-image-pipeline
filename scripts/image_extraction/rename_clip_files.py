import argparse
from minio import Minio, S3Error
from tqdm import tqdm

from utility.minio import cmd


def rename_files_in_bucket(minio_client: Minio, bucket_name: str):
    try:
        objects = minio_client.list_objects(bucket_name, recursive=True)

        total_objects = len(objects)
        print(f'Total number of objects: {total_objects}')
        for obj in tqdm(objects, total=total_objects):
            if obj.object_name.endswith('_clip-h.msgpack'):
                new_name = obj.object_name.replace('_clip-h.msgpack', '_clip_kandinsky.msgpack')
                minio_client.copy_object(bucket_name, new_name, f'/{bucket_name}/{obj.object_name}')
                minio_client.remove_object(bucket_name, obj.object_name)
    except S3Error as e:
        print(f'An error occurred: {e}')

def parse_args():
    parser = argparse.ArgumentParser()

    # Add arguments for MinIO connection
    parser.add_argument('--minio-access-key', type=str, help='MinIO access key')
    parser.add_argument('--minio-secret-key', type=str, help='MinIO secret key')

    return parser.parse_args()

def main():
    args = parse_args()

    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key)

    rename_files_in_bucket(minio_client, "extracts")

    
if __name__ == '__main__':
    main()