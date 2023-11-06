import time
from minio import Minio
import msgpack
import csv
import os
import sys
base_directory = "./"
sys.path.insert(0, base_directory)
from utility.minio import cmd
import argparse


def unpack_msgpack(data):
    return msgpack.unpackb(data, raw=False)

def write_csv(file_hash, file_path, image_prompt, job_uuid, dataset, output_directory, processing_time):
    csv_file = f'{output_directory}/{dataset}.csv'
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if os.stat(csv_file).st_size == 0:
            writer.writerow(['job uuid', 'image hash', 'image path', 'image prompt', 'processing time (s)'])
        writer.writerow([job_uuid, file_hash, file_path, image_prompt, processing_time])

def process_dataset(client, dataset_name, output_directory):
    start_time = time.time()
    print(f"Processing dataset: {dataset_name}")

    prefix = f'{dataset_name}/'
    print(f"Looking for msgpack files with prefix: {prefix}")
    msgpack_files = cmd.get_list_of_objects_with_prefix(client, 'datasets', prefix)
    msgpack_files = [f for f in msgpack_files if f.endswith('_data.msgpack')]
    print(f"Found {len(msgpack_files)} msgpack files.")

    for file_path in msgpack_files:
        file_start_time = time.time()
        print(f"Processing file: {file_path}")
        data = cmd.get_file_from_minio(client, 'datasets', file_path)
        if data:
            try:
                unpacked_data = unpack_msgpack(data.read())
                print(f"Unpacked data for {file_path}")
                job_uuid = unpacked_data.get('job_uuid')
                file_hash = unpacked_data.get('file_hash')
                image_prompt = unpacked_data.get('positive_prompt')
                image_file_path = file_path.replace('_data.msgpack', '.jpg')

                processing_time = time.time() - file_start_time
                write_csv(file_hash, image_file_path, image_prompt, job_uuid, dataset_name, output_directory, processing_time)
                print(f"CSV entry for file {image_file_path} created in {processing_time:.2f} seconds.")
            except msgpack.exceptions.UnpackException as e:
                print(f"Error unpacking msgpack data from file {file_path}: {e}")

    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds for {len(msgpack_files)} images.")
    print(f"Average time per image: {total_time / len(msgpack_files):.2f} seconds.")

def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--minio-addr', required=False, default='123.176.98.90:9000', help='Minio server address')
    parser.add_argument('--minio-access-key', required=True, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=True, help='Minio secret key')
    parser.add_argument('--dataset-name', required=True, help='Name of the dataset')
    parser.add_argument('--output-dir', required=False, default='./output', help='Directory to save the output CSV file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(f"Arguments received: addr={args.minio_addr}, access_key={args.minio_access_key}, secret_key={args.minio_secret_key}, dataset_name={args.dataset_name}, output_dir={args.output_dir}")

    try:
        minio_client = cmd.connect_to_minio_client(
            minio_ip_addr=args.minio_addr,
            access_key=args.minio_access_key,
            secret_key=args.minio_secret_key
        )
        process_dataset(minio_client, args.dataset_name, args.output_dir)
    except Exception as e:
        print(f"An error occurred: {e}")
