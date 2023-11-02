import json
import sys
import os
root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_directory)
import minio
import msgpack  # Required to deserialize the embedding.msgpack contents
import argparse  # Required to parse command-line arguments
from utility.minio import cmd

MINIO_ADDRESS = "192.168.3.5:9000"
ACCESS_KEY = "3lUCPCfLMgQoxrYaxgoz"
SECRET_KEY = "MXszqU6KFV6X95Lo5jhMeuu5Xm85R79YImgI3Xmp"

def check_discrepancies(dataset_name):
    discrepancies = []
    json_files_processed = 0
    msgpack_files_processed = 0
    
    # Initialize MinIO client using cmd function
    minio_client = cmd.get_minio_client(ACCESS_KEY, SECRET_KEY, MINIO_ADDRESS)

    # Define the prefix for the ranking .json files
    ranking_json_prefix = f"{dataset_name}/data/ranking/aggregate/"
    print(f"Searching for .json files with prefix: {ranking_json_prefix}")

    # Fetch the list of ranking .json objects using cmd function
    ranking_json_objects = cmd.get_list_of_objects_with_prefix(minio_client, 'datasets', ranking_json_prefix)

    # Iterate through ranking .json files to get the selected_image_hash
    for obj_name in ranking_json_objects:
        json_files_processed += 1
        print(f"Processing .json file ({json_files_processed}): {obj_name}")

        data = minio_client.get_object('datasets', obj_name)
        json_data = json.loads(data.read().decode())
        
        selected_image_hash = json_data.get('selected_image_hash', '')
        
        # Extract file_name of the selected image
        selected_image_index = json_data.get('selected_image_index', 0)
        # Extract file name of the selected image
        selected_image_metadata = json_data.get(f'image_{selected_image_index + 1}_metadata', {})
        selected_image_file_name = selected_image_metadata.get('file_name', '')
        selected_image_file_path_without_extension = selected_image_file_name.rsplit('.', 1)[0]  # Remove file extension

        # Construct the expected embedding msgpack file path based on the file name
        expected_msgpack_path = f"{selected_image_file_path_without_extension}_embedding.msgpack"
        expected_msgpack_path = os.path.relpath(expected_msgpack_path, 'datasets')

        # Check if the expected embedding msgpack file exists
        try:
            msgpack_data = minio_client.get_object('datasets', expected_msgpack_path)
            deserialized_data = msgpack.unpackb(msgpack_data.read(), raw=False)
            
            msgpack_files_processed += 1
            print(f"Processing .msgpack file ({msgpack_files_processed}): {expected_msgpack_path}")
            
            # Get file_hash from the deserialized object
            file_hash_from_msgpack = deserialized_data.get('file_hash', '')
            
            # Compare the file_hash from the msgpack with the selected_image_hash
            if file_hash_from_msgpack != selected_image_hash:
                discrepancy = {
                    'image_path': selected_image_file_name,
                    'datapoint_json_file': {
                        'path': obj_name,  
                        'hash': selected_image_hash  
                    },
                    'embedding_msgpack_file': {
                        'path': expected_msgpack_path,  
                        'hash': file_hash_from_msgpack  
                    },
                }
                discrepancies.append(discrepancy)
                print(f"Discrepancy found: {discrepancy}")

        except Exception as e:
            print(f"An error occurred: {e}")

    # Save discrepancies to a JSON file
    with open('discrepancies.json', 'w') as f:
        json.dump(discrepancies, f, indent=4)

    print(f"Processed {json_files_processed} .json files and {msgpack_files_processed} .msgpack files.")
    print(f"Found {len(discrepancies)} discrepancies. Saved to discrepancies.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check discrepancies between selected image hash and file hash in embedding msgpack.")
    parser.add_argument('--dataset_name', required=True, help="Name of the dataset to check.")
    args = parser.parse_args()

    check_discrepancies(args.dataset_name)
