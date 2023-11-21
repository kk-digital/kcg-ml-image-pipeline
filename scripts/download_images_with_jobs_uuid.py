import csv
import os
import sys
import requests
import argparse
from urllib.parse import unquote

def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def download_image(uuid, output_path, use_uuid):
    url = f"http://123.176.98.90:8764/get-image-by-job-uuid/{uuid}"
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        if use_uuid:
            filename = f"{uuid}.jpg"
        else:
            content_disposition = response.headers.get('content-disposition', '')
            if 'filename=' in content_disposition:
                filename = content_disposition.split('filename=')[1].strip('\"\'')
                filename = unquote(filename)
            else:
                filename = f"{uuid}.jpg"

        if os.path.isdir(output_path) or not os.path.basename(output_path):
            filename = os.path.join(output_path, filename)

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Image downloaded and saved as {filename}")
    else:
        print(f"Failed to download image for UUID {uuid}. Status code: {response.status_code}")

def download_images_from_csv(csv_file_path, output_path, use_uuid):
    with open(csv_file_path, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            uuid = row['task_uuid']
            download_image(uuid, output_path, use_uuid)

def main():
    parser = argparse.ArgumentParser(description='Download images from a CSV file of UUIDs.')
    parser.add_argument('--csv_filepath', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output directory path')
    parser.add_argument('--use_uuid', type=str_to_bool, default=True, help='Use UUID for image filenames (True/False)')

    args = parser.parse_args()

    download_images_from_csv(args.csv_filepath, args.output, args.use_uuid)

if __name__ == "__main__":
    main()
