import csv
import os
import sys
import requests
from urllib.parse import unquote

def download_image(uuid, output_path):
    url = f"http://localhost:8000/get-image-by-job-uuid/{uuid}"
    response = requests.get(url, stream=True)

    if response.status_code == 200:
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

def download_images_from_csv(csv_file_path, output_path):
    with open(csv_file_path, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            uuid = row['task_uuid']
            download_image(uuid, output_path)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python download_images.py <csv_file_path> <output_path>")
        sys.exit(1)

    csv_file_path = sys.argv[1]
    output_path = sys.argv[2]
    download_images_from_csv(csv_file_path, output_path)


