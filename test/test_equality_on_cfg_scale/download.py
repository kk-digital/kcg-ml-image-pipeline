import csv
import os
import requests
import argparse
import tempfile
import time
import pandas as pd

def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_output_file_path(uuid):
    # Update the URL to match your new API endpoint
    url = f"http://103.20.60.90:8764/job/get-job/{uuid}"
    response = requests.get(url)
    if response.status_code == 200:
        job_data = response.json()
        return job_data.get("task_output_file_dict", {}).get("output_file_path")
    else:
        print(f"Failed to fetch job details for UUID {uuid}. Status code: {response.status_code}")
        return None

def download_image(uuid, output_path, downloaded_uuids_path, task_cfg_scale, seed):
    start_time = time.time()

    output_file_path = get_output_file_path(uuid)
    if not output_file_path:
        return None, None

    filename, extension = os.path.splitext(output_file_path)

    new_filename = f"cfg_scale_{task_cfg_scale}_seed_{seed}{extension}"

    full_local_path = os.path.join(output_path, new_filename)

    url = f"http://103.20.60.90:8764/get-image-by-job-uuid/{uuid}"
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(full_local_path, 'w+b') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Image downloaded and saved as {full_local_path}")

        with open(downloaded_uuids_path, 'a') as file:
            file.write(uuid + '\n')
        
        end_time = time.time()
        return end_time - start_time, full_local_path
    elif response.status_code == 404:
        with open(downloaded_uuids_path, 'a') as file:
            file.write(uuid + '\n')
        
        return None, full_local_path
    else:
        print(f"Failed to download image for UUID {uuid}. Status code: {response.status_code}")
        return None, full_local_path

def download_images_from_csv(csv_file_path, output_path):
    downloaded_uuids_path = os.path.join(output_path, 'downloaded_uuids.txt')
    already_downloaded = set()
    if os.path.exists(downloaded_uuids_path):
        with open(downloaded_uuids_path, 'r') as downloaded_file:
            already_downloaded = set(line.strip() for line in downloaded_file)

    total_time = 0
    num_images_downloaded = 0
    generated_images_data = pd.read_csv(csv_file_path)

    if 'downloaded_image_path' not in generated_images_data.columns:
        generated_images_data['downloaded_image_path'] = None

    for i in range(len(generated_images_data)):
        uuid = generated_images_data.loc[i]['task_uuid']
        task_cfg_scale = generated_images_data.loc[i]['task_cfg_scale']
        seed = generated_images_data.loc[i]['task_seed']
        
        if uuid not in already_downloaded:
            time_taken, saved_image_path = download_image(uuid, output_path, downloaded_uuids_path, task_cfg_scale, seed)
            generated_images_data.loc[i, 'downloaded_image_path'] = saved_image_path
            if time_taken:
                total_time += time_taken
                num_images_downloaded += 1
            generated_images_data.to_csv(csv_file_path, index=False)
                    
    if num_images_downloaded > 0:
        avg_time_per_image = total_time / num_images_downloaded
        print(f"Total download time: {total_time:.2f} seconds")
        print(f"Total images downloaded: {num_images_downloaded}")
        print(f"Average time per image: {avg_time_per_image:.2f} seconds")

    # Delete the downloaded UUIDs file
    if os.path.exists(downloaded_uuids_path):
        os.remove(downloaded_uuids_path)

def main():
    parser = argparse.ArgumentParser(description='Download images from a CSV file of UUIDs.')
    parser.add_argument('--csv_filepath', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output directory path')

    args = parser.parse_args()

    downloaded_uuids_tempfile_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            downloaded_uuids_tempfile_path = tmp_file.name
            download_images_from_csv(args.csv_filepath, args.output)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if downloaded_uuids_tempfile_path:
            os.remove(downloaded_uuids_tempfile_path)

if __name__ == "__main__":
    main()
