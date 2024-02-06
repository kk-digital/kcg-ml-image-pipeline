import os
import gc
import libtorrent as lt
import time
import zipfile
from minio import Minio
from pathlib import Path

# Initialize directories
TARGET_DIR = Path('downloaded_files')
TARGET_DIR.mkdir(exist_ok=True)

# Torrent Download Function
def download_from_torrent(torrent_file, save_path):
    ses = lt.session()
    ses.listen_on(6881, 6891)
    params = {
        'save_path': save_path,
        'storage_mode': lt.storage_mode_t.storage_mode_sparse,
        'ti': lt.torrent_info(torrent_file)
    }
    handle = ses.add_torrent(params)

    print('Downloading', torrent_file)
    while not handle.is_seed():
        s = handle.status()
        print(f'{s.progress * 100:.2f}% complete (down: {s.download_rate / 1000:.2f} kB/s up: {s.upload_rate / 1000:.2f} kB/s peers: {s.num_peers})')
        time.sleep(5)

    print("Download complete")

# Unzip Function
def unzip_file(zip_file_path, extract_to_folder):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_folder)
    print(f"Extracted files to {extract_to_folder}")

# MinIO Upload Function
def upload_directory_to_minio(client, bucket_name, minio_folder, local_folder):
    for root, dirs, files in os.walk(local_folder):
        for filename in files:
            file_path = os.path.join(root, filename)
            object_name = os.path.join(minio_folder, os.path.relpath(file_path, local_folder))
            client.fput_object(
                bucket_name, object_name, file_path,
            )
            print(f"'{file_path}' is successfully uploaded as object '{object_name}' to bucket '{bucket_name}'.")

# Main function
def process_torrent_to_minio(torrent_file_path, minio_endpoint, access_key, secret_key, bucket_name, minio_folder):
    # Download from torrent
    download_folder = TARGET_DIR / 'torrent_download'
    download_folder.mkdir(exist_ok=True)
    download_from_torrent(torrent_file_path, str(download_folder))
    
    # Unzip if it's a ZIP file
    for item in os.listdir(download_folder):
        if item.endswith('.zip'):
            unzip_file_path = download_folder / item
            extract_to_folder = download_folder / 'unzipped_files'
            extract_to_folder.mkdir(exist_ok=True)
            unzip_file(unzip_file_path, str(extract_to_folder))
            upload_folder = extract_to_folder
            break
    else:
        upload_folder = download_folder
    
    # Upload to MinIO
    client = Minio(
        minio_endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=True
    )
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)
    else:
        print(f"Bucket '{bucket_name}' already exists")
    
    upload_directory_to_minio(client, bucket_name, minio_folder, str(upload_folder))
    print("Upload to MinIO complete")

# Usage
torrent_file_path = 'input/kandinsky-2-2.zip.torrent'  # Path to your .torrent file
minio_endpoint = "192.168.3.5:9000"
access_key = "v048BpXpWrsVIHUfdAix"
secret_key = "4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu"
bucket_name = 'models'
minio_folder = 'kandinsky'  # Folder path in MinIO
process_torrent_to_minio(torrent_file_path, minio_endpoint, access_key, secret_key, bucket_name, minio_folder)