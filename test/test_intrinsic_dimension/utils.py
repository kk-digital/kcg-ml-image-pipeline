import time
import pathlib
from minio import Minio
import requests

MINIO_ADDRESS = "192.168.3.5:9000"

# define the measure the running time
def measure_running_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return result, elapsed_time

def get_object(client, file_path):
    try:
        response = client.get_object("datasets", file_path)
        data = response.data
    finally:
        if 'response' in locals() and response:
            response.close()
            response.release_conn()

    return data


def separate_bucket_and_file_path(path_str):
    p = pathlib.Path(path_str)

    file_path = pathlib.Path(*p.parts[1:])
    file_path = "{}".format(file_path)
    bucket = str(p.parts[0])

    return bucket, file_path


def get_minio_client(minio_access_key, minio_secret_key, minio_ip_addr=None):
    global MINIO_ADDRESS

    if minio_ip_addr is not None:
        MINIO_ADDRESS = minio_ip_addr
    # check first if minio client is available
    minio_client = None
    while minio_client is None:
        # check minio server
        if is_minio_server_accessible(MINIO_ADDRESS):
            minio_client = connect_to_minio_client(MINIO_ADDRESS, minio_access_key, minio_secret_key)
            return minio_client


def connect_to_minio_client(minio_ip_addr=None, access_key=None, secret_key=None,):
    global MINIO_ADDRESS

    if minio_ip_addr is not None:
        MINIO_ADDRESS = minio_ip_addr

    print("Connecting to minio client...")
    client = Minio(MINIO_ADDRESS, access_key, secret_key, secure=False)
    print("Successfully connected to minio client...")
    return client

def is_minio_server_accessible(address=None):
    if address is None:
        address = MINIO_ADDRESS

    print("Checking if minio server is accessible...")
    try:
        r = requests.head("http://" + address + "/minio/health/live", timeout=5)
    except:
        print("Minio server is not accessible...")
        return False

    return r.status_code == 200