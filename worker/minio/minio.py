
from utility.minio import cmd

def get_minio_client(minio_access_key, minio_secret_key):
    # check first if minio client is available
    minio_client = None
    while minio_client is None:
        # check minio server
        if cmd.is_minio_server_accesssible():
            minio_client = cmd.connect_to_minio_client(minio_access_key, minio_secret_key)
            return minio_client