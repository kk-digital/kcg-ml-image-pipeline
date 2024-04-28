import time
import pathlib

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

def file_exists(file_path):
    return pathlib.Path(file_path).is_file()

def separate_bucket_and_file_path(path_str):
    p = pathlib.Path(path_str)

    file_path = pathlib.Path(*p.parts[1:])
    file_path = "{}".format(file_path)
    bucket = str(p.parts[0])

    return bucket, file_path
