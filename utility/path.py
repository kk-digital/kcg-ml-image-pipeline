import pathlib


def separate_bucket_and_file_path(path_str):
    p = pathlib.Path(path_str)

    file_path = pathlib.Path(*p.parts[1:])
    bucket = str(p.parts[0])

    return bucket, file_path
