import sys
import msgpack

base_directory = "./"
sys.path.insert(0, base_directory)

from utility.minio.cmd import get_file_from_minio, is_object_exists


def get_image_clip_from_minio(minio_client, image_path, bucket_name):
    # Removes the last 4 characters from the path
    # image.jpg => image
    base_path = image_path.rstrip(image_path[-4:])

    # finds the clip file associated with the image
    # example image => image_clip.msgpack
    image_clip_vector_path = f'{base_path}_clip.msgpack'

    print(f'image clip vector path : {image_clip_vector_path}')
    # get the clip.msgpack from minio
    file_exists = is_object_exists(minio_client, bucket_name, image_clip_vector_path)

    if not file_exists:
        print(f'{image_clip_vector_path} does not exist')
        return None

    clip_vector_data_msgpack = get_file_from_minio(minio_client, bucket_name, image_clip_vector_path)

    if clip_vector_data_msgpack is None:
        print(f'image not found {image_path}')
        return None

    # read file_data_into memory
    clip_vector_data_msgpack_memory = clip_vector_data_msgpack.read()

    try:
        # uncompress the msgpack data
        clip_vector = msgpack.unpackb(clip_vector_data_msgpack_memory)
        clip_vector = clip_vector["clip-feature-vector"]

        return clip_vector
    except Exception as e:
        print('Exception details : ', e)

    return None