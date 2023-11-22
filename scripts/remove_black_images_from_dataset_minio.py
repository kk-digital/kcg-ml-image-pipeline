import os
import sys
import argparse
from tqdm import tqdm
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from worker.http import request
from utility.minio import cmd
from data_loader.utils import DATASETS_BUCKET, get_object

black_white_character_colors = [(227232, (255, 255, 255)), (283, (253, 253, 253)), (261, (251, 251, 251)),
                                (158, (249, 249, 249)), (117, (247, 247, 247)), (94, (245, 245, 245)),
                                (73, (243, 243, 243)), (50, (241, 241, 241)), (29, (239, 239, 239)),
                                (24, (237, 237, 237)), (25, (235, 235, 235)), (13, (233, 233, 233)),
                                (6, (231, 231, 231)), (3, (229, 229, 229)), (8, (227, 227, 227)), (5, (225, 225, 225)),
                                (2, (223, 223, 223)), (1, (219, 219, 219)), (3, (217, 217, 217)), (1, (215, 215, 215)),
                                (2, (213, 213, 213)), (1, (211, 211, 211)), (1, (205, 205, 205)), (1, (201, 201, 201)),
                                (1, (195, 195, 195)), (2, (193, 193, 193)), (3, (189, 189, 189)), (1, (187, 187, 187)),
                                (1, (177, 177, 177)), (1, (173, 173, 173)), (2, (171, 171, 171)), (1, (169, 169, 169)),
                                (1, (167, 167, 167)), (1, (163, 163, 163)), (1, (161, 161, 161)), (1, (157, 157, 157)),
                                (1, (155, 155, 155)), (1, (151, 151, 151)), (1, (147, 147, 147)), (1, (145, 145, 145)),
                                (1, (141, 141, 141)), (1, (137, 137, 137)), (1, (135, 135, 135)), (2, (133, 133, 133)),
                                (1, (129, 129, 129)), (3, (127, 127, 127)), (1, (125, 125, 125)), (2, (117, 117, 117)),
                                (2, (113, 113, 113)), (1, (109, 109, 109)), (1, (107, 107, 107)), (1, (105, 105, 105)),
                                (2, (101, 101, 101)), (2, (95, 95, 95)), (5, (89, 89, 89)), (2, (85, 85, 85)),
                                (1, (77, 77, 77)), (1, (75, 75, 75)), (2, (73, 73, 73)), (1, (63, 63, 63)),
                                (2, (55, 55, 55)), (1, (53, 53, 53)), (1, (51, 51, 51)), (2, (49, 49, 49)),
                                (1, (47, 47, 47)), (1, (45, 45, 45)), (1, (41, 41, 41)), (3, (37, 37, 37)),
                                (2, (35, 35, 35)), (4, (29, 29, 29)), (4, (27, 27, 27)), (13, (25, 25, 25)),
                                (11, (23, 23, 23)), (19, (21, 21, 21)), (28, (19, 19, 19)), (33, (17, 17, 17)),
                                (47, (15, 15, 15)), (76, (13, 13, 13)), (103, (11, 11, 11)), (115, (9, 9, 9)),
                                (161, (7, 7, 7)), (204, (5, 5, 5)), (278, (3, 3, 3)), (370, (1, 1, 1)),
                                (326, (254, 254, 254)), (241, (252, 252, 252)), (173, (250, 250, 250)),
                                (142, (248, 248, 248)), (126, (246, 246, 246)), (105, (244, 244, 244)),
                                (59, (242, 242, 242)), (55, (240, 240, 240)), (25, (238, 238, 238)),
                                (18, (236, 236, 236)), (23, (234, 234, 234)), (13, (232, 232, 232)),
                                (6, (230, 230, 230)), (8, (228, 228, 228)), (4, (226, 226, 226)), (3, (224, 224, 224)),
                                (3, (220, 220, 220)), (2, (218, 218, 218)), (3, (216, 216, 216)), (1, (214, 214, 214)),
                                (1, (212, 212, 212)), (1, (206, 206, 206)), (2, (204, 204, 204)), (1, (202, 202, 202)),
                                (1, (196, 196, 196)), (2, (192, 192, 192)), (1, (188, 188, 188)), (4, (186, 186, 186)),
                                (2, (184, 184, 184)), (1, (182, 182, 182)), (2, (176, 176, 176)), (1, (170, 170, 170)),
                                (1, (162, 162, 162)), (2, (154, 154, 154)), (1, (152, 152, 152)), (2, (148, 148, 148)),
                                (2, (136, 136, 136)), (2, (130, 130, 130)), (3, (126, 126, 126)), (3, (120, 120, 120)),
                                (1, (118, 118, 118)), (1, (116, 116, 116)), (1, (110, 110, 110)), (1, (108, 108, 108)),
                                (1, (106, 106, 106)), (1, (104, 104, 104)), (2, (102, 102, 102)), (1, (100, 100, 100)),
                                (1, (94, 94, 94)), (2, (92, 92, 92)), (2, (90, 90, 90)), (1, (88, 88, 88)),
                                (4, (86, 86, 86)), (1, (84, 84, 84)), (1, (80, 80, 80)), (1, (78, 78, 78)),
                                (1, (76, 76, 76)), (2, (74, 74, 74)), (1, (72, 72, 72)), (1, (70, 70, 70)),
                                (2, (68, 68, 68)), (1, (66, 66, 66)), (3, (62, 62, 62)), (1, (60, 60, 60)),
                                (1, (54, 54, 54)), (1, (52, 52, 52)), (1, (50, 50, 50)), (2, (48, 48, 48)),
                                (3, (40, 40, 40)), (6, (38, 38, 38)), (3, (36, 36, 36)), (2, (34, 34, 34)),
                                (5, (30, 30, 30)), (7, (28, 28, 28)), (3, (26, 26, 26)), (6, (24, 24, 24)),
                                (14, (22, 22, 22)), (19, (20, 20, 20)), (27, (18, 18, 18)), (38, (16, 16, 16)),
                                (62, (14, 14, 14)), (95, (12, 12, 12)), (88, (10, 10, 10)), (143, (8, 8, 8)),
                                (187, (6, 6, 6)), (269, (4, 4, 4)), (304, (2, 2, 2)), (29527, (0, 0, 0))]


def get_images_file_paths(minio_client, dataset_name):
    print("Getting dataset paths...")
    prefix = dataset_name
    dataset_paths = cmd.get_list_of_objects_with_prefix(minio_client, "datasets", prefix=prefix)

    # filter out non image files
    image_file_extension = ".jpg"
    image_paths = []
    print("Getting only image file paths...")
    for path in tqdm(dataset_paths):
        if path.endswith(image_file_extension):
            image_paths.append(path)

    return image_paths


def check_if_exist_then_delete(minio_client, path):
    if cmd.is_object_exists(minio_client, "datasets", path):
        cmd.remove_an_object(minio_client, "datasets", path)

def delete_image(minio_client, image_path):
    print(f"Deleting {image_path}...")

    # delete image
    check_if_exist_then_delete(minio_client, image_path)
    # delete clip
    clip_path = image_path.replace(".jpg", "_clip.msgpack")
    check_if_exist_then_delete(minio_client, clip_path)

    # delete data
    data_path = image_path.replace(".jpg", "_data.msgpack")
    check_if_exist_then_delete(minio_client, data_path)

    # delete embedding
    embedding_path = image_path.replace(".jpg", "_embedding.msgpack")
    check_if_exist_then_delete(minio_client, embedding_path)

    # /embeddings
    new_path = image_path.replace("character", "character/embeddings/text-embedding")

    # embedding 2
    embedding_path_2 = new_path.replace(".jpg", "-text-embedding.msgpack")
    check_if_exist_then_delete(minio_client, embedding_path_2)

    average_pooled_embedding_path = new_path.replace(".jpg", "-text-embedding-average-pooled.msgpack")
    check_if_exist_then_delete(minio_client, average_pooled_embedding_path)

    max_pooled_embedding_path = new_path.replace(".jpg", "-text-embedding-max-pooled.msgpack")
    check_if_exist_then_delete(minio_client, max_pooled_embedding_path)

    signed_max_pooled_embedding_path = new_path.replace(".jpg", "-text-embedding-signed-max-pooled.msgpack")
    check_if_exist_then_delete(minio_client, signed_max_pooled_embedding_path)


def is_black(minio_client, image_path, dataset_name):
    image_bytes = get_object(minio_client, image_path)
    img = Image.open(io.BytesIO(image_bytes))

    colors = img.getcolors(maxcolors=256)

    black_white = [(0, 0, 0), (255, 255, 255)]
    # only black or white
    if len(colors) <= 1:
        return True

    if len(colors) == 2 and colors[0][1] in black_white and colors[1][1] in black_white:
        return True

    black_count = 0
    white_count = 0
    other_color_count = 0
    for i in range(len(colors)):
        if colors[i][1] == (0, 0, 0):
            black_count = colors[i][0]
        elif colors[i][1] == (255, 255, 255):
            white_count = colors[i][0]
        # special case
        elif dataset_name == "icons" and (colors[i][1][0] >= 244 and colors[i][1][1] >= 244 and colors[i][1][2] >= 244):
            white_count += colors[i][0]
        elif dataset_name == "character" and colors == black_white_character_colors:
            return True
        else:
            other_color_count += colors[i][0]

    total_count = white_count + black_count + other_color_count
    white_percentage = white_count / total_count
    black_percentage = black_count / total_count
    if white_percentage + black_percentage == 1.0:
        return True

    return False


def process_image(minio_client, image_path, dataset_name):
    if is_black(minio_client, image_path, dataset_name):
        delete_image(minio_client, image_path)


def process_dataset(minio_client,
                    dataset_name):
    image_paths = get_images_file_paths(minio_client, dataset_name)

    print("Processing dataset...")
    # use multiprocessing
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for path in image_paths:
            futures.append(executor.submit(process_image,
                                           minio_client=minio_client,
                                           image_path=path,
                                           dataset_name=dataset_name))

        for _ in tqdm(as_completed(futures), total=len(futures)):
            continue



def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Remove black images from dataset")

    parser.add_argument('--minio-access-key', type=str, help='Minio access key')
    parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
    parser.add_argument('--dataset-name', type=str,
                        help="The dataset name to process, use 'all' to process all datasets",
                        default='environmental')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    dataset_name = args.dataset_name
    # get minio client
    minio_client = cmd.get_minio_client(minio_ip_addr=None,  # will use default if none is given
                                        minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key)

    if dataset_name != "all":
        process_dataset(minio_client=minio_client,
                        dataset_name=dataset_name)
    else:
        # if all, run script for all existing datasets
        # get dataset name list
        dataset_names = request.http_get_dataset_names()
        print("dataset names=", dataset_names)
        for dataset in dataset_names:
            try:
                print("Running script for {}...".format(dataset))
                process_dataset(minio_client=minio_client,
                                dataset_name=dataset)
            except Exception as e:
                print("Error running script for {}: {}".format(dataset, e))
