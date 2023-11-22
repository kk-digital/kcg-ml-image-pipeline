import os
import sys
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
base_directory = "./"
sys.path.insert(0, base_directory)

from scripts.image_scorer import ImageScorer
from training_worker.http import request
from utility.minio import cmd


# delta score = text embedding sigma score - clip embedding sigma score
def get_delta_score(clip_hash_sigma_score_dict,
                    embedding_hash_sigma_score_dict):
    hash_delta_score_dict = {}
    for img_hash, clip_sigma_score in clip_hash_sigma_score_dict.items():
        delta_score = embedding_hash_sigma_score_dict[img_hash] - clip_sigma_score
        hash_delta_score_dict[img_hash] = delta_score

    return hash_delta_score_dict


def upload_scores_attributes_to_completed_jobs(clip_hash_score_pairs,
                                               clip_hash_sigma_score_dict,
                                               embedding_hash_score_pairs,
                                               embedding_hash_sigma_score_dict,
                                               hash_delta_score_dict):
    print("Uploading scores, sigma scores, and delta scores to mongodb...")
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = []
        for i in range(len(clip_hash_score_pairs)):
            img_hash = clip_hash_score_pairs[i][0]
            clip_score =  clip_hash_score_pairs[i][1]
            clip_sigma_score = clip_hash_sigma_score_dict[img_hash]

            if img_hash != embedding_hash_score_pairs[i][0]:
                print("Skipping due to inconsistent image hashes...")
                continue

            embedding_score =  embedding_hash_score_pairs[i][1]
            embedding_sigma_score = embedding_hash_sigma_score_dict[img_hash]
            delta_score = hash_delta_score_dict[img_hash]

            futures.append(executor.submit(request.http_add_score_attributes,
                                           img_hash=img_hash,
                                           clip_score=clip_score,
                                           clip_sigma_score=clip_sigma_score,
                                           embedding_score=embedding_score,
                                           embedding_sigma_score=embedding_sigma_score,
                                           delta_score=delta_score))

        for _ in tqdm(as_completed(futures), total=len(futures)):
            continue



def run_image_delta_scorer(minio_client,
                           dataset_name,
                           clip_model_filename,
                           embedding_model_filename):

    start_time = time.time()

    # clip
    clip_scorer = ImageScorer(minio_client=minio_client,
                              dataset_name=dataset_name,
                              model_name=clip_model_filename)

    clip_scorer.load_model()
    clip_paths = clip_scorer.get_paths()
    clip_hash_score_pairs, image_paths = clip_scorer.get_scores(clip_paths)
    clip_hash_sigma_score_dict = clip_scorer.get_sigma_scores(clip_hash_score_pairs)


    # embedding
    embedding_scorer = ImageScorer(minio_client=minio_client,
                                   dataset_name=dataset_name,
                                   model_name=embedding_model_filename)

    embedding_scorer.load_model()
    embedding_paths = embedding_scorer.get_paths()
    embedding_hash_score_pairs, image_paths = embedding_scorer.get_scores(embedding_paths)
    embedding_hash_sigma_score_dict = embedding_scorer.get_sigma_scores(embedding_hash_score_pairs)

    hash_delta_score_dict = get_delta_score(clip_hash_sigma_score_dict=clip_hash_sigma_score_dict,
                                            embedding_hash_sigma_score_dict=embedding_hash_sigma_score_dict)

    upload_scores_attributes_to_completed_jobs(clip_hash_score_pairs=clip_hash_score_pairs,
                                               clip_hash_sigma_score_dict=clip_hash_sigma_score_dict,
                                               embedding_hash_score_pairs=embedding_hash_score_pairs,
                                               embedding_hash_sigma_score_dict=embedding_hash_sigma_score_dict,
                                               hash_delta_score_dict=hash_delta_score_dict)

    time_elapsed = time.time() - start_time
    print("Dataset: {}: Total Time elapsed: {}s".format(dataset_name, format(time_elapsed, ".2f")))


def parse_args():
    parser = argparse.ArgumentParser(description="Image Delta Scorer")
    parser.add_argument('--minio-addr', required=False, help='Minio server address', default="192.168.3.5:9000")
    parser.add_argument('--minio-access-key', required=False, help='Minio access key')
    parser.add_argument('--minio-secret-key', required=False, help='Minio secret key')
    parser.add_argument('--dataset-name', required=True, help='Name of the dataset for embeddings')
    parser.add_argument('--clip-model-filename', required=True, help='Filename of the clip model (e.g., "XXX.pth")')
    parser.add_argument('--embedding-model-filename', required=True, help='Filename of the embedding model (e.g., "XXX.pth")')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    dataset_name = args.dataset_name
    minio_client = cmd.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key,
                                        minio_ip_addr=args.minio_addr)
    if dataset_name != "all":
        run_image_delta_scorer(minio_client,
                               args.dataset_name,
                               args.clip_model_filename,
                               args.embedding_model_filename)
    else:
        # if all, train models for all existing datasets
        # get dataset name list
        dataset_names = request.http_get_dataset_names()
        print("dataset names=", dataset_names)
        for dataset in dataset_names:
            try:
                run_image_delta_scorer(minio_client,
                                       dataset,
                                       args.clip_model_filename,
                                       args.embedding_model_filename)
            except Exception as e:
                print("Error running image scorer for {}: {}".format(dataset, e))


if __name__ == "__main__":
    main()
