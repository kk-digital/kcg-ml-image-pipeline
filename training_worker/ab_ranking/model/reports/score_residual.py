import os
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from utility.http import model_training_request


def add_model_card(model_card):
    model_id = model_training_request.http_add_model(model_card)

    return int(model_id)


def upload_score_residual(model_id: int,
                          train_prob_predictions,
                          training_targets,
                          validation_prob_predictions,
                          validation_targets,
                          training_pred_scores_img_x,
                          validation_pred_scores_img_x,
                          training_image_hashes,
                          validation_image_hashes,
                          training_shuffled_indices_origin,
                          validation_shuffled_indices_origin):
    print("Uploading scores and residuals...")

    # chronological_pred_scores_img_x_target_0 = [None] * int(
    #     (len(training_pred_scores_img_x) + len(validation_pred_scores_img_x)) / 2)
    #
    # count = 0
    # for score in training_pred_scores_img_x:
    #     if training_targets[count] == [0.0]:
    #         chronological_pred_scores_img_x_target_0[training_shuffled_indices_origin[count]] = score.item()
    #     else:
    #         chronological_pred_scores_img_x_target_1[training_shuffled_indices_origin[count]] = score.item()
    #     count += 1
    # count = 0
    #
    # for score in validation_pred_scores_img_x:
    #     if validation_targets[count] == [0.0]:
    #         chronological_pred_scores_img_x_target_0[validation_shuffled_indices_origin[count]] = score.item()
    #     else:
    #         chronological_pred_scores_img_x_target_1[validation_shuffled_indices_origin[count]] = score.item()
    #     count += 1
    chronological_residuals = [None] * int(len(training_targets) + len(validation_targets))
    chronological_image_hashes = [None] * int(len(training_targets) + len(validation_targets))

    print("From training datapoints...")
    # training
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []

        count = 0
        for i in range(len(training_targets)):
            if training_targets[i] == [1.0]:
                img_hash = training_image_hashes[i]
                img_score = training_pred_scores_img_x[i].item()
                img_residual = abs(1.0 - train_prob_predictions[i].item())

                chronological_index = training_shuffled_indices_origin[count]
                chronological_residuals[chronological_index] = img_residual
                chronological_image_hashes[chronological_index] = img_hash

                # upload score
                score_data = {
                    "model_id": model_id,
                    "image_hash": img_hash,
                    "score": img_score,
                }

                # upload residual
                residual_data = {
                    "model_id": model_id,
                    "image_hash": img_hash,
                    "residual": img_residual,
                }

                futures.append(executor.submit(model_training_request.http_add_score, score_data=score_data))
                futures.append(executor.submit(model_training_request.http_add_residual, residual_data=residual_data))
            count += 1

        for _ in tqdm(as_completed(futures), total=len(futures)):
            continue

    print("From validation datapoints...")
    # validation
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []

        count = 0
        for i in tqdm(range(len(validation_targets))):
            if validation_targets[i] == [1.0]:
                img_hash = validation_image_hashes[i]
                img_score = validation_pred_scores_img_x[i].item()
                img_residual = abs(1.0 - validation_prob_predictions[i].item())

                chronological_index = validation_shuffled_indices_origin[count]
                chronological_residuals[chronological_index] = img_residual
                chronological_image_hashes[chronological_index] = img_hash

                # upload score
                score_data = {
                    "model_id": model_id,
                    "image_hash": img_hash,
                    "score": img_score,
                }

                # upload residual
                residual_data = {
                    "model_id": model_id,
                    "image_hash": img_hash,
                    "residual": img_residual,
                }

                futures.append(executor.submit(model_training_request.http_add_score, score_data=score_data))
                futures.append(executor.submit(model_training_request.http_add_residual, residual_data=residual_data))
            count += 1

        for _ in tqdm(as_completed(futures), total=len(futures)):
            continue

    print("Process residual percentiles...")
    hash_residual_pairs = []
    # remove None items
    for i in range(len(chronological_residuals)):
        if chronological_residuals[i] != None:
            hash_residual_pairs.append((chronological_image_hashes[i], chronological_residuals[i]))

    hash_residual_percentile_dict = {}
    hash_residual_pairs.sort(key=lambda a: a[1])

    len_hash_scores = len(hash_residual_pairs)
    for i in range(len_hash_scores):
        percentile = i / len_hash_scores
        hash_residual_percentile_dict[hash_residual_pairs[i][0]] = percentile

    # upload residual percentile
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []

        for img_hash, residual_percentile in hash_residual_percentile_dict.items():
            residual_percentile_data = {
                "model_id": model_id,
                "image_hash": img_hash,
                "residual_percentile": residual_percentile,
            }

            futures.append(executor.submit(model_training_request.http_add_residual_percentile, residual_percentile_data=residual_percentile_data))

        for _ in tqdm(as_completed(futures), total=len(futures)):
            continue
