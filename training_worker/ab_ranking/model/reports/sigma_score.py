import os
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from utility.http import model_training_request
from utility.http import request


def get_mean_and_count(training_targets,
                       validation_targets,
                       training_pred_scores_img_x,
                       training_pred_scores_img_y,
                       validation_pred_scores_img_x,
                       validation_pred_scores_img_y):
    # get mean
    x_sum_score = 0.0
    y_sum_score = 0.0
    count = 0.0
    for i in range(len(training_targets)):
        if training_targets[i] == [1.0]:
            x_sum_score += training_pred_scores_img_x[i].item()
            y_sum_score += training_pred_scores_img_y[i].item()
            count += 1.0

    for i in range(len(validation_targets)):
        if validation_targets[i] == [1.0]:
            x_sum_score += validation_pred_scores_img_x[i].item()
            y_sum_score += validation_pred_scores_img_y[i].item()
            count += 1.0

    x_mean = x_sum_score / count
    y_mean = y_sum_score / count

    return x_mean, y_mean, count


def get_standard_deviation(training_targets,
                           validation_targets,
                           training_pred_scores_img_x,
                           training_pred_scores_img_y,
                           validation_pred_scores_img_x,
                           validation_pred_scores_img_y,
                           x_mean,
                           y_mean,
                           total_count):
    x_sum_squared_diff = 0
    y_sum_squared_diff = 0
    for i in range(len(training_targets)):
        if training_targets[i] == [1.0]:
            x_score = training_pred_scores_img_x[i].item()
            x_diff = x_score - x_mean
            x_squared_diff = x_diff * x_diff
            x_sum_squared_diff += x_squared_diff

            y_score = training_pred_scores_img_y[i].item()
            y_diff = y_score - y_mean
            y_squared_diff = y_diff * y_diff
            y_sum_squared_diff += y_squared_diff

    for i in range(len(validation_targets)):
        if validation_targets[i] == [1.0]:
            x_score = validation_pred_scores_img_x[i].item()
            x_diff = x_score - x_mean
            x_squared_diff = x_diff * x_diff
            x_sum_squared_diff += x_squared_diff

            y_score = validation_pred_scores_img_y[i].item()
            y_diff = y_score - y_mean
            y_squared_diff = y_diff * y_diff
            y_sum_squared_diff += y_squared_diff

    x_variance = x_sum_squared_diff / total_count
    x_standard_deviation = math.sqrt(x_variance)

    y_variance = y_sum_squared_diff / total_count
    y_standard_deviation = math.sqrt(y_variance)

    return x_standard_deviation, y_standard_deviation


def get_chronological_sigma_scores(training_targets,
                                   validation_targets,
                                   training_pred_scores_img_x,
                                   validation_pred_scores_img_x,
                                   training_pred_scores_img_y,
                                   validation_pred_scores_img_y,
                                   training_image_hashes,
                                   validation_image_hashes,
                                   training_shuffled_indices_origin,
                                   validation_shuffled_indices_origin):
    (x_mean,
     y_mean,
     total_count) = get_mean_and_count(training_targets,
                                       validation_targets,
                                       training_pred_scores_img_x,
                                       training_pred_scores_img_y,
                                       validation_pred_scores_img_x,
                                       validation_pred_scores_img_y)

    (x_standard_deviation,
     y_standard_deviation) = get_standard_deviation(training_targets,
                                                    validation_targets,
                                                    training_pred_scores_img_x,
                                                    training_pred_scores_img_y,
                                                    validation_pred_scores_img_x,
                                                    validation_pred_scores_img_y,
                                                    x_mean,
                                                    y_mean,
                                                    total_count)

    x_chronological_sigma_scores = [None] * int(len(training_targets) + len(validation_targets))
    x_chronological_image_hashes = [None] * int(len(training_targets) + len(validation_targets))
    y_chronological_sigma_scores = [None] * int(len(training_targets) + len(validation_targets))

    count = 0
    for i in range(len(training_targets)):
        if training_targets[i] == [1.0]:
            x_img_hash = training_image_hashes[i]
            x_img_score = training_pred_scores_img_x[i].item()
            x_img_sigma_score = (x_img_score - x_mean) / x_standard_deviation

            chronological_index = training_shuffled_indices_origin[count]
            x_chronological_sigma_scores[chronological_index] = x_img_sigma_score
            x_chronological_image_hashes[chronological_index] = x_img_hash

            y_img_score = training_pred_scores_img_y[i].item()
            y_img_sigma_score = (y_img_score - y_mean) / y_standard_deviation
            y_chronological_sigma_scores[chronological_index] = y_img_sigma_score

            count += 1

    count = 0
    for i in range(len(validation_targets)):
        if validation_targets[i] == [1.0]:
            x_img_hash = validation_image_hashes[i]
            x_img_score = validation_pred_scores_img_x[i].item()
            x_img_sigma_score = (x_img_score - x_mean) / x_standard_deviation

            chronological_index = validation_shuffled_indices_origin[count]
            x_chronological_sigma_scores[chronological_index] = x_img_sigma_score
            x_chronological_image_hashes[chronological_index] = x_img_hash

            y_img_score = validation_pred_scores_img_y[i].item()
            y_img_sigma_score = (y_img_score - y_mean) / y_standard_deviation
            y_chronological_sigma_scores[chronological_index] = y_img_sigma_score

            count += 1

    cleaned_x_chronological_sigma_scores = []
    cleaned_x_chronological_image_hashes = []
    cleaned_y_chronological_sigma_scores = []

    # remove none
    for i in range(len(x_chronological_sigma_scores)):
        if x_chronological_sigma_scores[i] is not None:
            cleaned_x_chronological_sigma_scores.append(x_chronological_sigma_scores[i])
            cleaned_x_chronological_image_hashes.append(x_chronological_image_hashes[i])

        if y_chronological_sigma_scores[i] is not None:
            cleaned_y_chronological_sigma_scores.append(y_chronological_sigma_scores[i])

    return cleaned_x_chronological_sigma_scores, cleaned_x_chronological_image_hashes, cleaned_y_chronological_sigma_scores, x_mean, x_standard_deviation


def upload_sigma_score(model_id: int,
                       chronological_sigma_scores,
                       chronological_image_hashes):
    print("Uploading sigma scores...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []

        for i in range(len(chronological_sigma_scores)):
            # upload score
            sigma_score_data = {
                "model_id": model_id,
                "image_hash": chronological_image_hashes[i],
                "sigma_score": chronological_sigma_scores[i],
            }

            futures.append(executor.submit(request.http_add_sigma_score, sigma_score_data=sigma_score_data))

        for _ in tqdm(as_completed(futures), total=len(futures)):
            continue
