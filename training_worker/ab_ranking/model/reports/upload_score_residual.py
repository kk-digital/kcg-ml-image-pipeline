import os
import sys
from tqdm import tqdm
base_directory = os.getcwd()
sys.path.insert(0, base_directory)

from training_worker.http import request


def add_model_card(model_card):
    model_id = request.http_add_model(model_card)

    return int(model_id)


def upload_score_residual(model_id: int,
                          train_prob_predictions,
                          training_targets,
                          validation_prob_predictions,
                          validation_targets,
                          training_pred_scores_img_x,
                          validation_pred_scores_img_x,
                          training_image_hashes,
                          validation_image_hashes):
    print("Uploading scores and residuals...")

    print("From training datapoints...")
    # training
    for i in tqdm(range(len(training_targets))):
        if training_targets[i] == [1.0]:
            img_hash = training_image_hashes[i]
            img_score = training_pred_scores_img_x[i].item()
            img_residual = abs(1.0 - train_prob_predictions[i].item())

            # upload score
            score_data = {
                "model_id": model_id,
                "image_hash": img_hash,
                "score": img_score,
            }
            request.http_add_score(score_data)

            # upload residual
            residual_data = {
                "model_id": model_id,
                "image_hash": img_hash,
                "residual": img_residual,
            }
            request.http_add_residual(residual_data)

    print("From validation datapoints...")
    # validation
    for i in tqdm(range(len(validation_targets))):
        if validation_targets[i] == [1.0]:
            img_hash = validation_image_hashes[i]
            img_score = validation_pred_scores_img_x[i].item()
            img_residual = abs(1.0 - validation_prob_predictions[i].item())

            # upload score
            # upload score
            score_data = {
                "model_id": model_id,
                "image_hash": img_hash,
                "score": img_score,
            }
            request.http_add_score(score_data)

            # upload residual
            residual_data = {
                "model_id": model_id,
                "image_hash": img_hash,
                "residual": img_residual,
            }
            request.http_add_residual(residual_data)

