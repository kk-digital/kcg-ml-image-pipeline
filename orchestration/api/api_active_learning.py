from fastapi import Request, HTTPException, APIRouter, Response, Query, status
from datetime import datetime, timedelta
import math
import random
import pymongo
from utility.minio import cmd
from .api_utils import PrettyJSONResponse
import os
from fastapi.responses import JSONResponse
from pymongo.collection import Collection


router = APIRouter()


@router.get("/active-learning/uncertainty-sampling-pair-v1", response_class=PrettyJSONResponse)
def get_ranking_comparison(
    request: Request,
    dataset: str,  # Added dataset parameter
    score_type: str,  # Added score_type parameter to choose between clip_sigma_score and embedding_sigma_score
    min_score: float,
    max_score: float,
    threshold: float
):
    if score_type not in ["clip_sigma_score", "embedding_sigma_score"]:
        raise HTTPException(status_code=400, detail="Invalid score_type parameter")

    image_scores_collection: Collection = request.app.image_scores_collection

    try:
        # Fetch a random image score within the score range and the specified dataset
        first_image_cursor = image_scores_collection.aggregate([
            {"$match": {
                "score": {"$gte": min_score, "$lte": max_score},
                "dataset": dataset  # Filter by dataset
            }},
            {"$sample": {"size": 1}}
        ])
        first_image_score = next(first_image_cursor, None)

        if not first_image_score:
            print("No images found within the provided score range and dataset")

        # Calculate the score range for the second image using the selected score_type
        base_score = first_image_score[score_type]  # Use dynamic score_type

        # Fetch candidate images for the second image within the specified dataset
        candidates_cursor = image_scores_collection.find({
            score_type: {"$gte": min_score, "$lte": max_score},
            "image_hash": {"$ne": first_image_score['image_hash']},
            "dataset": dataset  # Filter by dataset
        })

        # Compute probabilities using sigmoid function based on the score_type
        candidates = list(candidates_cursor)
        total_probability = 0
        for candidate in candidates:
            score_diff = abs(candidate[score_type] - base_score)  # Use dynamic score_type
            probability = 1 / (1 + math.exp((score_diff - threshold) / 50))
            candidate['probability'] = probability
            total_probability += probability

        # Select the second image based on computed probabilities
        if total_probability == 0:
            print("No second image found within the threshold range for the specified dataset")

        random_choice = random.uniform(0, total_probability)
        cumulative = 0
        for candidate in candidates:
            cumulative += candidate['probability']
            if cumulative >= random_choice:
                second_image_score = candidate
                break

    except StopIteration:
        return JSONResponse(
            status_code=500,
            content={"message": "Error fetching images from the database."}
        )

    # Prepare the images for the response
    images = [
        {
            "image_hash": first_image_score['image_hash'],
            "image_score": first_image_score[score_type]  # Use dynamic score_type
        },
        {
            "image_hash": second_image_score['image_hash'],
            "image_score": second_image_score[score_type]  # Use dynamic score_type
        }
    ]

    return {"images": images}


@router.get("/active-learning/uncertainty-sampling-pair-v2", response_class=PrettyJSONResponse)
def get_ranking_comparison(
    request: Request,
    dataset: str,  
    score_type: str,  # Added score_type parameter to choose between clip_sigma_score and embedding_sigma_score
    min_score: float,
    max_score: float,
    threshold: float
):
    if score_type not in ["clip_sigma_score", "embedding_sigma_score"]:
        raise HTTPException(status_code=400, detail="Invalid score_type parameter")

    completed_jobs_collection: Collection = request.app.completed_jobs_collection

    try:
        # Convert min_score and max_score to strings for the query
        min_score_str = str(min_score)
        max_score_str = str(max_score)

        first_image_cursor = completed_jobs_collection.aggregate([
            {"$match": {
                "task_attributes_dict." + score_type: {"$gte": min_score_str, "$lte": max_score_str},
                "task_input_dict.dataset": dataset  # Filter by dataset
            }},
            {"$sample": {"size": 1}}
        ])

        first_image_score = next(first_image_cursor, None)
        if not first_image_score:
            print("No images found within the provided score range and dataset.")

        # Check if 'task_attributes_dict' exists in the fetched document
        if 'task_attributes_dict' not in first_image_score or score_type not in first_image_score['task_attributes_dict']:
           print("Error: 'task_attributes_dict' not found or missing score_type in the fetched document.")

        base_score = first_image_score['task_attributes_dict'][score_type]
       
        candidates_cursor = completed_jobs_collection.find({
            "task_attributes_dict." + score_type: {"$gte": str(min_score), "$lte": str(max_score)},
            "task_output_file_dict.output_file_hash": {"$ne": first_image_score['task_output_file_dict']['output_file_hash']},
            "task_input_dict.dataset": dataset
        })

        candidates = list(candidates_cursor)

        if not candidates:
            print("No candidates found.")  # Debug print

        total_probability = 0
        for candidate in candidates:
            candidate_score = float(candidate['task_attributes_dict'][score_type])
            base_score_float = float(base_score)
            score_diff = abs(candidate_score - base_score_float)
            probability = 1 / (1 + math.exp((score_diff - threshold) / 50))
            candidate['probability'] = probability
            total_probability += probability

        print(f"Total probability: {total_probability}")  # Debug print

        if total_probability == 0:
             print("Probability = 0, No suitable second image found within the specified score range and dataset.")

            
        random_choice = random.uniform(0, total_probability)
        cumulative = 0
        second_image_score = None
        for candidate in candidates:
            cumulative += candidate['probability']
            if cumulative >= random_choice:
                second_image_score = candidate
                break

        if not second_image_score:
            print("Error: No suitable second image found.")

        images = [
            {
                "image_hash": first_image_score['task_output_file_dict']['output_file_hash'],
                "image_score": first_image_score['task_attributes_dict'][score_type]
            },
            {
                "image_hash": second_image_score['task_output_file_dict']['output_file_hash'],
                "image_score": second_image_score['task_attributes_dict'][score_type]
            }
        ]

        return {"images": images}

    except StopIteration:
        return JSONResponse(status_code=500, content={"message": "Error fetching images from the database."})
