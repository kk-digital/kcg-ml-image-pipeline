from fastapi import Request, APIRouter, HTTPException
from orchestration.api.mongo_schemas import RankingSigmaScore

router = APIRouter()


@router.post("/sigma-score/set-image-rank-sigma-score", description="Set image rank sigma_score")
def set_image_rank_sigma_score(request: Request, ranking_sigma_score: RankingSigmaScore):
    # check if exists
    query = {"image_hash": ranking_sigma_score.image_hash,
             "model_id": ranking_sigma_score.model_id}
    count = request.app.image_sigma_scores_collection.count_documents(query)
    if count > 0:
        raise HTTPException(status_code=409, detail="Score for specific model_id and image_hash already exists.")

    request.app.image_sigma_scores_collection.insert_one(ranking_sigma_score.to_dict())

    return True


@router.get("/sigma-score/get-image-rank-sigma-score-by-hash", description="Get image rank sigma_score by hash")
def get_image_rank_sigma_score_by_hash(request: Request, image_hash: str, model_id: int):
    # check if exist
    query = {"image_hash": image_hash,
             "model_id": model_id}

    item = request.app.image_sigma_scores_collection.find_one(query)
    if item is None:
        print("Image rank sigma_score data not found")

    # remove the auto generated field
    item.pop('_id', None)

    return item


@router.get("/sigma-score/get-image-rank-sigma-scores-by-model-id",
            description="Get image rank sigma_scores by model id. Returns as descending order of sigma_scores")
def get_image_rank_sigma_scores_by_model_id(request: Request, model_id: int):
    # check if exist
    query = {"model_id": model_id}
    items = request.app.image_sigma_scores_collection.find(query).sort("sigma_score", -1)
    if items is None:
        print("Image rank sigma_scores data not found")

    sigma_score_data = []
    for item in items:
        # remove the auto generated field
        item.pop('_id', None)
        sigma_score_data.append(item)

    return sigma_score_data


@router.delete("/sigma-score/delete-image-rank-sigma-scores-by-model-id", description="Delete all image rank sigma_scores by model id.")
def delete_image_rank_sigma_scores_by_model_id(request: Request, model_id: int):
    # check if exist
    query = {"model_id": model_id}
    res = request.app.image_sigma_scores_collection.delete_many(query)
    print(res.deleted_count, " documents deleted.")

    return None
