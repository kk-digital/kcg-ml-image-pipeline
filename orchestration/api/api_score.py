from fastapi import Request, APIRouter, HTTPException
from orchestration.api.mongo_schemas import RankingScore

router = APIRouter()


@router.post("/score/set-image-rank-score", description="Set image rank score")
def set_image_rank_score(request: Request, ranking_score: RankingScore):
    # check if exists
    query = {"image_hash": ranking_score.image_hash,
             "model_id": ranking_score.model_id}
    count = request.app.image_scores_collection.count_documents(query)
    if count > 0:
        raise HTTPException(status_code=409, detail="Score for specific model_id and image_hash already exists.")

    request.app.image_scores_collection.insert_one(ranking_score.to_dict())

    return True


@router.get("/score/get-image-rank-score-by-hash", description="Get image rank score by hash")
def get_image_rank_score_by_hash(request: Request, image_hash: str, model_id: int):
    # check if exist
    query = {"image_hash": image_hash,
             "model_id": model_id}

    item = request.app.image_scores_collection.find_one(query)
    if item is None:
        raise HTTPException(status_code=404, detail="Image rank score data not found")

    # remove the auto generated field
    item.pop('_id', None)

    return item


@router.get("/score/get-image-rank-scores-by-model-id",
            description="Get image rank scores by model id. Returns as descending order of scores")
def get_image_rank_scores_by_model_id(request: Request, model_id: int):
    # check if exist
    query = {"model_id": model_id}
    items = request.app.image_scores_collection.find(query).sort("score", -1)
    if items is None:
        raise HTTPException(status_code=404, detail="Image rank scores data not found")

    score_data = []
    for item in items:
        # remove the auto generated field
        item.pop('_id', None)
        score_data.append(item)

    return score_data


@router.delete("/score/delete-image-rank-scores-by-model-id", description="Delete all image rank scores by model id.")
def delete_image_rank_scores_by_model_id(request: Request, model_id: int):
    # check if exist
    query = {"model_id": model_id}
    res = request.app.image_scores_collection.delete_many(query)
    print(res.deleted_count, " documents deleted.")

    return None
