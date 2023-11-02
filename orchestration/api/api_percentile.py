from fastapi import Request, APIRouter, HTTPException
from orchestration.api.mongo_schemas import RankingPercentile

router = APIRouter()


@router.post("/percentile/set-image-rank-percentile", description="Set image rank percentile")
def set_image_rank_percentile(request: Request, ranking_percentile: RankingPercentile):
    # check if exists
    query = {"image_hash": ranking_percentile.image_hash,
             "model_id": ranking_percentile.model_id}
    count = request.app.image_percentiles_collection.count_documents(query)
    if count > 0:
        raise HTTPException(status_code=409, detail="Score for specific model_id and image_hash already exists.")

    request.app.image_percentiles_collection.insert_one(ranking_percentile.to_dict())

    return True


@router.get("/percentile/get-image-rank-percentile-by-hash", description="Get image rank percentile by hash")
def get_image_rank_percentile_by_hash(request: Request, image_hash: str, model_id: int):
    # check if exist
    query = {"image_hash": image_hash,
             "model_id": model_id}

    item = request.app.image_percentiles_collection.find_one(query)
    if item is None:
        raise HTTPException(status_code=404, detail="Image rank percentile data not found")

    # remove the auto generated field
    item.pop('_id', None)

    return item


@router.get("/percentile/get-image-rank-percentiles-by-model-id",
            description="Get image rank percentiles by model id. Returns as descending order of percentiles")
def get_image_rank_percentiles_by_model_id(request: Request, model_id: int):
    # check if exist
    query = {"model_id": model_id}
    items = request.app.image_percentiles_collection.find(query).sort("percentile", -1)
    if items is None:
        raise HTTPException(status_code=404, detail="Image rank percentiles data not found")

    percentile_data = []
    for item in items:
        # remove the auto generated field
        item.pop('_id', None)
        percentile_data.append(item)

    return percentile_data
