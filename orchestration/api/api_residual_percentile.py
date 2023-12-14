from fastapi import Request, APIRouter, HTTPException
from orchestration.api.mongo_schemas import RankingResidualPercentile

router = APIRouter()


@router.post("/residual-percentile/set-image-rank-residual-percentile", description="Set image rank residual-percentile")
def set_image_rank_residual_percentile(request: Request, ranking_residual_percentile: RankingResidualPercentile):
    # check if exists
    query = {"image_hash": ranking_residual_percentile.image_hash,
             "model_id": ranking_residual_percentile.model_id}
    count = request.app.image_residual_percentiles_collection.count_documents(query)
    if count > 0:
        raise HTTPException(status_code=409, detail="Residual Percentile for specific model_id and image_hash already exists.")

    request.app.image_residual_percentiles_collection.insert_one(ranking_residual_percentile.to_dict())

    return True


@router.get("/residual-percentile/get-image-rank-residual-percentile-by-hash", description="Get image rank residual_percentile by hash")
def get_image_rank_residual_percentile_by_hash(request: Request, image_hash: str, model_id: int):
    # check if exist
    query = {"image_hash": image_hash,
             "model_id": model_id}

    item = request.app.image_residual_percentiles_collection.find_one(query)
    if item is None:
        return None

    # remove the auto generated field
    item.pop('_id', None)

    return item


@router.get("/residual-percentile/get-image-rank-residual-percentiles-by-model-id",
            description="Get image rank residual percentiles by model id. Returns as descending order of residual percentile")
def get_image_rank_residual_percentiles_by_model_id(request: Request, model_id: int):
    # check if exist
    query = {"model_id": model_id}
    items = request.app.image_residual_percentiles_collection.find(query).sort("residual_percentile", -1)
    if items is None:
        return []

    residual_percentile_data = []
    for item in items:
        # remove the auto generated field
        item.pop('_id', None)
        residual_percentile_data.append(item)

    return residual_percentile_data


@router.delete("/residual-percentile/delete-image-rank-residual-percentiles-by-model-id", description="Delete all image rank residual percentiles by model id.")
def delete_image_rank_residual_percentiles_by_model_id(request: Request, model_id: int):
    # check if exist
    query = {"model_id": model_id}
    res = request.app.image_residual_percentiles_collection.delete_many(query)
    print(res.deleted_count, " documents deleted.")

    return None
