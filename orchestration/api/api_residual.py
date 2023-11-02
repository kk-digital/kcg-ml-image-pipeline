from fastapi import Request, APIRouter, HTTPException
from orchestration.api.mongo_schemas import RankingResidual

router = APIRouter()


@router.post("/residual/set-image-rank-residual", description="Set image rank residual")
def set_image_rank_residual(request: Request, ranking_residual: RankingResidual):
    request.app.image_residuals_collection.insert_one(ranking_residual.to_dict())

    return True


@router.get("/residual/get-image-rank-residual-by-hash", description="Get image rank residual by hash")
def get_image_rank_residual_by_hash(request: Request, image_hash: str, model_id: int):
    # check if exist
    query = {"$match":
                 {"image_hash": image_hash,
                  "model_id": model_id}}

    item = request.app.image_residuals_collection.find_one(query)
    if item is None:
        raise HTTPException(status_code=404, detail="Image rank residual data not found")

    # remove the auto generated field
    item.pop('_id', None)

    return item


@router.get("/residual/get-image-rank-residuals-by-model-id",
            description="Get image rank residuals by model id. Returns as descending order of residual")
def get_image_rank_residuals_by_model_id(request: Request, model_id: int):
    # check if exist
    query = {"model_id": model_id}
    items = request.app.image_residuals_collection.find(query).sort("residual", -1)
    if items is None:
        raise HTTPException(status_code=404, detail="Image rank residuals data not found")

    residual_data = []
    for item in items:
        # remove the auto generated field
        item.pop('_id', None)
        residual_data.append(item)

    return residual_data
