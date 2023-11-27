from fastapi import Request, APIRouter, Query, HTTPException
from datetime import datetime
from utility.minio import cmd
import os
import json
from io import BytesIO
from orchestration.api.mongo_schemas import Selection, RelevanceSelection
from .api_utils import PrettyJSONResponse
import random

router = APIRouter()


@router.get("/ranking/list-selection-policies")
def list_policies(request: Request):
    # hard code policies for now
    policies = ["random-uniform",
                "top k variance",
                "error sampling",
                "previously ranked"]

    return policies


@router.post("/rank/add-ranking-data-point")
def add_selection_datapoint(
    request: Request, 
    selection: Selection,
    dataset: str = Query(...)  # dataset now as a query parameter  
):
    time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    selection.datetime = time

    # prepare path
    file_name = "{}-{}.json".format(time, selection.username)
    path = "data/ranking/aggregate"
    full_path = os.path.join(dataset, path, file_name)

    # convert to bytes
    dict_data = selection.to_dict()
    json_data = json.dumps(dict_data, indent=4).encode('utf-8')
    data = BytesIO(json_data)

    # upload
    cmd.upload_data(request.app.minio_client, "datasets", full_path, data)

    image_1_hash = selection.image_1_metadata.file_hash
    image_2_hash = selection.image_2_metadata.file_hash

    # update rank count
    # get models counter
    for img_hash in [image_1_hash, image_2_hash]:
        update_image_rank_use_count(request, img_hash)

    return True


@router.post("/rank/update-image-rank-use-count", description="Update image rank use count")
def update_image_rank_use_count(request: Request, image_hash):
    counter = request.app.image_rank_use_count_collection.find_one({"image_hash": image_hash})

    if counter is None:
        # add
        count = 1
        rank_use_count_data = {"image_hash": image_hash,
                               "count": count,
                               }

        request.app.image_rank_use_count_collection.insert_one(rank_use_count_data)
    else:
        count = counter["count"]
        count += 1

        try:
            request.app.image_rank_use_count_collection.update_one(
                {"image_hash": image_hash},
                {"$set": {"count": count}})
        except Exception as e:
            raise Exception("Updating of model counter failed: {}".format(e))

    return True


@router.post("/rank/set-image-rank-use-count", description="Set image rank use count")
def set_image_rank_use_count(request: Request, image_hash, count: int):
    counter = request.app.image_rank_use_count_collection.find_one({"image_hash": image_hash})

    if counter is None:
        # add
        rank_use_count_data = {"image_hash": image_hash,
                               "count": count,
                               }

        request.app.image_rank_use_count_collection.insert_one(rank_use_count_data)
    else:
        try:
            request.app.image_rank_use_count_collection.update_one(
                {"image_hash": image_hash},
                {"$set": {"count": count}})
        except Exception as e:
            raise Exception("Updating of model counter failed: {}".format(e))

    return True


@router.get("/rank/get-image-rank-use-count", description="Get image rank use count")
def get_image_rank_use_count(request: Request, image_hash: str):
    # check if exist
    query = {"image_hash": image_hash}

    item = request.app.image_rank_use_count_collection.find_one(query)
    if item is None:
        raise HTTPException(status_code=404, detail="Image rank use count data not found")

    return item["count"]


@router.post("/ranking/submit-relevance-data")
def add_relevancy_selection_datapoint(request: Request, relevance_selection: RelevanceSelection, dataset: str = Query(...)):
    time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    relevance_selection.datetime = time

    # prepare path
    file_name = "{}-{}.json".format(time, relevance_selection.username)
    path = "data/relevancy/aggregate"
    full_path = os.path.join(dataset, path, file_name)

    # convert to bytes
    dict_data = relevance_selection.to_dict()
    json_data = json.dumps(dict_data, indent=4).encode('utf-8')
    data = BytesIO(json_data)

    # upload
    cmd.upload_data(request.app.minio_client, "datasets", full_path, data)

    return True
