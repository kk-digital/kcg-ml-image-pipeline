from fastapi import Request, APIRouter, Query
from datetime import datetime
from utility.minio import cmd
import os
import json
from io import BytesIO

from orchestration.api.mongo_schemas import Selection, RelevanceSelection

router = APIRouter()


@router.get("/ranking/list-selection-policies")
def list_policies(request: Request):
    # hard code policies for now
    policies = ["random-unifrom",
                "top k variance",
                "error sampling"]

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

    return True


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

