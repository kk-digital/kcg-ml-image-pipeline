
from fastapi import Request, APIRouter
from datetime import datetime
from utility.minio import cmd
import os
import json
from io import BytesIO

from orchestration.api.mongo_schemas import Selection

router = APIRouter()


@router.get("/ranking/list-selection-policies")
def list_policies(request: Request):
    # hard code policies for now
    policies = ["random-unifrom",
                "top k variance",
                "error sampling"]

    return policies


@router.post("/add-selection-datapoint/{dataset}")
def add_selection_datapoint(request: Request, dataset: str, selection: Selection):
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


@router.post("/ranking/submit-relevance-data/{dataset}")
def add_relevancy_selection_datapoint(request: Request, dataset: str, selection: Selection):
    time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    selection.datetime = time

    # prepare path
    file_name = "{}-{}.json".format(time, selection.username)
    path = "data/relevancy/aggregate"
    full_path = os.path.join(dataset, path, file_name)

    # convert to bytes
    dict_data = selection.to_dict()
    json_data = json.dumps(dict_data, indent=4).encode('utf-8')
    data = BytesIO(json_data)

    # upload
    cmd.upload_data(request.app.minio_client, "datasets", full_path, data)

    return True
