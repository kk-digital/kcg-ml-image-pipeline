from fastapi import Request, HTTPException, APIRouter, Response
from orchestration.api.mongo_schemas import SequentialID
from utility.minio import cmd
from datetime import datetime
router = APIRouter()


@router.delete("/dataset/clear-sequential-id")
def clear_dataset_sequential_id_jobs(request: Request):
    request.app.dataset_sequential_id_collection.delete_many({})

    return True


@router.get("/dataset/list")
def get_datasets(request: Request):
    objects = cmd.get_list_of_objects(request.app.minio_client, "datasets")

    return objects

@router.get("/dataset/sequential-id/{dataset}")
def get_sequential_id(request: Request, dataset: str, limit: int = 1):
    sequential_id_arr = []

    # find
    sequential_id = request.app.dataset_sequential_id_collection.find_one({"dataset_name": dataset})
    if sequential_id is None:
        # create one
        new_sequential_id = SequentialID(dataset)

        # get the sequential id arr
        for i in range(limit):
            sequential_id_arr.append(new_sequential_id.get_sequential_id())

        # add to collection
        request.app.dataset_sequential_id_collection.insert_one(new_sequential_id.to_dict())

        return sequential_id_arr

    # if found
    found_sequential_id = SequentialID(sequential_id["dataset_name"], sequential_id["subfolder_count"], sequential_id["file_count"])
    # get the sequential id arr
    for i in range(limit):
        sequential_id_arr.append(found_sequential_id.get_sequential_id())

    new_values = {"$set": found_sequential_id.to_dict()}

    # # update existing sequential id
    request.app.dataset_sequential_id_collection.update_one({"dataset_name": dataset}, new_values)

    return sequential_id_arr


# -------------------- Dataset rate -------------------------
@router.get("/dataset/get-rate/{dataset}")
def get_job(request: Request, dataset: str):
    # find
    query = {"dataset_name": dataset}
    job = request.app.dataset_rate_collection.find_one(query)
    if job is None:
        raise HTTPException(status_code=404)

    # remove the auto generated field
    job.pop('_id', None)

    return job


@router.put("/dataset/set-rate/{dataset}")
def update_job_completed(request: Request, dataset, rate=0):
    date_now = datetime.now()
    # check if exist
    query = {"dataset_name": dataset}
    job = request.app.dataset_rate_collection.find_one(query)
    if job is None:
        # add one
        dataset_rate = {
            "dataset_name": dataset,
            "last_update": date_now,
            "dataset_rate": rate,
        }
        request.app.dataset_rate_collection.insert_one(dataset_rate)

    # update
    new_values = {"$set": {"last_update": date_now,
                           "dataset_rate": rate}}
    request.app.dataset_rate_collection.update_one(query, new_values)

    return True
