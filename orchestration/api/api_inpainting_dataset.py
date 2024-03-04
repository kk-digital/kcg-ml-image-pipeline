from fastapi import Request, HTTPException, APIRouter, Response, Query
from utility.minio import cmd
from orchestration.api.mongo_schemas import SequentialID

router = APIRouter()


@router.get("/datasets-inpainting/list")
def get_datasets(request: Request):
    objects = cmd.get_list_of_objects(request.app.minio_client, "datasets-inpainting")

    return objects

@router.get("/datasets-inpainting/sequential-id/{dataset}")
def get_sequential_id(request: Request, dataset: str, limit: int = 1):
    sequential_id_arr = []

    # find
    sequential_id = request.app.inpainting_dataset_sequential_id_collection.find_one({"dataset_name": dataset})
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
    found_sequential_id = SequentialID(sequential_id["dataset_name"], sequential_id["subfolder_count"],
                                       sequential_id["file_count"])
    # get the sequential id arr
    for i in range(limit):
        sequential_id_arr.append(found_sequential_id.get_sequential_id())

    new_values = {"$set": found_sequential_id.to_dict()}

    # # update existing sequential id
    request.app.dataset_sequential_id_collection.update_one({"dataset_name": dataset}, new_values)

    return sequential_id_arr
