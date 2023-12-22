from fastapi import Request, APIRouter, Query, HTTPException
from datetime import datetime
from utility.minio import cmd
import os
import json
from io import BytesIO
from orchestration.api.mongo_schemas import Selection, RelevanceSelection, NewSelection
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
        return 0

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



@router.post("/rank/add-ranking-data-point-to-mongo")
def add_selection_datapoint(
    request: Request, 
    selection: NewSelection
):
    # Check if file_name and dataset are provided, otherwise compute them
    if not selection.file_name or not selection.dataset:
        current_time = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S')
        selection.file_name = f"{current_time}-{selection.Selection.username}.json"
        selection.dataset = selection.Selection.image_1_metadata.file_path.split('/')[1]

    # Convert selection data to dict
    dict_data = selection.to_dict()

    # Add a datetime field to dict_data if it doesn't exist
    #dict_data['datetime'] = dict_data.get('datetime', datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S'))

    # Insert the data into MongoDB without the '_id'
    mongo_data = dict_data.copy()
    mongo_result = request.app.image_pair_ranking_collection.insert_one(mongo_data)
    inserted_id = mongo_result.inserted_id

    # Prepare data for MinIO upload, excluding the '_id'
    path = "data/ranking/aggregate"
    full_path = os.path.join(selection.dataset, path, selection.file_name)
    json_data = json.dumps(dict_data, indent=4).encode('utf-8')  # dict_data does not include '_id'
    data = BytesIO(json_data)

    # Upload to MinIO
    request.app.minio_client.put_object(
        "datasets",  # Assuming 'datasets' is the bucket name
        full_path,   # The full path where the file will be stored in MinIO
        data,
        length=len(json_data),  # The length of the data to be uploaded
        content_type='application/json'
    )

    # Return the response
    return {"status": "success", "message": "Data point added successfully", "inserted_id": str(inserted_id)}




@router.get("/rank/list-ranking-data", response_class=PrettyJSONResponse)
def list_ranking_data(
    request: Request,
    start_date: str = Query(None),
    end_date: str = Query(None),
    skip: int = Query(0, alias="offset"),
    limit: int = Query(10, alias="limit"),
    order: str = Query("desc", regex="^(desc|asc)$")
):
    # Convert start_date and end_date strings to datetime objects, if provided
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

    # Build the query filter based on dates
    query_filter = {}
    if start_date_obj or end_date_obj:
        date_filter = {}
        if start_date_obj:
            date_filter["$gte"] = start_date_obj.strftime("%Y-%m-%d")
        if end_date_obj:
            date_filter["$lte"] = end_date_obj.strftime("%Y-%m-%d")
        query_filter["file_name"] = date_filter

    # Fetch data from MongoDB with pagination and ordering
    cursor = request.app.image_pair_ranking_collection.find(query_filter).sort("file_name", -1 if order == "desc" else 1).skip(skip).limit(limit)

    # Convert cursor to list of dictionaries
    try:
        ranking_data = []
        for doc in cursor:
            doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
            ranking_data.append(doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ranking_data

@router.delete("/rank/delete-ranking-data-point")
def delete_ranking_data_point(
    request: Request, 
    file_name: str = Query(...)
):
    # Attempt to delete the document with the specified file_name
    result = request.app.image_pair_ranking_collection.delete_one({"file_name": file_name})

    # Check if a document was deleted
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Document with the given file name not found")

    return True


@router.delete("/rank/delete-all-ranking-data-points")
def delete_all_ranking_data_points(request: Request):
    # Attempt to delete all documents from the collection
    result = request.app.image_pair_ranking_collection.delete_many({})

    # Check if any documents were deleted
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="No documents found in the collection")

    return {"message": "All documents deleted successfully"}
