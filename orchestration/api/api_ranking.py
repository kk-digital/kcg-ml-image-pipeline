from fastapi import Request, APIRouter, Query, HTTPException, Body
from datetime import datetime
from utility.minio import cmd
import os
import json
from io import BytesIO
from orchestration.api.mongo_schemas import Selection, RelevanceSelection
from .api_utils import PrettyJSONResponse, ApiResponseHandler, ErrorCode, StandardSuccessResponse, ApiResponseHandler
import random
from collections import OrderedDict
from bson import ObjectId
from typing import Optional

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


@router.post("/rank/add-ranking-data-point-v1", 
             status_code=201,
             description="Add Selection Datapoint",
             response_model=StandardSuccessResponse[Selection],
             responses=ApiResponseHandler.listErrors([422, 500]))
def add_selection_datapoint(request: Request, selection: Selection):
    api_handler = ApiResponseHandler(request)
    
    try:
        current_time = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S')
        file_name = f"{current_time}-{selection.username}.json"
        dataset = selection.image_1_metadata.file_path.split('/')[1]
        selection.datetime = current_time

        dict_data = selection.to_dict()

        # Prepare ordered data for MongoDB insertion
        mongo_data = OrderedDict([
            ("_id", ObjectId()),  # Generate new ObjectId
            ("file_name", file_name),
            ("dataset", dataset),
            *dict_data.items()  # Unpack the rest of dict_data
        ])

        # Insert the ordered data into MongoDB
        request.app.image_pair_ranking_collection.insert_one(mongo_data)

        # Prepare data for MinIO upload (excluding the '_id' field)
        minio_data = mongo_data.copy()
        minio_data.pop("_id")
        minio_data.pop("file_name")
        minio_data.pop("dataset")
        path = "data/ranking/aggregate"
        full_path = os.path.join(dataset, path, file_name)
        json_data = json.dumps(minio_data, indent=4).encode('utf-8')
        data = BytesIO(json_data)

        # Upload data to MinIO
        cmd.upload_data(request.app.minio_client, "datasets", full_path, data)

        image_1_hash = selection.image_1_metadata.file_hash
        image_2_hash = selection.image_2_metadata.file_hash

        # Update rank count for images
        for img_hash in [image_1_hash, image_2_hash]:
            update_image_rank_use_count(request, img_hash)

        # Return a success response
        return api_handler.create_success_response(
            response_data=minio_data,
            http_status_code=201
        )

    except Exception as e:

        return api_handler.create_error_response(
            error_code=ErrorCode.OTHER_ERROR,
            error_string="Internal Server Error",
            http_status_code=500
        )


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


@router.get("/rank/sort-ranking-data-by-residual", response_class=PrettyJSONResponse)
def list_ranking_data(
    request: Request,
    model_type: str = Query(..., description="Model type to filter by, e.g., 'linear' or 'elm-v1'"),
    dataset: Optional[str] = Query(None, description="Dataset to filter by"),
    start_date: str = Query(None),
    end_date: str = Query(None),
    skip: int = Query(0, alias="offset"),
    limit: int = Query(10, alias="limit"),
    order: str = Query("desc", regex="^(desc|asc)$")
):
    # Convert start_date and end_date strings to datetime objects, if provided
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

    # Build the query filter based on dates, model_type, and dataset
    query_filter = {"selected_residual.{}".format(model_type): {"$exists": True}}
    if dataset:
        query_filter["dataset"] = dataset
    if start_date_obj or end_date_obj:
        date_filter = {}
        if start_date_obj:
            date_filter["$gte"] = start_date_obj.strftime("%Y-%m-%d")
        if end_date_obj:
            date_filter["$lte"] = end_date_obj.strftime("%Y-%m-%d")
        query_filter["file_name"] = date_filter

    # Determine the sort order
    sort_order = -1 if order == "desc" else 1

    # Fetch data from MongoDB with pagination and sorting by residual value
    cursor = request.app.image_pair_ranking_collection.find(query_filter).sort(
        f"selected_residual.{model_type}", sort_order).skip(skip).limit(limit)

    # Convert cursor to list of dictionaries
    try:
        ranking_data = []
        for doc in cursor:
            doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
            ranking_data.append(doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ranking_data

@router.get("/rank/sort-ranking-data-by-date", response_class=PrettyJSONResponse)
def list_ranking_data(
    request: Request,
    model_type: str = Query(..., description="Model type to filter by, e.g., 'linear' or 'elm-v1'"),
    dataset: Optional[str] = Query(None, description="Dataset to filter by"),
    start_date: str = Query(None),
    end_date: str = Query(None),
    skip: int = Query(0, alias="offset"),
    limit: int = Query(10, alias="limit"),
    order: str = Query("desc", regex="^(desc|asc)$")
):
    # Convert start_date and end_date strings to datetime objects, if provided
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

    # Build the query filter based on dates, model_type, and dataset
    query_filter = {"selected_residual.{}".format(model_type): {"$exists": True}}
    if dataset:
        query_filter["dataset"] = dataset
    if start_date_obj or end_date_obj:
        date_filter = {}
        if start_date_obj:
            date_filter["$gte"] = start_date_obj.strftime("%Y-%m-%d")
        if end_date_obj:
            date_filter["$lte"] = end_date_obj.strftime("%Y-%m-%d")
        query_filter["file_name"] = date_filter

    # Determine the sort order
    sort_order = -1 if order == "desc" else 1

    # Fetch data from MongoDB with pagination and sorting by date
    cursor = request.app.image_pair_ranking_collection.find(query_filter).sort(
        "file_name", sort_order).skip(skip).limit(limit)

    # Convert cursor to list of dictionaries
    try:
        ranking_data = []
        for doc in cursor:
            doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
            ranking_data.append(doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ranking_data

@router.get("/rank/count-ranking-data", response_class=PrettyJSONResponse)
def count_ranking_data(request: Request):
    try:
        # Get the count of documents in the image_pair_ranking_collection
        count = request.app.image_pair_ranking_collection.count_documents({})

        return {"count": count}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rank/count-selected-residual-data", response_class=PrettyJSONResponse)
def count_ranking_data(request: Request):
    try:
        # Count documents that contain the 'selected_residual' field
        count = request.app.image_pair_ranking_collection.count_documents({"selected_residual": {"$exists": True}})

        return {"count": count}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/rank/delete-ranking-data-point-from-mongo")
def delete_ranking_data_point(request: Request, id: str):
    try:
        # Convert the string ID to ObjectId
        obj_id = ObjectId(id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ObjectId format")

    # Attempt to delete the document with the specified ObjectId
    result = request.app.image_pair_ranking_collection.delete_one({"_id": obj_id})

    # Check if a document was deleted
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Document with the given file id not found")

    return True



@router.delete("/rank/delete-ranking-data-point-from-minio")
def delete_ranking_data_point(request: Request, file_path: str):
    bucket_name = "datasets"  # Assuming the bucket name is 'datasets'

    try:
        # Attempt to remove the object from MinIO
        request.app.minio_client.remove_object(bucket_name, file_path)
    except Exception as e:
        # If there's an error (e.g., file not found), raise an HTTP exception
        raise HTTPException(status_code=500, detail=f"Failed to delete object: {str(e)}")

    return True


@router.delete("/rank/delete-ranking-data", response_class=PrettyJSONResponse)
def delete_ranking_data(request: Request, id: str):
    response_handler = ApiResponseHandler(request)
    try:
        # Convert the string ID to ObjectId
        obj_id = ObjectId(id)
    except Exception:
        return response_handler.create_error_response(ErrorCode.INVALID_PARAMS, "Invalid ObjectId format", 400)

    # Retrieve the document to get the file name and check its presence
    document = request.app.image_pair_ranking_collection.find_one({"_id": obj_id})
    was_present = bool(document)

    if document:
        file_name = document.get("file_name")
        dataset = document.get("dataset")
        if not file_name or not dataset:
            return response_handler.create_error_response(ErrorCode.OTHER_ERROR, "Required data for deletion not found in document", 404)

        # Delete the document from MongoDB
        request.app.image_pair_ranking_collection.delete_one({"_id": obj_id})

        # Construct the MinIO file path
        bucket_name = "datasets"
        minio_file_path = f"{dataset}/data/ranking/aggregate/{file_name}"

        try:
            # Attempt to remove the object from MinIO
            request.app.minio_client.remove_object(bucket_name, minio_file_path)
        except Exception as e:
            return response_handler.create_error_response(ErrorCode.OTHER_ERROR, f"Failed to delete object from MinIO: {str(e)}", 500)

    return response_handler.create_success_response({"wasPresent": was_present})



@router.delete("/rank/delete-all-ranking-data-points")
def delete_all_ranking_data_points(request: Request):
    # Attempt to delete all documents from the collection
    result = request.app.image_pair_ranking_collection.delete_many({})

    # Check if any documents were deleted
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="No documents found in the collection")

    return {"message": "All documents deleted successfully"}


@router.post("/update/add-residual-data", 
             status_code=200,
             description="Add Residual Data to Images")
def add_residual_data(request: Request, selected_img_hash: str, residual: float):
    try:
        # Fetching the MongoDB collection
        image_collection = request.app.image_pair_ranking_collection  

        # Finding and updating documents
        query = {"selected_image_hash": selected_img_hash}
        update = {"$set": {"model_data.residual": residual}}
        
        # Update all documents matching the query
        result = image_collection.update_many(query, update)

        # Check if documents were updated
        if result.modified_count == 0:
            return {"message": "No documents found or updated."}
        
        return {"message": f"Successfully updated {result.modified_count} documents."}

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}



@router.put("/job/add-selected-residual", description="Adds the selected_residual to a completed job.")
def add_selected_residual(
    request: Request,
    image_hash: str = Body(...),
    model_type: str = Body(...),
    residual: float = Body(...)
):
    query = {"selected_image_hash": image_hash}
    update_query = {"$set": {f"selected_residual.{model_type}": residual}}

    result = request.app.image_pair_ranking_collection.update_many(query, update_query)

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if result.modified_count == 0:
        raise HTTPException(status_code=304, detail="Job not updated, possibly no change in data")

    return {"message": f"Updated {result.modified_count} job(s) selected residual successfully."}



@router.put("/job/add-selected-residual-pair-v1", description="Adds the selected_residual for a pair of images based on their selection status in a job.")
def add_selected_residual_pair(
    request: Request,
    selected_image_hash: str = Body(...),
    unselected_image_hash: str = Body(...),
    model_type: str = Body(...),
    selected_residual: float = Body(...)
):
    try:
        # Build the query based on selected_image_index and the unselected_image_hash
        query = {
            "$or": [
                {"selected_image_hash": selected_image_hash, "image_2_metadata.file_hash": unselected_image_hash, "selected_image_index": 0},
                {"selected_image_hash": selected_image_hash, "image_1_metadata.file_hash": unselected_image_hash, "selected_image_index": 1}
            ]
        }

        # Update query for the selected image residual
        update_query = {"$set": {f"selected_residual.{model_type}": selected_residual}}

        # Perform the update for the matching object
        result = request.app.image_pair_ranking_collection.update_one(query, update_query)

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Matching job not found for the image pair")

        return {
            "message": "Selected residual updated successfully for the image pair.",
            "updated_count": result.modified_count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/migrate/minio-to-mongodb", status_code=202, description="Migrate datapoints from Minio to MongoDB.")
def migrate_datapoints_from_minio_to_mongodb(request: Request, minio_bucket: str = 'datasets'):
    api_handler = ApiResponseHandler(request)
    
    try:
        migrate_json_to_mongodb(request.app.minio_client, request.app.image_pair_ranking_collection, minio_bucket)
        return api_handler.create_success_response(
            response_data={"message": "Migration completed successfully."},
            http_status_code=202
        )
    except Exception as e:
        return api_handler.create_error_response(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )

def migrate_json_to_mongodb(minio_client, mongo_collection, minio_bucket):
    for dataset in list_datasets(minio_client, minio_bucket):
        folder_name = f'{dataset}/data/ranking/aggregate'
        print(f"Processing dataset '{dataset}' located at '{folder_name}' in bucket '{minio_bucket}'...")
        
        # Fetch all objects and sort them in reverse (e.g., by filename)
        objects = minio_client.list_objects(minio_bucket, prefix=folder_name, recursive=True)
        sorted_objects = sorted(objects, key=lambda obj: obj.object_name, reverse=True)

        for obj in sorted_objects:
            if obj.is_dir:
                continue

            json_filename = obj.object_name.split('/')[-1]
            if mongo_collection.count_documents({"file_name": json_filename}) > 0:
                print(f"Skipping '{json_filename}', already exists in MongoDB.")
                continue

            print(f"Found object '{obj.object_name}' in dataset '{dataset}'...")
            response = minio_client.get_object(minio_bucket, obj.object_name)
            data = response.read()
            original_data = json.loads(data.decode('utf-8'))
            ordered_data = OrderedDict([
                ("file_name", json_filename),
                ("dataset", dataset),
                *original_data.items()
            ])

            mongo_collection.insert_one(ordered_data)
            print(f"Migrated '{json_filename}' to MongoDB.")



def compare_and_migrate(minio_client, mongo_collection, minio_bucket: str):
    migrated_files = []
    missing_files = []  # Keep track of files in MinIO but not in MongoDB
    mongo_filenames = set(mongo_collection.find().distinct("file_name"))

    for dataset in list_datasets(minio_client, minio_bucket):
        folder_name = f'{dataset}/data/ranking/aggregate'
        objects = minio_client.list_objects(minio_bucket, prefix=folder_name, recursive=True)

        for obj in objects:
            if obj.is_dir or obj.object_name.endswith("/"):
                continue

            json_filename = obj.object_name.split('/')[-1]

            if json_filename not in mongo_filenames:
                missing_files.append(json_filename)  # Add to missing_files if not in MongoDB

    mongo_file_count = mongo_collection.count_documents({})

    print("Missing Files:", missing_files)  # Print the filenames that are present in MinIO but not in MongoDB

    return migrated_files, missing_files, len(missing_files), mongo_file_count



@router.post("/identify-missing-files/minio-mongodb", status_code=200, description="Identify missing files in MongoDB that are present in MinIO.")
def identify_missing_files(request: Request, minio_bucket: str = 'datasets'):
    api_handler = ApiResponseHandler(request)
    
    try:
        _, missing_files, missing_file_count, mongo_count = compare_and_migrate(request.app.minio_client, request.app.image_pair_ranking_collection, minio_bucket)
        return api_handler.create_success_response(
            response_data={
                "message": "Identification of missing files completed successfully.",
                "missing_files": missing_files,
                "missing_file_count": missing_file_count,
                "mongodb_object_count": mongo_count
            },
            http_status_code=200
        )
    except Exception as e:
        return api_handler.create_error_response(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )



def list_datasets(minio_client, bucket_name):
    datasets = set()
    objects = minio_client.list_objects(bucket_name, recursive=False)
    for obj in objects:
        if obj.is_dir:
            dataset_name = obj.object_name.strip('/').split('/')[0]
            datasets.add(dataset_name)
    return list(datasets)


@router.get("/find-duplicates", response_description="Find duplicate filenames")
async def find_duplicates(request: Request):
    try:
        duplicates = find_duplicate_filenames_in_mongo(request.app.image_pair_ranking_collection)
        return {
            "status": "success",
            "message": "Duplicate filenames found",
            "duplicates": duplicates
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def find_duplicate_filenames_in_mongo(collection):
    pipeline = [
        {"$group": {
            "_id": "$file_name",
            "count": {"$sum": 1}
        }},
        {"$match": {
            "count": {"$gt": 1}
        }},
        {"$project": {
            "file_name": "$_id",
            "_id": 0
        }}
    ]
    return [doc['file_name'] for doc in collection.aggregate(pipeline)]


@router.get("/ranking/list-score-fields")
def list_score_fields(request: Request):
    # hard code score fields for now
    fields = ["image_1_clip_sigma_score",
              "image_1_text_embedding_sigma_score",
              "image_2_clip_sigma_score",
              "image_2_text_embedding_sigma_score",
              "delta_score"]

    return fields


@router.get("/selection/list-selection-data-with-scores", response_description="List selection datapoints with detailed scores")
def list_selection_data_with_scores(
    request: Request,
    model_type: str = Query(..., regex="^(linear|elm-v1)$"),
    dataset: str = Query(None),  # Dataset parameter for filtering
    limit: int = Query(10, alias="limit"),
    sort_by: str = Query("delta_score")
):
    response_handler = ApiResponseHandler(request)
    try:
         # Connect to the MongoDB collections
        ranking_collection = request.app.image_pair_ranking_collection
        jobs_collection = request.app.completed_jobs_collection

        # Build query filter based on dataset
        query_filter = {"dataset": dataset} if dataset else {}

        # Fetch data from image_pair_ranking_collection with pagination
        cursor = ranking_collection.find(query_filter).limit(limit)

        # Build a list of hashes
        hashes = []
        for doc in cursor:
            hashes.extend([doc["selected_image_hash"], doc.get("image_2_metadata", {}).get("file_hash", "")])

        # Fetch all relevant jobs in one query
        jobs = {job["task_output_file_dict"]["output_file_hash"]: job for job in jobs_collection.find({"task_output_file_dict.output_file_hash": {"$in": hashes}})}

        selection_data = []
        for doc in cursor.rewind():
            selected_image_job = jobs.get(doc["selected_image_hash"])
            unselected_image_hash = doc["image_2_metadata"]["file_hash"] if doc["selected_image_index"] == 0 else doc["image_1_metadata"]["file_hash"]
            unselected_image_job = jobs.get(unselected_image_hash)

            if not (selected_image_job and unselected_image_job):
                continue

            selected_image_scores = selected_image_job.get("task_attributes_dict", {}).get(model_type, {})
            unselected_image_scores = unselected_image_job.get("task_attributes_dict", {}).get(model_type, {})

            selection_data.append({
                "image_1_hash": doc["selected_image_hash"],
                "image_1_file_path": doc["image_1_metadata"]["file_path"],
                "image_1_clip_sigma_score": selected_image_scores.get("image_clip_sigma_score", None),
                "image_1_text_embedding_sigma_score": selected_image_scores.get("text_embedding_sigma_score", None),
                "image_2_hash": unselected_image_hash,
                "image_2_file_path": doc["image_2_metadata"]["file_path"],
                "image_2_clip_sigma_score": unselected_image_scores.get("image_clip_sigma_score", None),
                "image_2_text_embedding_sigma_score": unselected_image_scores.get("text_embedding_sigma_score", None),
                "delta_score": abs(selected_image_scores.get("image_clip_sigma_score", 0) - unselected_image_scores.get("image_clip_sigma_score", 0))
            })

        # Implement sorting based on the sort_by parameter
        if sort_by in ["image_1_clip_sigma_score", "image_1_text_embedding_sigma_score", "image_2_clip_sigma_score", "image_2_text_embedding_sigma_score", "delta_score"]:
            sorted_selection_data = sorted(selection_data, key=lambda x: x.get(sort_by) or 0, reverse=True)
        else:
            sorted_selection_data = selection_data  # Fallback to no sorting if sort_by is not recognized

        return response_handler.create_success_response(
            sorted_selection_data,
            200
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))