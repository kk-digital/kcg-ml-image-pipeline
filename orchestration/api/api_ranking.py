from fastapi import Request, APIRouter, Query, HTTPException, Body
from datetime import datetime
from utility.minio import cmd
import os
import json
from io import BytesIO
from orchestration.api.mongo_schema.selection_schemas import Selection, RelevanceSelection, ListSelection, FlaggedSelection
from .mongo_schemas import FlaggedDataUpdate
from .api_utils import PrettyJSONResponse, ApiResponseHandler, ErrorCode, StandardSuccessResponse, ApiResponseHandler, TagCountResponse, ApiResponseHandlerV1, StandardSuccessResponseV1, RankCountResponse, CountResponse, JsonContentResponse
import random
from orchestration.api.mongo_schema.selection_schemas import ListRelevanceSelection, ListRankingSelection
from collections import OrderedDict
from bson import ObjectId
from pymongo import ReturnDocument
import pymongo
from typing import Optional, List
import time
import io





router = APIRouter()


@router.get("/ranking/list-selection-policies")
def list_policies(request: Request):
    # hard code policies for now
    policies = ["random-uniform",
                "top k variance",
                "error sampling",
                "previously ranked"]

    return policies


@router.post("/rank/add-ranking-data-point", tags = ['deprecated3'], description= "changed with /rank/add-ranking-data-point-v3")
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


@router.post("/rank/update-image-rank-use-count",tags = ["deprecated2"], description="Update image rank use count")
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


@router.post("/rank/set-image-rank-use-count", tags = ['deprecated3'], description= "changed with /rank/set-image-rank-use-count-v1")
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


@router.get("/rank/get-image-rank-use-count",tags = ['deprecated3'], description= "changed with /rank/get-image-rank-use-count-v1")
def get_image_rank_use_count(request: Request, image_hash: str):
    # check if exist
    query = {"image_hash": image_hash}

    item = request.app.image_rank_use_count_collection.find_one(query)
    if item is None:
        return 0

    return item["count"]


@router.post("/ranking/submit-relevance-data", tags= ["deprecated2"])
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

@router.post("/rank/submit-relevance-data-v1",
             tags=['ranking'],
             response_model=StandardSuccessResponseV1[bool],
             responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def add_relevancy_selection_datapoint_v1(request: Request, relevance_selection: RelevanceSelection, dataset: str = Query(...)):
    response_handler = ApiResponseHandlerV1(request)
    try:
        # If datetime is not provided, set it to the current time
        if not relevance_selection.datetime:
            relevance_selection.datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        # prepare path
        file_name = f"{relevance_selection.datetime}-{relevance_selection.username}.json"
        path = "data/relevancy/aggregate"
        full_path = os.path.join(dataset, path, file_name)

        # convert to JSON bytes
        dict_data = relevance_selection.dict(exclude_unset=True)
        json_data = json.dumps(dict_data, indent=4).encode('utf-8')
        data = BytesIO(json_data)

        # upload
        cmd.upload_data(request.app.minio_client, "datasets", full_path, data)

        # return a success response
        return response_handler.create_success_response_v1(
            response_data={"message": "Relevance data successfully uploaded."},
            http_status_code=201
        )
    except Exception as e:
        # Log the exception and return an error response
        request.app.logger.error(f"Exception occurred: {e}")
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string="Internal Server Error",
            http_status_code=500,
        )


@router.get("/rank/list-ranking-data", tags = ["deprecated2"], response_class=PrettyJSONResponse)
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


@router.get("/rank/sort-ranking-data-by-residual", tags = ['deprecated3'], description= "changed with /rank/sort-ranking-data-by-residual-v1")
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

@router.get("/rank/sort-ranking-data-by-date", tags = ["deprecated2"], response_class=PrettyJSONResponse)
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

@router.get("/rank/count-ranking-data", tags = ['deprecated3'], description= "changed with /rank/count-ranking-data-v1")
def count_ranking_data(request: Request):
    try:
        # Get the count of documents in the image_pair_ranking_collection
        count = request.app.image_pair_ranking_collection.count_documents({})

        return {"count": count}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rank/count-selected-residual-data", tags = ['deprecated3'], description= "changed with /rank/count-selected-residual-data-v1")
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


@router.get("/ranking/list-score-fields")
def list_score_fields(request: Request):
    # hard code score fields for now
    fields = ["selected_image_clip_sigma_score",
              "selected_text_embedding_sigma_score",
              "unselected_image_clip_sigma_score",
              "unselected_text_embedding_sigma_score",
              "delta_score"]

    return fields

@router.get("/ranking/list-score-models")
def list_score_fields(request: Request):
    # hard code score fields for now
    fields = ["linear",
              "elm-v1"]
    return fields

@router.get("/selection/list-selection-data-with-scores", response_description="List selection datapoints with detailed scores")
def list_selection_data_with_scores(
    request: Request,
    model_type: str = Query(..., regex="^(linear|elm-v1)$"),
    dataset: str = Query(None),  # Dataset parameter for filtering
    include_flagged: bool = Query(False),  # Parameter to include or exclude flagged documents
    limit: int = Query(10, alias="limit"),
    offset: int = Query(0, alias="offset"),  # Added for pagination
    sort_by: str = Query("delta_score"),  # Default sorting parameter
    order: str = Query("asc")  # Parameter for sort order
):
    response_handler = ApiResponseHandler(request)
    
    try:
        # Connect to the MongoDB collections
        ranking_collection = request.app.image_pair_ranking_collection
        jobs_collection = request.app.completed_jobs_collection

        # Build query filter based on dataset and ensure delta_score exists for the model_type
        query_filter = {}
        if dataset:
            query_filter["dataset"] = dataset

        if not include_flagged:
            query_filter["flagged"] = {"$ne": True}

        # Ensure delta_score for the model_type exists and is not null
        query_filter[f"delta_score.{model_type}"] = {"$exists": True, "$ne": None}

        # Prepare sorting
        sort_order = 1 if order == "asc" else -1
        # Adjust sorting query for nested delta_score by model_type
        sort_query = [("delta_score." + model_type, sort_order)] if sort_by == "delta_score" else [(sort_by, sort_order)]

        # Fetch and sort data with pagination
        cursor = ranking_collection.find(query_filter).sort(sort_query).skip(offset).limit(limit)


        selection_data = []
        doc_count = 0
        for doc in cursor:
            doc_count += 1
            print(f"Processing document {doc['_id']}")
            # Check if the document is flagged
            is_flagged = doc.get("flagged", False)
            selection_file_name = doc["file_name"]
            delta_score = doc.get("delta_score", {}).get(model_type, None)
            selected_image_index = doc["selected_image_index"]
            selected_image_hash = doc["selected_image_hash"]
            selected_image_path = doc["image_1_metadata"]["file_path"] if selected_image_index == 0 else doc["image_2_metadata"]["file_path"]
            # Determine unselected image hash and path based on selected_image_index
            if selected_image_index == 0:
                unselected_image_hash = doc["image_2_metadata"]["file_hash"]
                unselected_image_path = doc["image_2_metadata"]["file_path"]
            else:
                unselected_image_hash = doc["image_1_metadata"]["file_hash"]
                unselected_image_path = doc["image_1_metadata"]["file_path"]
                
            # Fetch scores from completed_jobs_collection for both images
            selected_image_job = jobs_collection.find_one({"task_output_file_dict.output_file_hash": selected_image_hash})
            unselected_image_job = jobs_collection.find_one({"task_output_file_dict.output_file_hash": unselected_image_hash})

            # Skip this job if task_attributes_dict is missing
            if not selected_image_job or "task_attributes_dict" not in selected_image_job or not unselected_image_job or "task_attributes_dict" not in unselected_image_job:
                print(f"Skipping document {doc['_id']} due to missing job data or task_attributes_dict.")
                continue

            # Extract scores for both images
            selected_image_scores = selected_image_job["task_attributes_dict"][model_type]
            unselected_image_scores = unselected_image_job["task_attributes_dict"][model_type]
            
            selection_data.append({
                "selected_image": {
                    "selected_image_path": selected_image_path,
                    "selected_image_hash": selected_image_hash,
                    "selected_image_clip_sigma_score": selected_image_scores.get("image_clip_sigma_score", None),
                    "selected_text_embedding_sigma_score": selected_image_scores.get("text_embedding_sigma_score", None)
                },
                "unselected_image": {
                    "unselected_image_path": unselected_image_path,
                    "unselected_image_hash": unselected_image_hash,
                    "unselected_image_clip_sigma_score": unselected_image_scores.get("image_clip_sigma_score", None),
                    "unselected_text_embedding_sigma_score": unselected_image_scores.get("text_embedding_sigma_score", None)
                },
                "selection_datapoint_file_name": selection_file_name,
                "delta_score": delta_score,
                "flagged": is_flagged 
            })
            print(f"Finished processing document {doc['_id']}.")

        print(f"Total documents processed: {doc_count}. Selection data count: {len(selection_data)}")    
        return response_handler.create_success_response(
            selection_data,
            200
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calculate-delta-scores", status_code=200)
async def calculate_delta_scores(request: Request):
    start_time = time.time()

    # Define the model types for which you want to calculate delta_scores
    model_types = ["linear", "elm-v1"]

    # Access collections
    ranking_collection = request.app.image_pair_ranking_collection
    jobs_collection = request.app.completed_jobs_collection

    processed_count = 0
    skipped_count = 0

    # Fetch all documents from ranking_collection
    for doc in ranking_collection.find({}):

        # Skip documents where delta_score already exists for all model_types
        if all(f"{model_type}" in doc.get("delta_score", {}) for model_type in model_types):
            print(f"Skipping document {doc['_id']} as delta_score already exists for all model types.")
            skipped_count += 1
            continue

        selected_image_index = doc["selected_image_index"]
        selected_image_hash = doc["selected_image_hash"]
        unselected_image_hash = doc["image_2_metadata"]["file_hash"] if selected_image_index == 0 else doc["image_1_metadata"]["file_hash"]

        for model_type in model_types:
            # Proceed only if the delta_score for this model_type does not exist
            if f"delta_score.{model_type}" not in doc:
                print(f"Processing document {doc['_id']} for model type '{model_type}'.")
                selected_image_job = jobs_collection.find_one({"task_output_file_dict.output_file_hash": selected_image_hash})
                unselected_image_job = jobs_collection.find_one({"task_output_file_dict.output_file_hash": unselected_image_hash})

                if selected_image_job and unselected_image_job and "task_attributes_dict" in selected_image_job and "task_attributes_dict" in unselected_image_job:
                    if model_type in selected_image_job["task_attributes_dict"] and model_type in unselected_image_job["task_attributes_dict"]:
                        selected_image_scores = selected_image_job["task_attributes_dict"][model_type]
                        unselected_image_scores = unselected_image_job["task_attributes_dict"][model_type]

                        if "image_clip_sigma_score" in selected_image_scores and "image_clip_sigma_score" in unselected_image_scores:
                            delta_score = selected_image_scores["image_clip_sigma_score"] - unselected_image_scores["image_clip_sigma_score"]

                            # Update the document in ranking_collection with the new delta_score under the specific model_type
                            update_field = f"delta_score.{model_type}"
                            ranking_collection.update_one(
                                {"_id": doc["_id"]},
                                {"$set": {update_field: delta_score}}
                            )
                            processed_count += 1

    end_time = time.time()
    total_time = end_time - start_time

    return {"message": f"Delta scores calculation and update complete. Processed: {processed_count}, Skipped: {skipped_count}.", "total_time": f"{total_time:.2f} seconds"}




# New Standardized APIs

@router.post("/rank/add-ranking-data-point-v2", 
             status_code=201,
             tags = ['deprecated3'], 
             description= "changed with /rank/add-ranking-data-point-v3",
             response_model=StandardSuccessResponseV1[Selection],  
             responses=ApiResponseHandlerV1.listErrors([422, 500]))
async def add_selection_datapoint_v2(
    request: Request, 
    selection: Selection,
    dataset: str = Query(..., description="Dataset as a query parameter")  
):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        selection.datetime = time

        # Prepare path
        file_name = "{}-{}.json".format(time, selection.username)
        path = "data/ranking/aggregate"
        full_path = os.path.join(dataset, path, file_name)

        # Convert to bytes
        dict_data = selection.to_dict()
        json_data = json.dumps(dict_data, indent=4).encode('utf-8')
        data = BytesIO(json_data)

        # Upload
        cmd.upload_data(request.app.minio_client, "datasets", full_path, data)

        image_1_hash = selection.image_1_metadata.file_hash
        image_2_hash = selection.image_2_metadata.file_hash

        # Update rank count for both images
        for img_hash in [image_1_hash, image_2_hash]:
            update_image_rank_use_count(request, img_hash)

        return response_handler.create_success_response_v1(
            response_data = dict_data, 
            http_status_code=201,
        )
    except Exception as e:
        # Handle exceptions and return an error response
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )
    

@router.post("/rank/add-ranking-data-point-v3", 
             status_code=201,
             tags = ['ranking'],
             description="'rank/add-ranking-data-point-v2' is the replacement.",
             response_model=StandardSuccessResponseV1[Selection],
             responses=ApiResponseHandlerV1.listErrors([422, 500]))
async def add_selection_datapoint_v3(request: Request, selection: Selection):
    api_handler = await ApiResponseHandlerV1.createInstance(request)
    
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
        return api_handler.create_success_response_v1(
            response_data=minio_data,
            http_status_code=201
        )

    except Exception as e:

        return api_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )


    
@router.post("/rank/set-image-rank-use-count-v1", 
             description="Set image rank use count", 
             tags=["ranking"],
             response_model=StandardSuccessResponseV1[RankCountResponse], 
             responses=ApiResponseHandlerV1.listErrors([ 422, 500]))
def set_image_rank_use_count(request: Request, image_hash: str, count: int):
    response_handler = ApiResponseHandlerV1(request)
    try:
        counter = request.app.image_rank_use_count_collection.find_one({"image_hash": image_hash})

        if counter is None:
            # If the image hash does not exist, create a new counter
            rank_use_count_data = {
                "image_hash": image_hash,
                "count": count,
            }
            request.app.image_rank_use_count_collection.insert_one(rank_use_count_data)
          
        else:
            # If the image hash exists, update the count
            request.app.image_rank_use_count_collection.update_one(
                {"image_hash": image_hash},
                {"$set": {"count": count}}
            )

        # Return a success response indicating the action performed
        return response_handler.create_success_response_v1(
            response_data={"image_hash": image_hash, "count": count},
            http_status_code=200,
        )
    except Exception as e:
        # Log the exception and return an error response
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string="Internal Server Error",
            http_status_code=500,
        )    
    
@router.get("/rank/get-image-rank-use-count-v1", 
            description="Get image rank use count",
            tags=["ranking"],
            response_model=StandardSuccessResponseV1[RankCountResponse],  
            responses=ApiResponseHandlerV1.listErrors([ 422, 500]))
def get_image_rank_use_count_v1(request: Request, image_hash: str):
    response_handler = ApiResponseHandlerV1(request)
    try:
        query = {"image_hash": image_hash}
        item = request.app.image_rank_use_count_collection.find_one(query)

        if item is None:
            # If the item doesn't exist, return a count of 0 with a success response
            return response_handler.create_success_response_v1(
                response_data={"image_hash": image_hash, "count": 0}, 
                http_status_code=200
            )

        # If the item exists, return the count with a success response
        return response_handler.create_success_response_v1(
            response_data={"image_hash": image_hash, "count": item["count"]}, 
            http_status_code=200
        )
    except Exception as e:
        # Handle exceptions and return an error response
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string="Internal Server Error",
            http_status_code=500,
        )

@router.post("/rank/add-relevance-data-point",
             status_code=201,
             tags=["ranking"],
             response_model=StandardSuccessResponseV1[RelevanceSelection],  
             responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
async def add_relevancy_selection_datapoint_v1(request: Request, relevance_selection: RelevanceSelection, dataset: str = Query(..., description="Dataset as a query parameter")):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        # Current datetime for filename and metadata
        time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        relevance_selection.datetime = time

        # Prepare the file path
        file_name = "{}-{}.json".format(time, relevance_selection.username)
        path = "data/relevancy/aggregate"
        full_path = os.path.join(dataset, path, file_name)

        # Convert selection to JSON bytes
        dict_data = relevance_selection.to_dict()
        json_data = json.dumps(dict_data, indent=4).encode('utf-8')
        data = BytesIO(json_data)


        # Upload data to MinIO
        cmd.upload_data(request.app.minio_client, "datasets", full_path, data)

        return response_handler.create_success_response_v1(
            response_data = dict_data, 
            http_status_code=201,
        )
    except Exception as e:
        # Log the exception and return an error response
        print(f"Exception occurred: {e}")  # For debugging
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )

    

@router.get("/rank/list-ranking-data-v1", 
            status_code=200,
            tags=["deprecated2"],
            response_model=StandardSuccessResponseV1[Selection],  
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def list_ranking_data_v1(
    request: Request,
    dataset: str = Query(None),
    start_date: str = Query(None),
    end_date: str = Query(None),
    skip: int = Query(0, alias="offset"),
    limit: int = Query(10, alias="limit"),
    order: str = Query("desc", regex="^(desc|asc)$")
):
    response_handler = ApiResponseHandlerV1(request)
    try:
        # Convert start_date and end_date strings to datetime objects, if provided
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

        # Build the query filter based on dates
        query_filter = {}
        if start_date_obj or end_date_obj:
            date_filter = {}
            if start_date_obj:
                date_filter["$gte"] = start_date_obj
            if end_date_obj:
                date_filter["$lte"] = end_date_obj
            query_filter["datetime"] = date_filter  # Assuming the field in the database is "datetime"

        if dataset:
            query_filter["dataset"] = dataset

        # Fetch data from MongoDB with pagination and ordering
        cursor = request.app.image_pair_ranking_collection.find(query_filter).sort("datetime", -1 if order == "desc" else 1).skip(skip).limit(limit)

        ranking_data = list(cursor)
        for doc in ranking_data:
            doc['_id'] = str(doc['_id'])  # Convert ObjectId to string

        # Return the fetched data with a success response
        return response_handler.create_success_response_v1(
            response_data=ranking_data, 
            http_status_code=200,
        )
    except Exception as e:
        # Log the exception and return an error response
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string="Internal Server Error",
            http_status_code=500,
        )
    
@router.get("/rank/sort-ranking-data-by-residual-v1", 
            description="rank data by residual",
            tags=["ranking"],
            response_model=StandardSuccessResponseV1[Selection],  
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def list_ranking_data_by_residual(
    request: Request,
    model_type: str = Query(..., description="Model type to filter by, e.g., 'linear' or 'elm-v1'"),
    dataset: Optional[str] = Query(None, description="Dataset to filter by"),
    start_date: str = Query(None),
    end_date: str = Query(None),
    skip: int = Query(0, alias="offset"),
    limit: int = Query(10, alias="limit"),
    order: str = Query("desc", regex="^(desc|asc)$")
):
    response_handler = ApiResponseHandlerV1(request)
    try:
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

        # Fetch and sort data from MongoDB with pagination
        cursor = request.app.image_pair_ranking_collection.find(query_filter).sort(
            f"selected_residual.{model_type}", sort_order).skip(skip).limit(limit)

        # Convert cursor to list of dictionaries
        ranking_data = list(cursor)
        for doc in ranking_data:
            doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
        
        # Return the fetched data with a success response
        return response_handler.create_success_response_v1(
            response_data=ranking_data, 
            http_status_code=200,
        )
    except Exception as e:
        # Handle exceptions and return an error response
        print(f"Exception occurred: {e}")  # For debugging
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )    

@router.get("/rank/sort-ranking-data-by-date-v2", 
            description="Sort rank data by date",
            tags=["ranking"],
            response_model=StandardSuccessResponseV1[ListSelection],  
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
async def sort_ranking_data_by_date_v2(
    request: Request,
    dataset: str = Query(..., description="Dataset to filter by"),
    start_date: Optional[str] = Query(None, description="Start date (inclusive) in YYYY-MM-DD format"),
    rank_model_id: int = Query(None, description="Rank model ID to filter by"),
    end_date: Optional[str] = Query(None, description="End date (inclusive) in YYYY-MM-DD format"),
    skip: int = Query(0, alias="offset"),
    limit: int = Query(10, alias="limit"),
    order: str = Query("desc", regex="^(desc|asc)$")
):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        # Convert start_date and end_date strings to datetime objects, if provided
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else None

        # Build the query filter based on dataset and optional dates
        query_filter = {"dataset": dataset, "rank_model_id": rank_model_id}
        if start_date_obj or end_date_obj:
            date_filter = {}
            if start_date_obj:
                date_filter["$gte"] = start_date_obj
            if end_date_obj:
                date_filter["$lte"] = end_date_obj
            query_filter["datetime"] = date_filter 

        # Determine the sort order
        sort_order = pymongo.DESCENDING if order == "desc" else pymongo.ASCENDING

        # Fetch and sort data from MongoDB with pagination
        cursor = request.app.image_pair_ranking_collection.find(query_filter).sort(
            "datetime", sort_order  
        ).skip(skip).limit(limit)

        # Convert cursor to list of dictionaries
        ranking_data = [doc for doc in cursor]
        for doc in ranking_data:
            doc['_id'] = str(doc['_id'])  # Convert ObjectId to string for JSON serialization
        
        return response_handler.create_success_response_v1(
            response_data={"ranking_data": ranking_data}, 
            http_status_code=200
        )
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=f"Internal Server Error: {str(e)}",
            http_status_code=500
        )    

@router.get("/rank/sort-ranking-data-by-date-v1", 
            description="list ranking data by date",
            tags=["deprecated2"],
            response_model=StandardSuccessResponseV1[List[Selection]],  
            responses=ApiResponseHandlerV1.listErrors([400, 422, 500]))
def list_ranking_data_by_date(
    request: Request,
    model_type: str = Query(..., description="Model type to filter by, e.g., 'linear' or 'elm-v1'"),
    dataset: Optional[str] = Query(None, description="Dataset to filter by"),
    start_date: str = Query(None),
    end_date: str = Query(None),
    skip: int = Query(0, alias="offset"),
    limit: int = Query(10, alias="limit"),
    order: str = Query("desc", regex="^(desc|asc)$")
):
    response_handler = ApiResponseHandlerV1(request)
    try:
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

        # Fetch and sort data from MongoDB with pagination
        cursor = request.app.image_pair_ranking_collection.find(query_filter).sort(
            "file_name", sort_order).skip(skip).limit(limit)  # Assuming the field is "datetime"

        # Convert cursor to list of dictionaries
        ranking_data = list(cursor)
        for doc in ranking_data:
            doc['_id'] = str(doc['_id'])  # Convert ObjectId to string

        # Return the fetched data with a success response
        return response_handler.create_success_response_v1(
            response_data=ranking_data, 
            http_status_code=200,
        )
    except Exception as e:
        # Handle exceptions and return an error response
        print(f"Exception occurred: {e}")  # For debugging
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )    
    

@router.get("/rank/count-ranking-data-v1", 
            description="count ranking data",
            tags=["ranking"],
            response_model=StandardSuccessResponseV1[CountResponse],  
            responses=ApiResponseHandlerV1.listErrors([500]))
def count_ranking_data(request: Request):
    response_handler = ApiResponseHandlerV1(request)
    try:
        # Get the count of documents in the image_pair_ranking_collection
        count = request.app.image_pair_ranking_collection.count_documents({})

        # Return the count with a success response
        return response_handler.create_success_response_v1(
            response_data={"count": count}, 
            http_status_code=200,
        )
    except Exception as e:
        # Handle exceptions and return an error response
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )

@router.get("/rank/count-selected-residual-data-v1", 
            description="count ranking data for selected residual field",
            tags=["ranking"],
            response_model=StandardSuccessResponseV1[CountResponse],  
            responses=ApiResponseHandlerV1.listErrors([500]))
def count_selected_residual_data(request: Request):
    response_handler = ApiResponseHandlerV1(request)
    try:
        # Count documents that contain the 'selected_residual' field
        count = request.app.image_pair_ranking_collection.count_documents({"selected_residual": {"$exists": True}})

        # Return the count with a success response
        return response_handler.create_success_response_v1(
            response_data={"count": count}, 
            http_status_code=200,
        )
    except Exception as e:
        # Handle exceptions and return an error response
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )

@router.post("/rank/add-residual-data-v1", 
             response_model=StandardSuccessResponseV1[List[Selection]],
             tags=["ranking"],
             status_code=200,
             description="Add Residual Data to Images",
             responses=ApiResponseHandlerV1.listErrors([400, 500]))
def add_residual_data(request: Request, selected_img_hash: str, residual: float):
    response_handler = ApiResponseHandlerV1(request)
    try:
        # Fetching the MongoDB collection
        image_collection = request.app.image_pair_ranking_collection  

        # Finding and updating documents
        query = {"selected_image_hash": selected_img_hash}
        update = {"$set": {"model_data.residual": residual}}
        
        # Update all documents matching the query
        result = image_collection.update_many(query, update)

        # If no documents were updated, return an appropriate response
        if result.matched_count == 0:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="No matching documents found or no new data to update.",
                http_status_code=404,
            )
        
        # Re-fetch the updated documents to return them
        updated_documents = list(image_collection.find(query))
        
        # Serialize the MongoDB documents, excluding fields like '_id'
        updated_documents_data = [{k: v for k, v in document.items() if k != '_id'} for document in updated_documents]

        # Return the updated documents with a success response
        return response_handler.create_success_response_v1(
            response_data=updated_documents_data, 
            http_status_code=200,
        )
    except Exception as e:
        # Handle exceptions and return an error response
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )

@router.put("/job/add-selected-residual-pair-v2",
            tags=["ranking"],
            response_model=StandardSuccessResponseV1[Selection], 
            description="Adds the selected_residual for a pair of images based on their selection status in a job.",
            responses=ApiResponseHandlerV1.listErrors([400, 404, 422, 500]))
def add_selected_residual_pair(
    request: Request,
    selected_image_hash: str = Body(...),
    unselected_image_hash: str = Body(...),
    model_type: str = Body(...),
    selected_residual: float = Body(...)
):
    response_handler = ApiResponseHandlerV1(request)
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

        # Check if documents were updated
        if result.matched_count == 0:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string="Matching job not found for the image pair",
                http_status_code=404,
            )

        # Retrieve the updated document
        updated_document = request.app.image_pair_ranking_collection.find_one(query)

        # Serialize the MongoDB document, excluding fields like '_id'
        updated_document_data = {k: v for k, v in updated_document.items() if k != '_id'}

        # Return the updated document with a success response
        return response_handler.create_success_response_v1(
            response_data=updated_document_data, 
            http_status_code=200,
        )

    except Exception as e:
        # Log the exception and return an error response
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )

@router.get("/rank/read",tags = ['deprecated2'], response_class=PrettyJSONResponse)
def read_ranking_file(request: Request, dataset: str,
                      filename: str = Query(..., description="Filename of the JSON to read")):
    # Construct the object name for ranking
    object_name = f"{dataset}/data/ranking/aggregate/{filename}"

    # Fetch the content of the specified JSON file
    data = cmd.get_file_from_minio(request.app.minio_client, "datasets", object_name)

    if data is None:
        raise HTTPException(status_code=410, detail=f"File {filename} not found.")

    file_content = ""
    for chunk in data.stream(32 * 1024):
        file_content += chunk.decode('utf-8')

    # Return the content of the JSON file
    return json.loads(file_content)

@router.get("/rank/read-ranking-datapoint", 
            tags=['ranking'], 
            description = "read ranking datapoints",
            response_model=StandardSuccessResponseV1[JsonContentResponse], 
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def read_ranking_file(request: Request, dataset: str, filename: str = Query(..., description="Filename of the JSON to read")):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        # Construct the object name for ranking
        object_name = f"{dataset}/data/ranking/aggregate/{filename}"

        # Fetch the content of the specified JSON file
        data = cmd.get_file_from_minio(request.app.minio_client, "datasets", object_name)

        if data is None:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string=f"File {filename} not found.",
                http_status_code=404,
            )

        file_content = ""
        for chunk in data.stream(32 * 1024):
            file_content += chunk.decode('utf-8')

        # Successfully return the content of the JSON file
        return response_handler.create_success_response_v1(
            response_data={"json_content":json.loads(file_content)},
            http_status_code=200
        )
    except Exception as e:
        # Handle exceptions and return an error response
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string="Internal Server Error",
            http_status_code=500,
        )



@router.get("/relevancy/read", tags = ['deprecated2'], response_class=PrettyJSONResponse)
def read_relevancy_file(request: Request, dataset: str,
                        filename: str = Query(..., description="Filename of the JSON to read")):
    # Construct the object name for relevancy
    object_name = f"{dataset}/data/relevancy/aggregate/{filename}"

    # Fetch the content of the specified JSON file
    data = cmd.get_file_from_minio(request.app.minio_client, "datasets", object_name)

    if data is None:
        raise HTTPException(status_code=410, detail=f"File {filename} not found.")

    file_content = ""
    for chunk in data.stream(32 * 1024):
        file_content += chunk.decode('utf-8')

    # Return the content of the JSON file
    return json.loads(file_content)

@router.get("/rank/read-relevance-datapoint", 
            tags=['ranking'],
            description = "read relevancy datapoints",
            response_model=StandardSuccessResponseV1[JsonContentResponse], 
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def read_relevancy_file(request: Request, dataset: str, filename: str = Query(..., description="Filename of the JSON to read")):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        # Construct the object name for relevancy
        object_name = f"{dataset}/data/relevancy/aggregate/{filename}"

        # Fetch the content of the specified JSON file
        data = cmd.get_file_from_minio(request.app.minio_client, "datasets", object_name)

        if data is None:
            return response_handler.create_error_response_v1(
                error_code=ErrorCode.ELEMENT_NOT_FOUND,
                error_string=f"File {filename} not found.",
                http_status_code=410,
            )

        file_content = ""
        for chunk in data.stream(32 * 1024):
            file_content += chunk.decode('utf-8')

        # Successfully return the content of the JSON file
        return response_handler.create_success_response_v1(
            response_data={"json_content":json.loads(file_content)},
            http_status_code=200
        )
    except Exception as e:
        # Handle exceptions and return an error response
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500,
        )


@router.put("/rank/update-ranking-datapoint", 
            tags=['ranking'], 
            response_model=StandardSuccessResponseV1[FlaggedSelection],
            responses=ApiResponseHandlerV1.listErrors([404, 422]))
async def update_ranking_file(request: Request, dataset: str, filename: str, update_data: FlaggedDataUpdate):
    response_handler = await ApiResponseHandlerV1.createInstance(request)

    # Construct the object name based on the dataset
    object_name = f"{dataset}/data/ranking/aggregate/{filename}"

    flagged_time = datetime.now().isoformat()

    # Fetch the content of the specified JSON file from MinIO
    try:
        data = cmd.get_file_from_minio(request.app.minio_client, "datasets", object_name)
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.ELEMENT_NOT_FOUND,
            error_string=f"Error fetching file {filename}: {str(e)}",
            http_status_code=404,
        )

    if data is None:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.ELEMENT_NOT_FOUND,
            error_string=f"File {filename} not found.",
            http_status_code=404,
        )

    file_content = ""
    for chunk in data.stream(32 * 1024):
        file_content += chunk.decode('utf-8')

    try:
        # Load the existing content and update it
        content_dict = json.loads(file_content)
        content_dict["flagged"] = update_data.flagged
        content_dict["flagged_by_user"] = update_data.flagged_by_user
        content_dict["flagged_time"] = flagged_time

        # Save the modified file back to MinIO
        updated_content = json.dumps(content_dict, indent=2)
        updated_data = io.BytesIO(updated_content.encode('utf-8'))
        request.app.minio_client.put_object("datasets", object_name, updated_data, len(updated_content))
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=f"Failed to update file {filename}: {str(e)}",
            http_status_code=500,
        )

    # Update the document in MongoDB
    query = {"file_name": filename}
    update = {"$set": {
        "flagged": update_data.flagged,
        "flagged_by_user": update_data.flagged_by_user,
        "flagged_time": datetime.now().isoformat()
    }}
    updated_document = request.app.image_pair_ranking_collection.find_one_and_update(
        query, update, return_document=ReturnDocument.AFTER
    )

    if updated_document is None:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.ELEMENT_NOT_FOUND,
            error_string=f"Document with filename {filename} not found in MongoDB.",
            http_status_code=404,
        )
    
    if '_id' in updated_document:
        updated_document['_id'] = str(updated_document['_id'])

    return response_handler.create_success_response_v1(
        response_data=updated_document,
        http_status_code=200,
    )


@router.get("/rank/list-relevance-datapoints",
            description="List relevancy files for a dataset",
            tags=["ranking"],
            response_model=StandardSuccessResponseV1[ListRelevanceSelection],
            responses=ApiResponseHandlerV1.listErrors([500]))
async def list_relevancy_files(request: Request, dataset: str):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        path_prefix = f"{dataset}/data/relevancy/aggregate"
        objects = cmd.get_list_of_objects_with_prefix(request.app.minio_client, "datasets", path_prefix)
        json_files = [obj for obj in objects if obj.endswith('.json')]

        return response_handler.create_success_response_v1(
            response_data={"datapoints": json_files}, 
            http_status_code=200
        )
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )        

@router.get("/rank/list-ranking-datapoints",
            description="List ranking files for a dataset",
            tags=["ranking"],
            response_model=StandardSuccessResponseV1[ListRankingSelection],
            responses=ApiResponseHandlerV1.listErrors([500]))
async def list_ranking_files(request: Request, dataset: str):
    response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        path_prefix = f"{dataset}/data/ranking/aggregate"
        objects = cmd.get_list_of_objects_with_prefix(request.app.minio_client, "datasets", path_prefix)
        json_files = [obj for obj in objects if obj.endswith('.json')]

        return response_handler.create_success_response_v1(
            response_data={"datapoints": json_files}, 
            http_status_code=200
        )
    except Exception as e:
        return response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR,
            error_string=str(e),
            http_status_code=500
        )  