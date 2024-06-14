
from fastapi import APIRouter, Request
from .api_utils import ApiResponseHandlerV1, StandardSuccessResponseV1, ErrorCode, WasPresentResponse
from .mongo_schemas import ExtractImageData, ListExtractImageData
from datetime import datetime
import uuid

router = APIRouter()

@router.get("/extracts/get-current-data-batch-sequential-id", 
            description="Get the sequential id for numpy file batches stored for a dataset",
            tags=["extracts"])
async def get_current_data_batch_sequential_id(request: Request, dataset: str):

    # get batch counter
    counter = request.app.extract_data_batch_sequential_id.find_one({"dataset": dataset})
    # create counter if it doesn't exist already
    if counter is None:
        # insert the new counter
        insert_result= request.app.extract_data_batch_sequential_id.insert_one({"dataset": dataset, "sequence_num": 0, "complete": True})
        # Retrieve the inserted counter using the inserted_id
        counter = request.app.extract_data_batch_sequential_id.find_one({'_id': insert_result.inserted_id})
    
    # remove _id field
    counter.pop("_id")

    return counter

@router.get("/extracts/get-next-data-batch-sequential-id", 
            description="Increment the sequential id for numpy file batches stored for a dataset",
            tags=["extracts"])
async def get_next_data_batch_sequential_id(request: Request, dataset: str, complete: bool):

    # get batch counter
    counter = request.app.extract_data_batch_sequential_id.find_one({"dataset": dataset})
    # create counter if it doesn't exist already
    if counter is None:
        request.app.extract_data_batch_sequential_id.insert_one({"dataset": dataset})

    # get current last batch count
    counter_seq = counter["sequence_number"] if counter else 0
    counter_seq += 1
    try:
        ret = request.app.counters_collection.update_one(
            {"dataset": dataset},
            {"$set": 
                {
                    "sequence_number": counter_seq,
                    "complete": complete
                }})
    except Exception as e:
        raise Exception("Updating of classifier counter failed: {}".format(e))

    return counter_seq

@router.post("/extracts/add-extracted-image", 
            description="Add an extracted image data",
            tags=["extracts"],  
            response_model=StandardSuccessResponseV1[ListExtractImageData],  
            responses=ApiResponseHandlerV1.listErrors([404,422, 500]))
async def add_extract(request: Request, image_data: ExtractImageData):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:

        image_data.uuid = str(uuid.uuid4())

        existed = request.app.extracts_collection.find_one({
            "image_hash": image_data.image_hash
        })

        if existed is None:
            image_data.upload_date = str(datetime.now())
            request.app.extracts_collection.insert_one(image_data.to_dict())
        else:
            return api_response_handler.create_error_response_v1(
                error_code=ErrorCode.INVALID_PARAMS,
                error_string="An image with this hash already exists",
                http_status_code=400
            )
        
        return api_response_handler.create_success_response_v1(
            response_data={"data": image_data.to_dict()},
            http_status_code=200  
        )
    
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )

@router.get("/extracts/get-all-extracts-list", 
            description="Get all extracted images. If 'dataset' parameter is set, it only returns images from that dataset, and if the 'size' parameter is set, a random sample of that size will be returned.",
            tags=["extracts"],  
            response_model=StandardSuccessResponseV1[ListExtractImageData],  
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def get_all_extracts_list(request: Request, dataset: str=None, size: int = None):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)
    try:
        query={}
        if dataset:
            query['dataset']= dataset

        aggregation_pipeline = [{"$match": query}]

        if size:
            aggregation_pipeline.append({"$sample": {"size": size}})

        image_data_list = list(request.app.extracts_collection.aggregate(aggregation_pipeline))

        for image_data in image_data_list:
            image_data.pop('_id', None)  # Remove the auto-generated field

        return api_response_handler.create_success_response_v1(
            response_data={"data": image_data_list},
            http_status_code=200  
        )
    
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )
    
@router.delete("/external-images/delete-extract", 
            description="Delete an extracted image",
            tags=["extracts"],  
            response_model=StandardSuccessResponseV1[WasPresentResponse],  
            responses=ApiResponseHandlerV1.listErrors([404, 422, 500]))
async def delete_external_image_data(request: Request, image_hash: str):
    api_response_handler = await ApiResponseHandlerV1.createInstance(request)

    try:
        result = request.app.extracts_collection.delete_one({
            "image_hash": image_hash
        })
        
        if result.deleted_count == 0:
            return api_response_handler.create_success_delete_response_v1(
                response_data=False, 
                http_status_code=200
            )
        
        return api_response_handler.create_success_delete_response_v1(
                response_data=True, 
                http_status_code=200
            )
    
    except Exception as e:
        return api_response_handler.create_error_response_v1(
            error_code=ErrorCode.OTHER_ERROR, 
            error_string=str(e),
            http_status_code=500
        )
