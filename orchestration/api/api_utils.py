from starlette.responses import Response
import json, typing
import time
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from enum import Enum
import time
from fastapi import Request
from typing import TypeVar, Generic, List, Any, Dict, Optional
from pydantic import BaseModel
from orchestration.api.mongo_schema.tag_schemas import TagDefinition, TagCategory, ImageTag
from orchestration.api.mongo_schema.pseudo_tag_schemas import ImagePseudoTag
from datetime import datetime
from minio import Minio
from dateutil import parser
from datetime import datetime
import os
from typing import List, Union
from urllib.parse import urlparse

class IrrelevantResponse(BaseModel):
    uuid: str
    file_hash: str
    rank_model_id: int


class ListGenerationsCountPerDayResponse(BaseModel):
    results: dict

class JobStatsResponse(BaseModel):
    total: int
    pending_count: int
    progress_count: int
    completed_count: int
    failed_count: int    

class DeletedCount(BaseModel):
    deleted_count: int    

class BoolIrrelevantResponse(BaseModel):
    irrelevant: bool    

class ListIrrelevantResponse(BaseModel):
    images: List[IrrelevantResponse]    

class DoneResponse(BaseModel):
    Done: bool

class DatasetResponse(BaseModel):
    datasets: List[str]

class SeqIdResponse(BaseModel):
    sequential_ids : List[str]
    
class SeqIdDatasetResponse(BaseModel):
    dataset: str
    sequential_id: int

class SetRateResponse(BaseModel):
    dataset: str
    last_update: datetime
    dataset_rate: float
    relevance_model: str
    ranking_model: str

class ResponseRelevanceModel(BaseModel):
    last_update: datetime
    relevance_model: str

class SetHourlyResponse(BaseModel):
    dataset: str
    last_update: datetime
    hourly_limit: int
    relevance_model: str
    ranking_model: str

class HourlyResponse(BaseModel):
    hourly_limit: str

class RateResponse(BaseModel):
    dataset_rate: str

class FilePathResponse(BaseModel):
    file_path: str

class ListFilePathResponse(BaseModel):
    file_paths: List[FilePathResponse]

class DatasetConfig(BaseModel):
    dataset_name: str
    dataset_rate: Union[str, None] = None
    relevance_model: Union[str, None] = None
    ranking_model: Union[str, None] = None
    hourly_limit: Union[int, None] = None
    top_k: Union[int, None] = None
    generation_policy: Union[str, None] = None
    relevance_threshold: Union[int, None] = None

class ResponseDatasetConfig(BaseModel):
    dataset_name: Optional[str]
    last_update: Optional[datetime]
    dataset_rate: Optional[str]
    relevance_model: Optional[str]
    ranking_model: Optional[str]
    hourly_limit: Optional[int]
    top_k: Optional[int]
    generation_policy: Optional[str]
    relevance_threshold: Optional[int]    

class RankinModelResponse(BaseModel):
    last_update: datetime
    ranking_model: str

class ListDatasetConfig(BaseModel):
    configs: List[ResponseDatasetConfig]

class SingleModelResponse(BaseModel):
    model_name: str
    model_architecture: str
    model_creation_date: str
    model_type: str
    model_path: str
    model_file_hash: str
    input_type: str
    output_type: str
    number_of_training_points: str
    number_of_validation_points: str
    training_loss: str
    validation_loss: str
    graph_report: str

class JsonContentResponse(BaseModel):
    json_content: dict

class ModelResponse(BaseModel):
    models: List[SingleModelResponse]

class TagListForImages(BaseModel):
    tags: List[TagDefinition]

class ModelTypeResponse(BaseModel):
    model_types: List[str]
    
class ModelsAndScoresResponse(BaseModel):
    models: List[str]
    scores: List[str]

class ListImageTag(BaseModel):
     images: List[ImageTag]

class RankCountResponse(BaseModel):
    image_hash: str
    count: int

class CountResponse(BaseModel):
    count: int

class RechableResponse(BaseModel):
    reachable: bool

class ResponsePolicies(BaseModel):
    generation_policies: List[str]

class VectorIndexUpdateRequest(BaseModel):
    vector_index: int

class WasPresentResponse(BaseModel):
    wasPresent: bool

class TagsCategoryListResponse(BaseModel):
    tag_categories: List[TagCategory]

class TagsListResponse(BaseModel):
    tags: List[TagDefinition]

class TagCountResponse(BaseModel):
    tag_id: int
    count: int

class ModelIdResponse(BaseModel):
    model_id: int

class UrlResponse(BaseModel):
    url: str

class TagIdResponse(BaseModel):
    tag_id: int

class PseudoTagIdResponse(BaseModel):
    pseudo_tag_id: int

class GetClipPhraseResponse(BaseModel):
    phrase : str
    clip_vector: List[List[float]]

class GetKandinskyClipResponse(BaseModel):
    clip_vector: List[List[float]]

class ImageData(BaseModel):
    image_path: str
    image_hash: str
    score: float

class TagResponse(BaseModel):
    tag_id: int
    tag_string: str 
    tag_type: int
    tag_category_id: int
    tag_description: str  
    tag_vector_index: int
    deprecated: bool = False
    deprecated_tag_category: bool = False
    user_who_created: str
    creation_time: str

class AddJob(BaseModel):
    uuid: str
    creation_time: str

def validate_date_format(date_str: str):
    try:
        # Attempt to parse the date string using dateutil.parser
        parsed_date = parser.parse(date_str)
        # If parsing succeeds, return the original date string
        return date_str
    except ValueError:
        # If parsing fails, return None
        return None


class PrettyJSONResponse(Response):
    media_type = "application/json"

    def render(self, content: typing.Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=4,
            separators=(", ", ": "),
        ).encode("utf-8")


class ErrorCode(Enum):
    SUCCESS = 0
    OTHER_ERROR = 1
    ELEMENT_NOT_FOUND = 2
    INVALID_PARAMS = 3


T = TypeVar('T')
class StandardSuccessResponse(BaseModel, Generic[T]):
    url: str
    duration: int
    response: T


class StandardErrorResponse(BaseModel):
    url: str
    duration: int
    errorCode: int
    errorString: str


class ApiResponseHandler:
    def __init__(self, request: Request):
        self.url = str(request.url)
        self.start_time = time.time()

    def _elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    @staticmethod
    def listErrors(errors: List[int]) -> dict:
        repsonse = {}
        for err in errors:
            repsonse[err] = {"model": StandardErrorResponse}
        return repsonse

    def create_success_response(self, response_data: dict, http_status_code: int, headers: dict = {"Cache-Control": "no-store"}):
        # Validate the provided HTTP status code
        if not 200 <= http_status_code < 300:
            raise ValueError("Invalid HTTP status code for a success response. Must be between 200 and 299.")

        response_content = {
            "url": self.url,
            "duration": self._elapsed_time(),
            "response": response_data
        }
        return PrettyJSONResponse(status_code=http_status_code, content=response_content, headers=headers)
    
    def create_success_delete_response(self, reachable: bool):
        return PrettyJSONResponse(
            status_code=200,
            content={
                "url": self.url,
                "duration": self._elapsed_time(),
                "response": {"reachable": reachable}
            },
            headers={"Cache-Control": "no-store"}
        )

    def create_error_response(self, error_code: ErrorCode, error_string: str, http_status_code: int):
        return PrettyJSONResponse(
            status_code=http_status_code,
            content={
                "url": self.url,
                "duration": self._elapsed_time(),
                "errorCode": error_code.value,
                "errorString": error_string
            }
        )

     


class StandardSuccessResponseV1(BaseModel, Generic[T]):
    request_error_string: str = ""
    request_error_code: int = 0
    request_url: str
    request_dictionary: dict 
    request_method: str
    request_complete_time: float
    request_time_start: datetime 
    request_time_finished: datetime
    request_response_code: int 
    response: T 


class StandardErrorResponseV1(BaseModel):
    request_error_string: str = ""
    request_error_code: int = -1
    request_url: str
    request_dictionary: dict 
    request_method: str
    request_complete_time: float
    request_time_start: str 
    request_time_finished: str
    request_response_code: int 

     
class ApiResponseHandlerV1:
    def __init__(self, request: Request, body_data: Optional[Dict[str, Any]] = None):
        self.request = request
        self.url = str(request.url)
        self.start_time = datetime.now() 
        self.query_params = dict(request.query_params)

        # Parse the URL to extract and store the path
        parsed_url = urlparse(self.url)
        self.url_path = parsed_url.path  # Store the path part of the URL

        self.request_data = {
            "body": body_data or {},  # Set from the provided body data
            "query": dict(request.query_params)  # Extracted from request
        }

    @staticmethod
    async def createInstance(request: Request):
        body = await request.body()
        body_dictionary = {}
        if (len(body) > 0):
            body_string = body.decode('utf-8')
            body_dictionary = json.loads(body_string)

        instance = ApiResponseHandlerV1(request, body_dictionary)
        return instance
    
    # In middlewares, this must be called instead of "createInstance", as "createInstance" may hang trying to get the request body.
    @staticmethod
    def createInstanceWithBody(request: Request, body_data: Dict[str, Any]):
        instance = ApiResponseHandlerV1(request, body_data)
        return instance

    
    def _elapsed_time(self) -> float:
        return datetime.now() - self.start_time
    
    @staticmethod
    def listErrors(errors: List[int]) -> dict:
        repsonse = {}
        for err in errors:
            repsonse[err] = {"model": StandardErrorResponseV1}
        return repsonse

    def create_success_response_v1(
        self,
        response_data: dict,
        http_status_code: int, 
        headers: dict = {"Cache-Control": "no-store"},
    ):
        # Validate the provided HTTP status code
        if not 200 <= http_status_code < 300:
            raise ValueError("Invalid HTTP status code for a success response. Must be between 200 and 299.")

        response_content = {
            "request_error_string": '',
            "request_error_code": 0, 
            "request_url": self.url_path,
            "request_dictionary": self.request_data,  # Or adjust how you access parameters
            "request_method": self.request.method,
            "request_complete_time": str(self._elapsed_time()),
            "request_time_start": self.start_time.isoformat(),  
            "request_time_finished": datetime.now().isoformat(), 
            "request_response_code": http_status_code,
            "response": response_data
        }
        return PrettyJSONResponse(status_code=http_status_code, content=response_content, headers=headers)


    def create_success_delete_response_v1(
            self, 
            wasPresent: bool, 
            http_status_code: int,
            headers: dict = {"Cache-Control": "no-store"} ):
        """Construct a success response for deletion operations."""
        response_content = {
            "request_error_string": '',
            "request_error_code": 0, 
            "request_url": self.url_path,
            "request_dictionary": self.request_data,
            "request_method": self.request.method,
            "request_complete_time": str(self._elapsed_time()),
            "request_time_start": self.start_time.isoformat(),
            "request_time_finished": datetime.now().isoformat(),
            "request_response_code": http_status_code,
            "response": {"wasPresent": wasPresent}
        }
        return PrettyJSONResponse(status_code=http_status_code, content=response_content, headers=headers)

    def create_error_response_v1(
            self,
            error_code: ErrorCode,
            error_string: str,
            http_status_code: int,
            headers: dict = {"Cache-Control": "no-store"},
        ):
            
            response_content = {
                "request_error_string": error_string,
                "request_error_code": error_code.value,  # Using .name for the enum member name
                "request_url": self.url_path,
                "request_dictionary": self.request_data,  # Convert query params to a more usable dict format
                "request_method": self.request.method,
                "request_complete_time": str(self._elapsed_time()),
                "request_time_start": self.start_time.isoformat(),
                "request_time_finished": datetime.now().isoformat(),
                "request_response_code": http_status_code
            }
            return PrettyJSONResponse(status_code=http_status_code, content=response_content, headers=headers)

            

def find_or_create_next_folder_and_index(client: Minio, bucket: str, base_folder: str) -> (str, int):
    """
    Finds the next folder for storing an image, creating a new one if the last is full,
    and determines the next image index.
    """
    try:
        objects = client.list_objects(bucket, prefix=base_folder+"/", recursive=True)
        folder_counts = {}
        latest_index = -1  # Start before the first possible index
        
        for obj in objects:
            folder, filename = os.path.split(obj.object_name)
            folder_counts[folder] = folder_counts.get(folder, 0) + 1
            
            # Attempt to parse the filename as an index
            try:
                index = int(os.path.splitext(filename)[0])
                latest_index = max(latest_index, index)
            except ValueError:
                pass  # Filename isn't a simple integer index

        if folder_counts:
            sorted_folders = sorted(folder_counts.items(), key=lambda x: x[0])
            last_folder, count = sorted_folders[-1]
            if count < 1000:
                return last_folder, latest_index + 1
            else:
                folder_number = int(last_folder.split('/')[-1]) + 1
                new_folder = f"{base_folder}/{folder_number:04d}"
                return new_folder, 0  # Start indexing at 0 for a new folder
        else:
            # No folders exist yet, start with the first one
            return f"{base_folder}/0001", 0
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MinIO error: {e}")
    

class CountLastHour(BaseModel):
    jobs_count: dict

def get_id(bucket: str, dataset: str) -> str:
    return '{}_{}'.format(bucket, dataset)

def get_minio_file_path(seq_id, dataset_name, format, sample_size = 1000):

    folder_id = (seq_id // sample_size) + 1
    file_id = (seq_id % sample_size)
    
    folder_name = f"{folder_id:04d}"
    file_name = f"{file_id:06d}"

    path = f'{dataset_name}/{folder_name}/{file_name}'

    return f'{path}.{format}'

def get_next_seq_id(request: Request, bucket: str, dataset: str):
    # get ingress video counter
    counter = request.app.counters_collection.find_one({"_id": '{}_{}'.format(bucket, dataset)})
    # create counter if it doesn't exist already
    if counter is None:
        request.app.counters_collection.insert_one({"_id": get_id(bucket, dataset), "seq":0})
    counter_seq = counter["seq"] if counter else 0 
    counter_seq += 1
    
    return counter_seq

def update_seq_id(request: Request, bucket:str, dataset: str, seq_id = 0):

    try:
        ret = request.app.counters_collection.update_one(
            {"_id": get_id(bucket, dataset)},
            {"$set": {"seq": seq_id}})
    except Exception as e:
        raise Exception("Updating of  failed: {}".format(e))
    
def get_extenal_img_next_seq_id(request: Request, bucket:str, dataset: str):
    counter = \
        request.app.extenal_img_seq_id_collection.find_one({"bucket": bucket, 
                                                            "dataset": dataset})
    if counter is None:
        request.app.extenal_img_seq_id_collection({"bucket": bucket,
                                                   "dataset": dataset,
                                                   "image_count": 0})
    counter_seq = counter["image_count"] if counter else 0 
    counter_seq += 1
    
    return counter_seq

def update_external_img_next_seq_id(request: Request, bucket:str, dataset: str, seq_id = 0):

    try:
        ret = request.app.extenal_img_seq_id_collection.update_one(
            {"bucket": bucket, "dataset": dataset},
            {"$set": {"image_count": seq_id}})
    except Exception as e:
        raise Exception("Updating of external image sequential id failed: {}".format(e))