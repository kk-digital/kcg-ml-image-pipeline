from starlette.responses import Response
import json, typing
import time
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from enum import Enum
import time
from fastapi import Request
from typing import TypeVar, Generic, List
from pydantic import BaseModel
from .mongo_schemas import TagDefinition, TagCategory
from datetime import datetime
from minio import Minio
from dateutil import parser
from datetime import datetime
from .mongo_schemas import ImageTag


class ListImageTag(BaseModel):
     images: List[ImageTag]


class RechableResponse(BaseModel):
    reachable: bool

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

class TagIdResponse(BaseModel):
    tag_id: int

class GetClipPhraseResponse(BaseModel):
    phrase : str
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
    request_url: str
    request_dictionary: dict 
    request_method: str
    request_time_total: float
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
    request_time_total: float
    request_time_start: datetime 
    request_time_finished: datetime
    request_response_code: int 

     
class ApiResponseHandlerV1:
    def __init__(self, request: Request):
        self.request = request
        self.url = str(request.url)
        self.start_time = datetime.now() 

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
        request_dictionary:dict,
        http_status_code: int, 
        headers: dict = {"Cache-Control": "no-store"},
    ):
        # Validate the provided HTTP status code
        if not 200 <= http_status_code < 300:
            raise ValueError("Invalid HTTP status code for a success response. Must be between 200 and 299.")

        response_content = {
            "request_error_string": '',
            "request_error_code": 0, 
            "request_url": self.url,
            "request_dictionary": request_dictionary,  # Or adjust how you access parameters
            "request_method": self.request.method,
            "request_time_total": str(self._elapsed_time()),
            "request_time_start": self.start_time.isoformat(),  
            "request_time_finished": datetime.now().isoformat(), 
            "request_response_code": http_status_code,
            "response": response_data
        }
        return PrettyJSONResponse(status_code=http_status_code, content=response_content, headers=headers)


    def create_success_delete_response_v1(
            self, 
            reachable: bool, 
            request_dictionary:dict,
            http_status_code: int,
            headers: dict = {"Cache-Control": "no-store"} ):
        """Construct a success response for deletion operations."""
        response_content = {
            "request_error_string": '',
            "request_error_code": 0, 
            "request_url": self.url,
            "request_dictionary": request_dictionary,
            "request_method": self.request.method,
            "request_time_total": self._elapsed_time(),
            "request_time_start": self.start_time.isoformat(),
            "request_time_finished": datetime.now().isoformat(),
            "request_response_code": http_status_code,
            "response": {"reachable": reachable}
        }
        return PrettyJSONResponse(status_code=http_status_code, content=response_content, headers=headers)

    def create_error_response_v1(
            self,
            error_code: ErrorCode,
            error_string: str,
            request_dictionary: dict,
            http_status_code: int,
            headers: dict = {"Cache-Control": "no-store"},
        ):
            
            response_content = {
                "request_error_string": error_string,
                "request_error_code": error_code.value,  # Using .name for the enum member name
                "request_url": self.url,
                "request_dictionary": request_dictionary,  # Convert query params to a more usable dict format
                "request_method": self.request.method,
                "request_time_total": str(self._elapsed_time()),
                "request_time_start": self.start_time.isoformat(),
                "request_time_finished": datetime.now().isoformat(),
                "request_response_code": http_status_code
            }
            return PrettyJSONResponse(status_code=http_status_code, content=response_content, headers=headers)

            
def find_or_create_next_folder(client: Minio, bucket: str, base_folder: str) -> str:
    """
    Finds the next folder for storing an image, creating a new one if the last is full.
    """
    try:
        # List objects in the base folder and count them
        objects = client.list_objects(bucket, prefix=base_folder+"/", recursive=True)
        folder_counts = {}
        for obj in objects:
            folder = os.path.dirname(obj.object_name)
            folder_counts[folder] = folder_counts.get(folder, 0) + 1

        # Find the last folder and check if it's full
        if folder_counts:
            sorted_folders = sorted(folder_counts.items(), key=lambda x: x[0])
            last_folder, count = sorted_folders[-1]
            if count < 1000:
                return last_folder
            else:
                # Create a new folder by incrementing the last folder's name
                folder_number = int(last_folder.split('/')[-1]) + 1
                new_folder = f"{base_folder}/{folder_number:04d}"
                return new_folder
        else:
            # No folders exist yet, start with the first one
            return f"{base_folder}/0001"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MinIO error: {e}")        
     
