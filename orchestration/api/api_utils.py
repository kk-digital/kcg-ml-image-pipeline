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

     
