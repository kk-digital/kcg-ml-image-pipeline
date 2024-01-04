from starlette.responses import Response
import json, typing
import time
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from enum import Enum
import time

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

class ApiResponseHandler:
    def __init__(self, url: str):
        self.url = url
        self.start_time = time.time()

    def _elapsed_time(self) -> float:
        return time.time() - self.start_time

    def create_success_response(self, response_data: dict, headers: dict = None):
        response_content = {
            "url": self.url,
            "duration": self._elapsed_time(),
            "response": response_data
        }
        if headers:
            return PrettyJSONResponse(status_code=200, content=response_content, headers=headers)
        else:
            return PrettyJSONResponse(status_code=200, content=response_content)


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
     