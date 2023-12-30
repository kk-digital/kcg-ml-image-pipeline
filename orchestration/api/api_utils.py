from starlette.responses import Response
import json, typing
import time
from fastapi import HTTPException
from fastapi.responses import JSONResponse

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

class ApiResponseHandler:
    def __init__(self, url: str):
        self.url = url
        self.start_time = time.time()

    def _elapsed_time(self) -> float:
        return time.time() - self.start_time

    def create_success_response(self, response_data: dict):
        return PrettyJSONResponse(
            status_code=200,
            content={
                "url": self.url,
                "duration": self._elapsed_time(),
                "response": response_data
            }
        )

    def create_error_response(self, error_code: int, error_string: str):
        return PrettyJSONResponse(
            status_code=error_code,
            content={
                "url": self.url,
                "duration": self._elapsed_time(),
                "errorCode": error_code,
                "errorString": error_string
            }
        )        