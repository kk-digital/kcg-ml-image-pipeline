from pydantic import BaseModel, Field, constr, validator
from typing import List, Union, Optional
import re
from datetime import datetime

class Bucket(BaseModel):
    bucket_name: str

    def to_dict(self):
        return{
            "bucket_name": self.bucket_name
        }

class ResponseBucket(BaseModel):
    bucket_id: int
    bucket_name: str
    creation_time: datetime

class ListResponseBucket(BaseModel):
    buckets: List[ResponseBucket]   