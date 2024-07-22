from pydantic import BaseModel, Field, constr, validator
from typing import List, Union, Optional
import re
from datetime import datetime
from orchestration.api.mongo_schemas import ImageMetadata

class AllImagesResponse(BaseModel):
    uuid: int
    index: int
    bucket_id: int
    dataset_id: int
    image_hash: str
    image_path: str
    date: int            


class ListAllImagesResponse(BaseModel):
    images: List[AllImagesResponse]