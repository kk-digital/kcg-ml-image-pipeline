from pydantic import BaseModel, Field, constr, validator
from typing import List, Union, Optional
import re
from orchestration.api.mongo_schemas import ImageMetadata

class Selection(BaseModel):
    task: str
    username: str
    image_1_metadata: ImageMetadata
    image_2_metadata: ImageMetadata
    selected_image_index: Union[int, None] = None
    selected_image_hash: Union[str, None] = None
    datetime: Union[str, None] = None
    training_mode: Union[str, None] = None
    active_learning_type: Union[str, None] = None
    active_learning_policy: Union[str, None] = None

    def to_dict(self):
        return {
            "task": self.task,
            "username": self.username,
            "image_1_metadata": self.image_1_metadata.to_dict(),
            "image_2_metadata": self.image_2_metadata.to_dict(),
            "selected_image_index": self.selected_image_index,
            "selected_image_hash": self.selected_image_hash,
            "datetime": self.datetime,
            "training_mode": self.training_mode,
            "active_learning_type": self.active_learning_type,
            "active_learning_policy": self.active_learning_policy,
        }



class RelevanceSelection(BaseModel):
    username: str
    image_hash: str
    image_path: str
    relevance: int  # if relevant, should be 1, otherwise 0
    datetime: Union[str, None] = None

    def to_dict(self):
        return {
            "username": self.username,
            "image_hash": self.image_hash,
            "image_path": self.image_path,
            "relevance": self.relevance,
            "datetime": self.datetime,
        }