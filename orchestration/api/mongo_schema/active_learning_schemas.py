from pydantic import BaseModel, Field, constr, validator
from typing import List, Union, Optional
import re
from orchestration.api.mongo_schemas import ImageMetadata


class ActiveLearningPolicy(BaseModel):
    active_learning_policy_id: Union[int, None] = None 
    active_learning_policy: str
    active_learning_policy_description: str
    creation_time: Union[str, None] = None 

    def to_dict(self):
        return{
            "active_learning_policy_id": self.active_learning_policy_id,
            "active_learning_policy": self.active_learning_policy,
            "active_learning_policy_description": self.active_learning_policy_description,
            "creation_time": self.creation_time
        }

class ListActiveLearningPolicy(BaseModel):
    policies: List[ActiveLearningPolicy]

class RequestActiveLearningPolicy(BaseModel):
    active_learning_policy: str
    active_learning_policy_description: str

    def to_dict(self):
        return{
            "active_learning_policy": self.active_learning_policy,
            "active_learning_policy_description": self.active_learning_policy_description,
        }

class ActiveLearningQueuePair(BaseModel):
    image1_job_uuid: str
    image2_job_uuid: str
    active_learning_policy_id: int
    metadata: str
    generator_string: str
    creation_time: Union[str, None] = None 

    def to_dict(self):
        return{
            "image1_job_uuid": self.image1_job_uuid,
            "image2_job_uuid": self.image2_job_uuid,
            "active_learning_policy_id": self.active_learning_policy_id,
            "metadata": self.metadata,
            "generator_string":self.generator_string,
            "creation_time": self.creation_time
        }
    
class RankActiveLearningPair(BaseModel):
    file_name: str
    rank_model_id: int
    rank_model_string: str
    active_learning_policy_id: int
    active_learning_policy: str
    dataset_name: str
    metadata: str
    generation_string: str
    creation_date: str
    images: List[str]

    def to_dict(self):
        return {
            "file_name": self.file_name,
            "rank_model_id": self.rank_model_id,
            "rank_model_string": self.rank_model_string,
            "active_learning_policy_id": self.active_learning_policy_id,
            "active_learning_policy": self.active_learning_policy,
            "dataset_name": self.dataset_name,
            "metadata": self.metadata,
            "generation_string": self.generation_string,
            "creation_date": self.creation_date,
            "images": self.images
        }
    

class ListRankActiveLearningPair(BaseModel):
    datapoints: List[RankActiveLearningPair]


class RankSelection(BaseModel):
    rank_model_id: int
    rank_model_string: str
    task: str
    username: str
    image_1_metadata: ImageMetadata
    image_2_metadata: ImageMetadata
    selected_image_index: Union[int, None] = None
    selected_image_hash: Union[str, None] = None
    datetime: Union[str, None] = None
    training_mode: Union[str, None] = None
    active_learning_policy_id: Union[int, None] = None
    active_learning_policy: Union[str, None] = None

    def to_dict(self):
        return {
            "rank_model_id": self.rank_model_id,
            "rank_model_string": self.rank_model_string,
            "task": self.task,
            "username": self.username,
            "image_1_metadata": self.image_1_metadata.to_dict(),
            "image_2_metadata": self.image_2_metadata.to_dict(),
            "selected_image_index": self.selected_image_index,
            "selected_image_hash": self.selected_image_hash,
            "datetime": self.datetime,
            "training_mode": self.training_mode,
            "active_learning_type": self.active_learning_policy_id,
            "active_learning_policy": self.active_learning_policy,
        }
    
class ListRankSelection(BaseModel):
    datapoints: List[RankSelection]    