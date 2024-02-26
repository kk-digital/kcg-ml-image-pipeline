from pydantic import BaseModel, Field, constr, validator
from typing import List, Union, Optional
import re

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