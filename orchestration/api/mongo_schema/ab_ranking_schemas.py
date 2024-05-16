from pydantic import BaseModel, Field, constr, validator
from typing import List, Union, Optional
import re
from orchestration.api.mongo_schemas import ImageMetadata

class Rankmodel(BaseModel):
    rank_model_id: Optional[int] = None
    rank_model_string: str = Field(..., description="Name of the rank")
    classifier_id: int
    rank_model_category_id: Optional[int] = None
    rank_model_description: str = Field(..., description="Description of the rank")
    rank_model_vector_index: Optional[int] = Field(-1, description="rank model vector index")
    deprecated: bool = False
    user_who_created: str = Field(..., description="User who created the rank")
    creation_time: Union[str, None] = None 

    def to_dict(self):
        return {
            "rank_model_id": self.rank_model_id,
            "rank_model_string": self.rank_model_string,
            "classifier_id": self.classifier_id,
            "rank_model_category_id": self.rank_model_category_id,
            "rank_model_description": self.rank_model_description,
            "rank_model_vector_index": self.rank_model_vector_index,
            "deprecated": self.deprecated,
            "user_who_created": self.user_who_created,
            "creation_time": self.creation_time
        }


NonEmptyString = constr(strict=True, min_length=1)


class RankListResponse(BaseModel):
    ranks: List[Rankmodel]

class RankRequest(BaseModel):
    rank_model_string: str = Field(..., description="Name of the rank. Should only contain letters, numbers, hyphens, and underscores.")
    classifier_id: Optional[int] = None
    rank_model_category_id: Optional[int] = Field(None, description="ID of the rank category")
    rank_model_description: str = Field(..., description="Description of the rank")
    rank_model_vector_index: Optional[int] = None
    deprecated: bool = False
    user_who_created: NonEmptyString = Field(..., description="Username of the user who created the rank")

    @validator('rank_model_string')
    def validate_rank_model_string(cls, value):
        if not re.match(r'^[a-zA-Z0-9_-]+$', value):
            raise ValueError('Invalid rank string')
        return value

class RankCategoryRequest(BaseModel):
    rank_model_category_string: str = Field(..., description="Name of the rank category. Should only contain letters, numbers, hyphens, and underscores.")
    rank_model_category_description: str = Field(..., description="Description of the rank category")
    deprecated: bool = False
    user_who_created: NonEmptyString = Field(..., description="Username of the user who created the rank")

    @validator('rank_model_category_string')
    def validate_rank_model_string(cls, value):
        if not re.match(r'^[a-zA-Z0-9_-]+$', value):
            raise ValueError('Invalid rank string')
        return value


class RankCategory(BaseModel):
    rank_model_category_id: Optional[int] = None
    rank_model_category_string: str = Field(..., description="Name of the rank category")
    rank_model_category_description: str = Field(..., description="Description of the rank category")
    deprecated: bool = False
    user_who_created: str = Field(..., description="User who created the rank category")
    creation_time: Union[str, None] = None

    def to_dict(self):
        return {
            "rank_model_category_id": self.rank_model_category_id,
            "rank_model_category_string": self.rank_model_category_string,
            "rank_model_category_description": self.rank_model_category_description,
            "deprecated": self.deprecated,
            "user_who_created": self.user_who_created,
            "creation_time": self.creation_time
        }

class RankCategoryListResponse(BaseModel):
    rank_model_categories: List[RankCategory]


class ImageRank(BaseModel):
    rank_model_id: Optional[int] = None
    file_path: str
    image_hash: str
    user_who_created: str = Field(..., description="User who created the rank")
    creation_time: Union[str, None] = None 

    def to_dict(self):
        return {
            "rank_model_id": self.rank_model_id,
            "file_path": self.file_path,
            "image_hash": self.image_hash,
            "user_who_created": self.user_who_created,
            "creation_time": self.creation_time
        }
    
class ListImageRank(BaseModel):
    images: List[ImageRank]

class RankCountResponse(BaseModel):
    rank_model_id: int
    count: int

class RankedSelection(BaseModel):
    rank_model_id: int
    task: str
    username: str
    image_1_metadata: ImageMetadata
    image_2_metadata: ImageMetadata
    selected_image_index: int
    selected_image_hash: str
    datetime: str
    training_mode: str
    active_learning_type: str
    active_learning_policy: str
