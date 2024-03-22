from pydantic import BaseModel, Field, constr, validator
from typing import List, Union, Optional
import re

class Rankmodel(BaseModel):
    rank_id: Optional[int] = None
    rank_string: str = Field(..., description="Name of the rank")
    rank_category_id: Optional[int] = None
    rank_description: str = Field(..., description="Description of the rank")
    rank_vector_index: Optional[int] = Field(-1, description="rank model vector index")
    deprecated: bool = False
    user_who_created: str = Field(..., description="User who created the rank")
    creation_time: Union[str, None] = None 

    def to_dict(self):
        return {
            "rank_id": self.rank_id,
            "rank_string": self.rank_string,
            "rank_category_id": self.rank_category_id,
            "rank_description": self.rank_description,
            "rank_vector_index": self.rank_vector_index,
            "deprecated": self.deprecated,
            "user_who_created": self.user_who_created,
            "creation_time": self.creation_time
        }


NonEmptyString = constr(strict=True, min_length=1)


class RankListResponse(BaseModel):
    ranks: List[Rankmodel]

class RankRequest(BaseModel):
    rank_string: str = Field(..., description="Name of the rank. Should only contain letters, numbers, hyphens, and underscores.")
    rank_category_id: Optional[int] = Field(None, description="ID of the rank category")
    rank_description: str = Field(..., description="Description of the rank")
    rank_vector_index: Optional[int] = None
    deprecated: bool = False
    user_who_created: NonEmptyString = Field(..., description="Username of the user who created the rank")

    @validator('rank_string')
    def validate_rank_string(cls, value):
        if not re.match(r'^[a-zA-Z0-9_-]+$', value):
            raise ValueError('Invalid rank string')
        return value

class RankCategoryRequest(BaseModel):
    rank_category_string: str = Field(..., description="Name of the rank category. Should only contain letters, numbers, hyphens, and underscores.")
    rank_category_description: str = Field(..., description="Description of the rank category")
    deprecated: bool = False
    user_who_created: NonEmptyString = Field(..., description="Username of the user who created the rank")

    @validator('rank_category_string')
    def validate_rank_string(cls, value):
        if not re.match(r'^[a-zA-Z0-9_-]+$', value):
            raise ValueError('Invalid rank string')
        return value


class RankCategory(BaseModel):
    rank_category_id: Optional[int] = None
    rank_category_string: str = Field(..., description="Name of the rank category")
    rank_category_description: str = Field(..., description="Description of the rank category")
    deprecated: bool = False
    user_who_created: str = Field(..., description="User who created the rank category")
    creation_time: Union[str, None] = None

    def to_dict(self):
        return {
            "rank_category_id": self.rank_category_id,
            "rank_category_string": self.rank_category_string,
            "rank_category_description": self.rank_category_description,
            "deprecated": self.deprecated,
            "user_who_created": self.user_who_created,
            "creation_time": self.creation_time
        }

class RankCategoryListResponse:
    rank_categories: List[RankCategory]


class ImageRank(BaseModel):
    rank_id: Optional[int] = None
    file_path: str
    image_hash: str
    rank_type: int = Field(..., description="1 for positive, 0 for negative")
    user_who_created: str = Field(..., description="User who created the rank")
    creation_time: Union[str, None] = None 
    
    @validator("rank_type")
    def validate_rank_type(cls, value):
        if value not in [0, 1]:
            raise ValueError("rank_type should be either 0 or 1.")
        return value

    def to_dict(self):
        return {
            "rank_id": self.rank_id,
            "file_path": self.file_path,
            "image_hash": self.image_hash,
            "rank_type": self.rank_type,
            "user_who_created": self.user_who_created,
            "creation_time": self.creation_time
        }
    
class ListImageRank(BaseModel):
    images: List[ImageRank]

class RankCountResponse(BaseModel):
    rank_id: int
    count: int