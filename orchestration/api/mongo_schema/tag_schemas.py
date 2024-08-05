from pydantic import BaseModel, Field, constr, validator
from typing import List, Union, Optional, Dict
import re

class TagDefinition(BaseModel):
    tag_id: Optional[int] = None
    tag_string: str = Field(..., description="Name of the tag")
    tag_category_id: Optional[int] = None
    tag_description: str = Field(..., description="Description of the tag")
    tag_vector_index: Optional[int] = Field(-1, description="Tag definition vector index")
    deprecated: bool = False
    user_who_created: str = Field(..., description="User who created the tag")
    creation_time: Union[str, None] = None 

    def to_dict(self):
        return {
            "tag_id": self.tag_id,
            "tag_string": self.tag_string,
            "tag_category_id": self.tag_category_id,
            "tag_description": self.tag_description,
            "tag_vector_index": self.tag_vector_index,
            "deprecated": self.deprecated,
            "user_who_created": self.user_who_created,
            "creation_time": self.creation_time
        }


NonEmptyString = constr(strict=True, min_length=1)

class NewTagRequest(BaseModel):
    tag_string: str = Field(..., description="Name of the tag. Should only contain letters, numbers, hyphens, and underscores.")
    tag_category_id: Optional[int] = Field(None, description="ID of the tag category")
    tag_description: str = Field(..., description="Description of the tag")
    tag_vector_index: Optional[int] = None
    deprecated: bool = False
    user_who_created: NonEmptyString = Field(..., description="Username of the user who created the tag")

    @validator('tag_string')
    def validate_tag_string(cls, value):
        if not re.match(r'^[a-zA-Z0-9_-]+$', value):
            raise ValueError('Invalid tag string')
        return value

class NewTagCategory(BaseModel):
    tag_category_string: str = Field(..., description="Name of the tag category. Should only contain letters, numbers, hyphens, and underscores.")
    tag_category_description: str = Field(..., description="Description of the tag category")
    deprecated: bool = False
    user_who_created: NonEmptyString = Field(..., description="Username of the user who created the tag")

    @validator('tag_category_string')
    def validate_tag_string(cls, value):
        if not re.match(r'^[a-zA-Z0-9_-]+$', value):
            raise ValueError('Invalid tag string')
        return value


class TagCategory(BaseModel):
    tag_category_id: Optional[int] = None
    tag_category_string: str = Field(..., description="Name of the tag category")
    tag_category_description: str = Field(..., description="Description of the tag category")
    deprecated: bool = False
    user_who_created: str = Field(..., description="User who created the tag category")
    creation_time: Union[str, None] = None

    def to_dict(self):
        return {
            "tag_category_id": self.tag_category_id,
            "tag_category_string": self.tag_category_string,
            "tag_category_description": self.tag_category_description,
            "deprecated": self.deprecated,
            "user_who_created": self.user_who_created,
            "creation_time": self.creation_time
        }


class ImageTag(BaseModel):
    tag_id: Optional[int] = None
    file_path: str
    image_hash: str
    tag_type: int = Field(..., description="1 for positive, 0 for negative")
    user_who_created: str = Field(..., description="User who created the tag")
    creation_time: Union[str, None] = None 
    
    @validator("tag_type")
    def validate_tag_type(cls, value):
        if value not in [0, 1]:
            raise ValueError("tag_type should be either 0 or 1.")
        return value

    def to_dict(self):
        return {
            "tag_id": self.tag_id,
            "file_path": self.file_path,
            "image_hash": self.image_hash,
            "tag_type": self.tag_type,
            "user_who_created": self.user_who_created,
            "creation_time": self.creation_time
        }

class ImageTagResponse(BaseModel):
    tag_id: int
    file_path: str
    image_hash: str
    tag_type: int
    image_source: str
    user_who_created: str
    tag_count: int
    creation_time: str

class ListImageTagResponse(BaseModel):
    images: List[ImageTagResponse]   
    
class ListImageTag(BaseModel):
    images: List[ImageTag]    
    
class ExternalImageTag(BaseModel):
    tag_id: Optional[int] = None
    file_path: str
    image_hash: str
    tag_type: int = Field(..., description="1 for positive, 0 for negative")
    image_source: str
    user_who_created: str = Field(..., description="User who created the tag")
    creation_time: Union[str, None] = None 
    
    @validator("tag_type")
    def validate_tag_type(cls, value):
        if value not in [0, 1]:
            raise ValueError("tag_type should be either 0 or 1.")
        return value

    def to_dict(self):
        return {
            "tag_id": self.tag_id,
            "file_path": self.file_path,
            "image_hash": self.image_hash,
            "tag_type": self.tag_type,
            "image_source": self.image_source,
            "user_who_created": self.user_who_created,
            "creation_time": self.creation_time
        }    
    
class ListExternalImageTag(BaseModel):
    images: List[ExternalImageTag]   


class TagDefinitionV1(BaseModel):
    file_hash: str
    tag_id: int
    tag_string: str
    tag_category_id: int
    tag_description: str 
    tag_vector_index: int
    deprecated: bool 
    user_who_created: str 
    creation_time: str

class TagListForImagesV1(BaseModel):
    images: List[TagDefinitionV1]


class TagListForImages(BaseModel):
    tags: List[TagDefinition]

class TagDefinitionV2(BaseModel):
    tag_id: int
    tag_string: str
    tag_type: int
    tag_category_id: int
    tag_description: str 
    tag_vector_index: int
    deprecated: bool 
    deprecated_tag_category: bool
    user_who_created: str 
    creation_time: str

class TagListForImagesV2(BaseModel):
    tags: List[TagDefinitionV2]

class TagListForImages3(BaseModel):
    file_hash: str 
    tags: List[TagDefinitionV2]

class ListTagListForImages3(BaseModel):
    images: List[TagListForImages3]  

class ListImageTag(BaseModel):
     images: List[ImageTag]


class TagsCategoryListResponse(BaseModel):
    tag_categories: List[TagCategory]

class TagsListResponse(BaseModel):
    tags: List[TagDefinition]

class TagCountResponse(BaseModel):
    tag_id: int
    count: dict

class ImageTypes(BaseModel):
    generated_image: str 
    extract_image: str
    external_image: str

class TagCountResponseV1(BaseModel):
    tag_id: int
    count: ImageTypes


class TagIdResponse(BaseModel):
    tag_id: int


class TagResponse(BaseModel):
    tag_id: int
    tag_string: str 
    tag_type: int
    tag_category_id: int
    tag_description: str  
    tag_vector_index: int
    deprecated: bool = False
    deprecated_tag_category: bool = False
    user_who_created: str
    creation_time: str    

