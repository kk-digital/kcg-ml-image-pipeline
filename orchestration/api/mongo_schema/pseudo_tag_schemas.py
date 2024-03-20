from pydantic import BaseModel, Field, constr, validator
from typing import List, Union, Optional
import re
from datetime import datetime

class PseudoTagDefinition(BaseModel):
    pseudo_tag_id: Optional[int] = None
    pseudo_tag_string: str = Field(..., description="Name of the pseudo tag")
    pseudo_tag_category_id: Optional[int] = None
    pseudo_tag_description: str = Field(..., description="Description of the pseudo tag")
    pseudo_tag_vector_index: Optional[int] = Field(-1, description="pseudo Tag vector index")
    deprecated: bool = False
    user_who_created: str = Field(..., description="User who created the pseudo tag")
    creation_time: Union[str, None] = None 

    def to_dict(self):
        return {
            "pseudo_tag_id": self.pseudo_tag_id,
            "pseudo_tag_string": self.pseudo_tag_string,
            "pseudo_tag_category_id": self.pseudo_tag_category_id,
            "pseudo_tag_description": self.pseudo_tag_description,
            "pseudo_tag_vector_index": self.pseudo_tag_vector_index,
            "deprecated": self.deprecated,
            "user_who_created": self.user_who_created,
            "creation_time": self.creation_time
        }


NonEmptyString = constr(strict=True, min_length=1)

class PseudoTagListForImages(BaseModel):
    pseudotags: List [PseudoTagDefinition]

class NewPseudoTagRequest(BaseModel):
    pseudo_tag_string: str = Field(..., description="Name of the tag. Should only contain letters, numbers, hyphens, and underscores.")
    pseudo_tag_category_id: Optional[int] = Field(None, description="ID of the tag category")
    pseudo_tag_description: str = Field(..., description="Description of the tag")
    pseudo_tag_vector_index: Optional[int] = None
    deprecated: bool = False
    user_who_created: NonEmptyString = Field(..., description="Username of the user who created the tag")

    @validator('pseudo_tag_string')
    def validate_tag_string(cls, value):
        if not re.match(r'^[a-zA-Z0-9_-]+$', value):
            raise ValueError('Invalid tag string')
        return value

class NewPseudoTagCategory(BaseModel):
    pseudo_tag_category_string: str = Field(..., description="Name of the tag category. Should only contain letters, numbers, hyphens, and underscores.")
    pseudo_tag_category_description: str = Field(..., description="Description of the tag category")
    deprecated: bool = False
    user_who_created: NonEmptyString = Field(..., description="Username of the user who created the tag")

    @validator('pseudo_tag_category_string')
    def validate_tag_string(cls, value):
        if not re.match(r'^[a-zA-Z0-9_-]+$', value):
            raise ValueError('Invalid tag string')
        return value


class PseudoTagCategory(BaseModel):
    pseudo_tag_category_id: Optional[int] = None
    pseudo_tag_category_string: str = Field(..., description="Name of the pesudo tag category")
    pseudo_tag_category_description: str = Field(..., description="Description of the pseudo tag category")
    deprecated: bool = False
    user_who_created: str = Field(..., description="User who created the pesudo tag category")
    creation_time: Union[str, None] = None

    def to_dict(self):
        return {
            "pseudo_tag_category_id": self.pseudo_tag_category_id,
            "pseudo_tag_category_string": self.pseudo_tag_category_string,
            "pseudo_tag_category_description": self.pseudo_tag_category_description,
            "deprecated": self.deprecated,
            "user_who_created": self.user_who_created,
            "creation_time": self.creation_time
        }


class ImagePseudoTag(BaseModel):
    pseudo_tag_id: Optional[int] = None
    file_path: str
    image_hash: str
    user_who_created: str = Field(..., description="User who created the pseudo_tag")
    creation_time: Union[str, None] = None 
    

    def to_dict(self):
        return {
            "pseudo_tag_id": self.pseudo_tag_id,
            "file_path": self.file_path,
            "image_hash": self.image_hash,
            "user_who_created": self.user_who_created,
            "creation_time": self.creation_time
        }
    
class PseudoTagScore(BaseModel):
    uuid: str
    tag_id: int
    score: float
    classifier_name: str

    def to_dict(self):
        return {
            "uuid": self.uuid,
            "tag_id": self.tag_id,
            "classifier": self.classifier_name,
            "score": self.score,
            "creation_time": datetime.utcnow().isoformat()  # Handle inside API
        }

class ListImagePseudoTag(BaseModel):
     images: List[ImagePseudoTag]