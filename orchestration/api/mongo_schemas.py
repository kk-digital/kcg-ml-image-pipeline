from pydantic import BaseModel, Field, constr, validator
from typing import Union, Optional
import re

class Task(BaseModel):
    task_type: str
    uuid: str  # required, should be passed by generator
    model_name: Union[str, None] = None
    model_file_name: Union[str, None] = None
    model_file_path: Union[str, None] = None
    sd_model_hash: Union[str, None] = None
    task_creation_time: Union[str, None] = None
    task_start_time: Union[str, None] = None
    task_completion_time: Union[str, None] = None
    task_error_str: Union[str, None] = None
    task_input_dict: dict  # required
    task_input_file_dict: Union[dict, None] = None
    task_output_file_dict: Union[dict, None] = None
    task_attributes_dict: Union[dict, None] = {}

    def to_dict(self):
        return {
            "task_type": self.task_type,
            "uuid": self.uuid,
            "model_name": self.model_name,
            "model_file_name": self.model_file_name,
            "model_file_path": self.model_file_path,
            "sd_model_hash": self.sd_model_hash,
            "task_creation_time": self.task_creation_time,
            "task_start_time": self.task_start_time,
            "task_completion_time": self.task_completion_time,
            "task_error_str": self.task_error_str,
            "task_input_dict": self.task_input_dict,
            "task_input_file_dict": self.task_input_file_dict,
            "task_output_file_dict": self.task_output_file_dict,
            "task_attributes_dict": self.task_attributes_dict,
        }


class SequentialID:
    dataset_name: str
    subfolder_count: int = 0
    file_count: int = -1

    def __init__(self, dataset_name: str, subfolder_count=1, file_count=-1):
        self.dataset_name = dataset_name
        self.subfolder_count = subfolder_count
        self.file_count = file_count

    def add_count(self):
        max_num_files = 1000

        self.file_count += 1
        if self.file_count != 0 and self.file_count % max_num_files == 0:
            self.subfolder_count += 1

    def get_sequential_id(self) -> str:
        self.add_count()

        return "{0:04}/{1:06}".format(self.subfolder_count, self.file_count)

    def to_dict(self):
        return {
            "dataset_name": self.dataset_name,
            "subfolder_count": self.subfolder_count,
            "file_count": self.file_count
        }


class ImageMetadata(BaseModel):
    file_name: str
    file_hash: Union[str, None] = None
    file_path: Union[str, None] = None
    image_type: Union[str, None] = None
    image_width: Union[str, None] = None
    image_height: Union[str, None] = None
    image_size: Union[str, None] = None
    features_type: Union[str, None] = None
    features_model: Union[str, None] = None
    features_vector: Union[list, None] = None

    def to_dict(self):
        return {
            "file_name": self.file_name,
            "file_hash": self.file_hash,
            "file_path": self.file_path,
            "image_type": self.image_type,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "image_size": self.image_size,
            "features_type": self.features_type,
            "features_model": self.features_model,
            "features_vector": self.features_vector,
        }


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


class TrainingTask(BaseModel):
    uuid: str  # required, should be passed by generator
    model_name: str = Field(pattern="^[a-zA-Z0-9_-]+$")  # Constrains model name to the required characters
    model_task: str = Field(pattern="^(ranking-clip|ranking-embedding|relevance-embedding)$")  # constraints based on the given model tasks
    model_architecture: str 
    dataset: str
    learning_rate: float = 0.01
    weight_decay: float = 0.01
    training_time_kimgs: int = 10
    epochs: int
    train_percent: float
    task_creation_time: Union[str, None] = None
    task_start_time: Union[str, None] = None
    task_completion_time: Union[str, None] = None
    task_error_str: Union[str, None] = None
    task_output_file_dict: Union[dict, None] = None

    def to_dict(self):
        return {
            "uuid": self.uuid,
            "model_name": self.model_name,
            "model_task": self.model_task,
            "model_architecture": self.model_architecture,
            "dataset": self.dataset,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "training_time_kimgs": self.training_time_kimgs,
            "epochs": self.epochs,
            "train_percent": self.train_percent,
            "task_creation_time": self.task_creation_time,
            "task_start_time": self.task_start_time,
            "task_completion_time": self.task_completion_time,
            "task_error_str": self.task_error_str,
            "task_output_file_dict": self.task_output_file_dict,
        }


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
        
class FlaggedDataUpdate(BaseModel):
    flagged: bool = Field(..., description="Indicates whether the data is flagged or not")
    flagged_by_user: str = Field(..., description="User who is flagging the data")
    flagged_time: Optional[str] = None

    def to_dict(self):
        return {
            "flagged": self.flagged,
            "flagged_by_user": self.flagged_by_user,
            "flagged_time": self.flagged_time
        }


class User(BaseModel):
    username: str = Field(...)
    password: str = Field(...)
    "role: constr(pattern='^(admin|user)$') = Field(...)"

    def to_dict(self):
        return {
            "username": self.username,
            "password": self.password,
            "role": self.role
        }
    
class LoginRequest(BaseModel):
    username: str = Field(...)
    password: str = Field(...)

class TokenPayload(BaseModel):
    sub: str = None
    exp: int = None


class RankingModel(BaseModel):
    model_id: int = None
    model_creation_date: str
    model_type: str
    model_path: str
    model_file_hash: str = None
    input_type: str = None
    output_type: str = None
    number_of_training_points: str = None
    number_of_validation_points: str = None
    training_loss: str = None
    validation_loss: str = None
    graph_report: str = None

    def to_dict(self):
        return {
            "model_id": self.model_id,
            "model_creation_date": self.model_creation_date,
            "model_type": self.model_type,
            "model_path": self.model_path,
            "model_file_hash": self.model_file_hash,
            "input_type": self.input_type,
            "output_type": self.output_type,
            "number_of_training_points": self.number_of_training_points,
            "number_of_validation_points": self.number_of_validation_points,
            "training_loss": self.training_loss,
            "validation_loss": self.validation_loss,
            "graph_report": self.graph_report,
        }


class RankingScore(BaseModel):
    model_id: int
    image_hash: str
    score: float

    def to_dict(self):
        return {
            "model_id": self.model_id,
            "image_hash": self.image_hash,
            "score": self.score,
        }


class RankingSigmaScore(BaseModel):
    model_id: int
    image_hash: str
    sigma_score: float

    def to_dict(self):
        return {
            "model_id": self.model_id,
            "image_hash": self.image_hash,
            "sigma_score": self.sigma_score,
        }


class RankingResidual(BaseModel):
    model_id: int
    image_hash: str
    residual: float

    def to_dict(self):
        return {
            "model_id": self.model_id,
            "image_hash": self.image_hash,
            "residual": self.residual,
        }


class RankingPercentile(BaseModel):
    model_id: int
    image_hash: str
    percentile: float

    def to_dict(self):
        return {
            "model_id": self.model_id,
            "image_hash": self.image_hash,
            "percentile": self.percentile,
        }

class RankingResidualPercentile(BaseModel):
    model_id: int
    image_hash: str
    residual_percentile: float

    def to_dict(self):
        return {
            "model_id": self.model_id,
            "image_hash": self.image_hash,
            "residual_percentile": self.residual_percentile,
        }


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


class PhraseModel(BaseModel):
    phrase: str

    def to_dict(self):
        return{
            "phrase": self.phrase
        }