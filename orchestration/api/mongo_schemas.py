from pydantic import BaseModel, Field, constr, validator, model_validator
from typing import List, Union, Optional, Dict
import json
import re
from typing import Union, Tuple
from datetime import datetime
import bson

class Task(BaseModel):
    task_type: str
    uuid: str  # required, should be passed by generator
    model_name: Union[str, None] = None
    model_file_name: Union[str, None] = None
    model_file_path: Union[str, None] = None
    model_hash: Union[str, None] = None
    task_creation_time: Union[datetime, None] = None
    task_start_time: Union[datetime, None] = None
    task_completion_time: Union[datetime, None] = None
    task_error_str: Union[str, None] = None
    task_input_dict: dict  # required
    task_input_file_dict: Union[dict, None] = None
    task_output_file_dict: Union[dict, None] = None
    task_attributes_dict: Union[dict, None] = {}
    prompt_generation_data: Union[dict, None] = {}

    @validator('task_creation_time', 'task_start_time', 'task_completion_time', pre=True, always=True)
    def default_datetime(cls, value):
        return value or datetime.now()

    @validator('task_creation_time', 'task_start_time', 'task_completion_time', pre=False)
    def format_datetime(cls, value):
        if isinstance(value, datetime):
            return value.isoformat()
        return value
    
    def to_dict(self):
        return {
            "task_type": self.task_type,
            "uuid": self.uuid,
            "model_name": self.model_name,
            "model_file_name": self.model_file_name,
            "model_file_path": self.model_file_path,
            "model_hash": self.model_hash,
            "task_creation_time": self.task_creation_time,
            "task_start_time": self.task_start_time,
            "task_completion_time": self.task_completion_time,
            "task_error_str": self.task_error_str,
            "task_input_dict": self.task_input_dict,
            "task_input_file_dict": self.task_input_file_dict,
            "task_output_file_dict": self.task_output_file_dict,
            "task_attributes_dict": self.task_attributes_dict,
            "prompt_generation_data": self.prompt_generation_data
        }

    @model_validator(mode='before')
    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


class ListTask(BaseModel):
    jobs: List[Task]

class KandinskyTask(BaseModel):
    job: Task # task data
    positive_embedding: list
    negative_embedding: Union[list, None] = None

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

class UUIDImageMetadata(BaseModel):
    uuid: str
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
            "uuid": self.uuid,
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
    
class ResponseRankingScore(BaseModel):
    scores: List[RankingScore]  

class ClassifierScore(BaseModel):
    uuid: Union[str, None]
    classifier_id: int
    image_hash: str
    tag_id: int
    score: float

    def to_dict(self):
        return {
            "uuid": self.uuid,
            "classifier_id": self.classifier_id,
            "image_hash": self.image_hash,
            "tag_id": self.tag_id,
            "score": self.score,
        }


class ImageHash(BaseModel):
    image_hash: str
    image_global_id: int

    def to_dict_for_mongodb(self):
        return {
            "image_hash": self.image_hash,
            "image_global_id":  bson.int64.Int64(self.image_global_id)
        }
    
    def to_dict(self):
        return {
            "image_hash": self.image_hash,
            "image_global_id": self.image_global_id,
        }
    
class ImageHashRequest(BaseModel):
    image_hash: str

    def to_dict(self):
        return {
            "image_hash": self.image_hash,
        }

class ClassifierScoreV1(BaseModel):
    uuid: Union[str, None]
    task_type: str
    classifier_id: int
    image_hash: str
    tag_id: int
    score: float
    creation_time: Union[str, None] = None

    def to_dict(self):
        return {
            "uuid": self.uuid,
            "task_type": self.task_type,
            "classifier_id": self.classifier_id,
            "image_hash": self.image_hash,
            "tag_id": self.tag_id,
            "score": self.score,
            "creation_time" : self.creation_time
        }

class ImageResolution(BaseModel):
    width: int
    height: int

    def to_dict(self):
        return {
            "width": self.width,
            "height": self.height
        }
    
class ExternalImageData(BaseModel):
    image_hash: str
    dataset:str
    image_resolution: ImageResolution
    image_format: str
    file_path: str
    upload_date: Union[str, None] = None
    source_image_dict: dict
    task_attributes_dict: dict

    def to_dict(self):
        return {
            "dataset": self.dataset,
            "upload_date": self.upload_date,
            "image_hash": self.image_hash,
            "image_resolution": self.image_resolution.to_dict(),
            "image_format": self.image_format,
            "file_path": self.file_path,
            "source_image_dict": self.source_image_dict,
            "task_attributes_dict": self.task_attributes_dict
        }

    
class ListClassifierScore(BaseModel):
    images: List[ClassifierScore]

class ListClassifierScore1(BaseModel):
    images: List[ClassifierScoreV1]

class ListClassifierScore2(BaseModel):
    scores: List[ClassifierScoreV1]

class ClassifierScoreRequest(BaseModel):
    job_uuid: Union[str, None]
    classifier_id: int
    score: float

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

class ResponseRankingSigmaScore(BaseModel):
    scores: List[RankingSigmaScore]

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

class ResponseRankingResidual(BaseModel):
    residuals: List[RankingResidual]

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

class ResponseRankingPercentile(BaseModel):
    percentile: List[RankingPercentile]
    
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


class PhraseModel(BaseModel):
    phrase: str

    def to_dict(self):
        return{
            "phrase": self.phrase
        }
    
class DatapointDeltaScore(BaseModel):
    model_type: str
    file_name: str
    delta_score: float

    def to_dict(self):
        return{
            "model_type": self.model_type,
            "file_name": self.file_name,
            "delta_score": self.delta_score
        }    

class Classifier(BaseModel):
        classifier_id: Union[int, None] = None
        classifier_name: str
        tag_id: int
        model_sequence_number: Union[int, None] = None
        latest_model: str
        model_path: str
        creation_time: str

        def to_dict(self):
            return{
                "classifier_id": self.classifier_id,
                "classifier_name": self.classifier_name,
                "tag_id": self.tag_id,
                "model_sequence_number": self.model_sequence_number,
                "latest_model": self.latest_model,
                "model_path": self.model_path,
                "creation_time": self.creation_time
            }

class ListClassifier(BaseModel):
    classifiers : List[Classifier]    

class RequestClassifier(BaseModel):
        
        classifier_name: str
        tag_id: int
        latest_model: str
        model_path: str

        def to_dict(self):
            return{
                "classifier_name": self.classifier_name,
                "tag_id": self.tag_id,
                "latest_model": self.latest_model,
                "model_path": self.model_path,
            }    

class UpdateClassifier(BaseModel):
        classifier_name: str
        latest_model: str
        model_path: str

        def to_dict(self):
            return{
                "classifier_name": self.classifier_name,
                "latest_model": self.latest_model,
                "model_path": self.model_path,
            }   

class Worker(BaseModel):
    last_seen: Union[str, None] = None
    worker_id: str
    worker_type: str
    worker_address: Optional[str] = None
    worker_computer_id: str
    worker_ip: Optional[str] = None

    def to_dict(self):
        return{
            "last_seen": self.last_seen,
            "worker_id": self.worker_id,
            "worker_type": self.worker_type,
            "worker_address": self.worker_address,
            "worker_computer_id": self.worker_computer_id,
            "worker_ip": self.worker_ip
        }   
    
class ListWorker(BaseModel):
    worker: List[Worker]    


class SigmaScoreResponse(BaseModel):
    dataset: str
    job_uuid: str
    file_hash: str
    file_path: str
    clip_sigma_score: float

    def to_dict(self):
        return{
            "job_uuid": self.job_uuid,
            "file_hash": self.file_hash,
            "file_path": self.file_path,
            "clip_sigma_score": self.clip_sigma_score
        }

class ListSigmaScoreResponse(BaseModel):
    job_info: List[SigmaScoreResponse]

class RankActiveLearningPolicy(BaseModel):
    rank_active_learning_policy_id: Union[int, None] = None 
    rank_active_learning_policy: str
    rank_active_learning_policy_description: str
    creation_time: Union[str, None] = None 

    def to_dict(self):
        return{
            "rank_active_learning_policy_id": self.rank_active_learning_policy_id,
            "rank_active_learning_policy": self.rank_active_learning_policy,
            "rank_active_learning_policy_description": self.rank_active_learning_policy_description,
            "creation_time": self.creation_time
        }

class ListRankActiveLearningPolicy(BaseModel):
    policies: List[RankActiveLearningPolicy]

class RequestRankActiveLearningPolicy(BaseModel):
    rank_active_learning_policy: str
    rank_active_learning_policy_description: str

    def to_dict(self):
        return{
            "rank_active_learning_policy": self.rank_active_learning_policy,
            "rank_active_learning_policy_description": self.rank_active_learning_policy_description,
        }

class Dataset(BaseModel):
    dataset_name: str

    def to_dict(self):
        return{
            "dataset_name": self.dataset_name
        }    
class ListDataset(BaseModel):
    datasets: List[Dataset]   