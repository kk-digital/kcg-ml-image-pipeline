from pydantic import BaseModel, Field
from typing import Union


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

    def to_dict(self):
        return {
            "task": self.task,
            "username": self.username,
            "image_1_metadata": self.image_1_metadata.to_dict(),
            "image_2_metadata": self.image_2_metadata.to_dict(),
            "selected_image_index": self.selected_image_index,
            "selected_image_hash": self.selected_image_hash,
            "datetime": self.datetime,
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
    buffer_size: int
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
            "buffer_size": self.buffer_size,
            "train_percent": self.train_percent,
            "task_creation_time": self.task_creation_time,
            "task_start_time": self.task_start_time,
            "task_completion_time": self.task_completion_time,
            "task_error_str": self.task_error_str,
            "task_output_file_dict": self.task_output_file_dict,
        }
