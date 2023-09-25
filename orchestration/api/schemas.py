from pydantic import BaseModel
from typing import Union


class Task(BaseModel):
    uuid: Union[str, None] = None
    task_type: str
    model_name: Union[str, None] = None
    model_file_name: Union[str, None] = None
    model_file_path: Union[str, None] = None
    sd_model_hash: Union[str, None] = None
    task_creation_time: Union[str, None] = None
    task_start_time: Union[str, None] = None
    task_completion_time: Union[str, None] = None
    task_error_str: Union[str, None] = None
    task_input_dict:Union[ dict, None] = None
    task_input_file_dict: Union[dict, None] = None
    task_output_file_dict: Union[dict, None] = None

    def to_dict(self):
        return {
            "uuid": self.uuid,
            "task_type": self.task_type,
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
