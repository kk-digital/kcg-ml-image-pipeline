import sys
base_directory = "./"
sys.path.insert(0, base_directory)


class GenerationTask:
    uuid: str
    task_type: str
    model_name: str
    model_file_name: str
    model_file_path: str
    model_hash: str
    task_creation_time: str
    task_start_time: str
    task_completion_time: str
    task_error_str: str
    task_input_dict: dict  # required
    task_input_file_dict: dict
    task_output_file_dict: dict
    prompt_generation_data: dict

    def __init__(self,
                 uuid,
                 task_type,
                 model_name=None,
                 model_file_name=None,
                 model_file_path=None,
                 model_hash=None,
                 task_creation_time=None,
                 task_start_time=None,
                 task_completion_time=None,
                 task_error_str=None,
                 task_input_dict=None,
                 task_input_file_dict=None,
                 task_output_file_dict=None,
                 prompt_generation_data= None
                 ):
        self.uuid = uuid
        self.task_type = task_type
        self.model_name = model_name
        self.model_file_name = model_file_name
        self.model_file_path = model_file_path
        self.model_hash = model_hash
        self.task_creation_time = task_creation_time
        self.task_start_time = task_start_time
        self.task_completion_time = task_completion_time
        self.task_error_str = task_error_str
        self.task_input_dict = task_input_dict
        self.task_input_file_dict = task_input_file_dict
        self.task_output_file_dict = task_output_file_dict
        self.prompt_generation_data = prompt_generation_data

    def to_dict(self):
        return {
            "uuid": self.uuid,
            "task_type": self.task_type,
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
            "prompt_generation_data": self.prompt_generation_data
        }

    def from_dict(data):
        return GenerationTask(
            uuid=data.get("uuid"),
            task_type=data.get("task_type"),
            model_name=data.get("model_name", ""),
            model_file_name=data.get("model_file_name", ""),
            model_file_path=data.get("model_file_path", ""),
            model_hash=data.get("model_hash", ""),
            task_creation_time=data.get("task_creation_time", ""),
            task_start_time=data.get("task_start_time", ""),
            task_completion_time=data.get("task_completion_time", ""),
            task_error_str=data.get("task_error_str", ""),
            task_input_dict=data.get("task_input_dict", {}),
            task_input_file_dict=data.get("task_input_file_dict", {}),
            task_output_file_dict=data.get("task_output_file_dict", {}),
            prompt_generation_data=data.get("prompt_generation_data", {}),
        )
