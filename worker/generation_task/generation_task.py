import sys
base_directory = "./"
sys.path.insert(0, base_directory)

from worker.prompt_generation.prompt_generator import generate_image_generation_jobs_using_generated_prompts, \
    generate_inpainting_generation_jobs_using_generated_prompts


class GenerationTask:
    uuid: str
    task_type: str
    model_name: str
    model_file_name: str
    model_file_path: str
    sd_model_hash: str
    task_creation_time: str
    task_start_time: str
    task_completion_time: str
    task_error_str: str
    task_input_dict: dict  # required
    task_input_file_dict: dict
    task_output_file_dict: dict

    def __init__(self,
                 uuid,
                 task_type,
                 model_name=None,
                 model_file_name=None,
                 model_file_path=None,
                 sd_model_hash=None,
                 task_creation_time=None,
                 task_start_time=None,
                 task_completion_time=None,
                 task_error_str=None,
                 task_input_dict=None,
                 task_input_file_dict=None,
                 task_output_file_dict=None
                 ):
        self.uuid = uuid
        self.task_type = task_type
        self.model_name = model_name
        self.model_file_name = model_file_name
        self.model_file_path = model_file_path
        self.sd_model_hash = sd_model_hash
        self.task_creation_time = task_creation_time
        self.task_start_time = task_start_time
        self.task_completion_time = task_completion_time
        self.task_error_str = task_error_str
        self.task_input_dict = task_input_dict
        self.task_input_file_dict = task_input_file_dict
        self.task_output_file_dict = task_output_file_dict

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

    def from_dict(data):
        return GenerationTask(
            uuid=data.get("uuid"),
            task_type=data.get("task_type"),
            model_name=data.get("model_name", ""),
            model_file_name=data.get("model_file_name", ""),
            model_file_path=data.get("model_file_path", ""),
            sd_model_hash=data.get("sd_model_hash", ""),
            task_creation_time=data.get("task_creation_time", ""),
            task_start_time=data.get("task_start_time", ""),
            task_completion_time=data.get("task_completion_time", ""),
            task_error_str=data.get("task_error_str", ""),
            task_input_dict=data.get("task_input_dict", {}),
            task_input_file_dict=data.get("task_input_file_dict", {}),
            task_output_file_dict=data.get("task_output_file_dict", {}),
        )


def run_generate_image_generation_task(generation_task: GenerationTask):
    generate_image_generation_jobs_using_generated_prompts(
        csv_dataset_path=generation_task.task_input_dict["csv_dataset_path"],
        prompt_count=generation_task.task_input_dict["prompt_count"],
        dataset_name=generation_task.task_input_dict["dataset_name"],
        positive_prefix=generation_task.task_input_dict["positive_prefix"]
    )


def run_generate_inpainting_generation_task(generation_task: GenerationTask):
    generate_inpainting_generation_jobs_using_generated_prompts(
        csv_dataset_path=generation_task.task_input_dict["csv_dataset_path"],
        prompt_count=generation_task.task_input_dict["prompt_count"],
        dataset_name=generation_task.task_input_dict["dataset_name"],
        positive_prefix=generation_task.task_input_dict["positive_prefix"],
        init_img_path=generation_task.task_input_dict["init_img_path"],
        mask_path=generation_task.task_input_dict["mask_path"],
    )