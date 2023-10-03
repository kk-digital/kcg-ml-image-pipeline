from enum import Enum
from typing import List
import sys
import base64
base_directory = "./"
sys.path.insert(0, base_directory)

from app_ranking.http.request import http_get_random_image, http_get_datasets, http_add_selection

# Create a enum State Error: Rank, User Name, None
class StateErrorType(Enum):
    RANK = 1
    USER_NAME = 2
    NONE = 3
    DATASET = 4


class StateError():
    def __init__(self, state_error_type: StateErrorType, error_message: str):
        self.state_error_type = state_error_type
        self.error_message = error_message


class State:
    def __init__(self, user_name: str = ""):
        self._imgs: List = []
        self._user_name = user_name
        self._dataset = None
        self._error = None

    @classmethod
    def from_dict(cls, data: dict):
        state = cls()
        state._imgs = data["_imgs"]
        state._user_name = data["_user_name"]
        state._dataset = data["_dataset"]
        return state

    @property
    def user_name(self):
        return self._user_name

    def set_user_name(self, name: str):
        if name:
            self.clear_errors()
        self._user_name = name

    @property
    def dataset(self):
        return self._dataset

    def set_dataset(self, dataset):
        self._dataset = dataset

    @property
    def imgs(self) -> List:
        return self._imgs

    @imgs.setter
    def imgs(self, idx: list):
        self._imgs = idx

    @property
    def has_error(self) -> bool:
        return self._error is not None

    def get_error(self, state_error_type: StateErrorType):
        return self._error and self._error.state_error_type == state_error_type and self._error.error_message or None

    def set_error(self, error: StateError):
        self._error = error

    def clear_errors(self):
        self._error = None

    def reset(self):
        self._imgs = []

    def copy(self):
        new_state = State.from_dict(self.__dict__)
        return new_state

    def copy_with_error(self):
        new_state = self.copy()
        new_state.set_error(self._error)
        return new_state


class StateController:
    def __init__(self):
        self.datasets = []
        self.dataset_name = "icons"  # default to icons for now

    def rand_select(self, state: State):
        if state.has_error:
            return

        if self.dataset_name == None:
            return

        # get two random image
        image_1_json = http_get_random_image(self.dataset_name)
        image_2_json = http_get_random_image(self.dataset_name)

        state.imgs = []
        state.imgs.append(image_1_json)
        state.imgs.append(image_2_json)

    def select_image(self, option: str, state: State) -> State:
        if state.user_name == "":
            state.set_error(StateError(StateErrorType.USER_NAME, "Please enter your name"))
            return state.copy_with_error()

        if state.dataset is None:
            state.set_error(StateError(StateErrorType.DATASET, "Please select a dataset"))
            return state.copy_with_error()

        select_idx = ord(option) % 65

        img_1_file_hash = ""
        if "output_file_hash" in state.imgs[0]["task_output_file_dict"]:
            img_1_file_hash = state.imgs[0]["task_output_file_dict"]["output_file_hash"]

        img_2_file_hash = ""
        if "output_file_hash" in state.imgs[1]["task_output_file_dict"]:
            img_2_file_hash = state.imgs[1]["task_output_file_dict"]["output_file_hash"]

        selected_file_hash = img_1_file_hash
        if select_idx == 1:
            selected_file_hash = img_2_file_hash

        img_1_features = []
        if "features-vector" in state.imgs[0]:
            img_1_features = state.imgs[0]["features-vector"]

        img_2_features = []
        if "features-vector" in state.imgs[1]:
            img_2_features = state.imgs[1]["features-vector"]

        selection_data = {
            "task": "selection",
            "username": state.user_name,
            "image_1_metadata": {
                                    "file_name": state.imgs[0]["task_output_file_dict"]["output_file_path"],
                                    "file_hash": img_1_file_hash,
                                    "file_path": state.imgs[0]["task_output_file_dict"]["output_file_path"],
                                    "image_type": "jpeg",
                                    "image_width": "",
                                    "image_height": "",
                                    "image_size": "",
                                    "features_type": "",
                                    "features_model": "",
                                    "features_vector": img_1_features,
                                },
            "image_2_metadata": {
                                    "file_name": state.imgs[1]["task_output_file_dict"]["output_file_path"],
                                    "file_hash": img_2_file_hash,
                                    "file_path": state.imgs[1]["task_output_file_dict"]["output_file_path"],
                                    "image_type": "jpeg",
                                    "image_width": "",
                                    "image_height": "",
                                    "image_size": "",
                                    "features_type": "",
                                    "features_model": "",
                                    "features_vector": img_2_features,
                                },
            "selected_image_index": select_idx,
            "selected_image_hash": selected_file_hash,
        }

        # data = json.dumps(selection_data)
        http_add_selection(state.dataset, selection_data)

        # update images
        self.rand_select(state)

        return state.copy()

    def get_datasets(self):
        self.datasets = http_get_datasets()

        return self.datasets

    def set_dataset(self, dataset: str, state: State):
        self.dataset_name = dataset
        state.set_dataset(dataset)

    def get_image(self, option: str, state: State) -> bytes:
        image_data_json = state.imgs[ord(option) % 65]
        if "image-data" not in image_data_json:
            return None

        image_data = image_data_json["image-data"]

        img_bytes = base64.b64decode(image_data)
        return img_bytes
