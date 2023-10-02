import random
from enum import Enum
from pathlib import PurePath, PurePosixPath
from typing import List
from zipfile import ZipFile
import sys

base_directory = "./"
sys.path.insert(0, base_directory)

from app_ranking.http.request import http_get_random_image


class Rank:
    def __init__(self, name: str, id: str, points: int, **kwargs):
        self.name = name
        self.id = id
        self.points = points

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


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
        self._rank = None
        self._error = None

    @classmethod
    def from_dict(cls, data: dict):
        state = cls()
        state._imgs = data["_imgs"]
        state._user_name = data["_user_name"]
        state._rank = data["_rank"]
        return state

    @property
    def user_name(self):
        return self._user_name

    def set_user_name(self, name: str):
        if name:
            self.clear_errors()
        self._user_name = name

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, rank: str):
        if rank:
            self.clear_errors()
        self._rank = rank

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
    NUM_IMAGES = 2

    def __init__(self):
        self.datasets = []
        self.dataset_name = ""

    def rand_select(self, state: State):
        if state.has_error:
            return

        # get two random image
        image_1_json = http_get_random_image(self.dataset_name)
        image_2_json = http_get_random_image(self.dataset_name)
        print(image_1_json)
        state.imgs = []
        state.imgs.append(image_1_json["image-data"])
        state.imgs.append(image_2_json["image-data"])

    def select_image(self, option: str, state: State) -> State:
        # if state.user_name == "":
        #     state.set_error(StateError(StateErrorType.USER_NAME, "Please enter your name"))
        #     return state.copy_with_error()
        #
        # if state.rank is None:
        #     state.set_error(StateError(StateErrorType.RANK, "Please select a rank"))
        #     return state.copy_with_error()
        #
        # select_idx = ord(option) % 65
        # # self.db_manager.insert_selection(state.imgs, select_idx, state.user_name, state.rank.id)
        # #
        # # state.rank.points += 1
        # # self.db_manager.ranks_db.update({"points": state.rank.points}, Query().id == state.rank.id)
        #
        # self.rand_select(state)
        return state.copy()

    def get_datasets(self):
        self.ranks = [Rank(**rank) for rank in self.db_manager.ranks_db.all()]
        return [rank.name for rank in self.ranks]

    def set_dataset(self, dataset: str):
        self.dataset = dataset
