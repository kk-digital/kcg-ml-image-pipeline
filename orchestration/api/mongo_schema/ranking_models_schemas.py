from typing import List
from pydantic import BaseModel

class RankingModel(BaseModel):
    ranking_model_id: int
    model_name: str
    model_type: str
    rank_id: int
    latest_model_creation_time: str
    model_path: str
    creation_time: str

    def to_dict(self):
        return{
            "ranking_model_id": self.ranking_model_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "rank_id": self.rank_id,
            "latest_model_creation_time": self.latest_model_creation_time,
            "model_path": self.model_path,
            "creation_time": self.creation_time
        }

class RequestRanking_model(BaseModel):
    model_name: str
    model_type: str
    rank_id: int
    latest_model_creation_time: str
    model_path: str

    def to_dict(self):
        return{
            "model_name": self.model_name,
            "model_type": self.model_type,
            "rank_id": self.rank_id,
            "latest_model_creation_time": self.latest_model_creation_time,
            "model_path": self.model_path,
        }
    
class ListRankingModels(BaseModel):
    classifiers : List[RankingModel]