from fastapi import Request, APIRouter, HTTPException
from orchestration.api.mongo_schemas import RankingResidual

router = APIRouter()

def http_clip_server_add_phrase(dataset_name: str):
    url = CLIP_SERVER_ADRESS + "/add-phrase/count-in-progress?dataset=" + dataset_name

    try:
        response = requests.get(url)

        if response.status_code == 200:
            job_json = response.json()
            return job_json

    except Exception as e:
        print('request exception ', e)

    return None

@router.post("/clip/add-phrase", description="Adds a phrase to the clip server")
def add_phrase(request: Request,
               phrase : str):



    return True