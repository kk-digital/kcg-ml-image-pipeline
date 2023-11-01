from fastapi import Request, APIRouter, HTTPException
import requests

CLIP_SERVER_ADRESS = 'http://127.0.0.1:8002'

router = APIRouter()

def http_clip_server_add_phrase(phrase: str):
    url = CLIP_SERVER_ADRESS + "/add-phrase/phrase=" + phrase

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

    return http_clip_server_add_phrase(phrase)



    return True