from fastapi import Request, APIRouter, HTTPException
import requests

CLIP_SERVER_ADRESS = 'http://127.0.0.1:8002'

router = APIRouter()


# --------- Http requests -------------
def http_clip_server_add_phrase(phrase: str):
    url = CLIP_SERVER_ADRESS + "/add-phrase?phrase=" + phrase

    try:
        response = requests.put(url)

        if response.status_code == 200:
            result_json = response.json()
            return result_json

    except Exception as e:
        print('request exception ', e)

    return None




def http_clip_server_clip_vector_from_phrase(phrase: str):
    url = CLIP_SERVER_ADRESS + "/clip-vector?phrase=" + phrase

    try:
        response = requests.get(url)

        if response.status_code == 200:
            result_json = response.json()
            return result_json

    except Exception as e:
        print('request exception ', e)

    return None

@router.put("/clip/add-phrase", description="Adds a phrase to the clip server")
def add_phrase(request: Request,
               phrase : str):

    return http_clip_server_add_phrase(phrase)


@router.get("/clip/clip-vector", description="Gets a clip vector of a specific phrase")
def add_phrase(request: Request,
               phrase : str):

    return http_clip_server_clip_vector_from_phrase(phrase)