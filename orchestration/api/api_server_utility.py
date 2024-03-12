from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/ping-server", response_class=JSONResponse)
async def ping_server():
    headers = {
        "content-type": "application/json; charset=utf-8",
    }
    return JSONResponse(content="Pong!", headers=headers, status_code=200)