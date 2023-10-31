from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pymongo
from dotenv import dotenv_values
from clip_server.api.api_clip import router as clip_router
from clip_server.clip_server import ClipServer
from utility.minio import cmd

config = dotenv_values("./orchestration/api/.env")
app = FastAPI(title="Orchestration API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(clip_router)


def get_minio_client(minio_access_key, minio_secret_key):
    # check first if minio client is available
    minio_client = None
    while minio_client is None:
        # check minio server
        if cmd.is_minio_server_accesssible():
            minio_client = cmd.connect_to_minio_client(access_key=minio_access_key, secret_key=minio_secret_key)
            return minio_client


@app.on_event("startup")
def startup_db_client():
    # get minio client
    app.minio_client = get_minio_client(minio_access_key=config["MINIO_ACCESS_KEY"],
                                        minio_secret_key=config["MINIO_SECRET_KEY"])
    app.clip_server = ClipServer()




