from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pymongo
from dotenv import dotenv_values
from api.api_clip import router as clip_router
from server_state import ClipServer
from utility.minio import cmd
import multiprocessing
import uvicorn
import threading
import time

config = dotenv_values("./orchestration/api/.env")
app = FastAPI(title="Clip Server API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(clip_router)


def get_minio_client(minio_address, minio_access_key, minio_secret_key):
    # check first if minio client is available
    minio_client = None
    while minio_client is None:
        # check minio server
        if cmd.is_minio_server_accessible(minio_address):
            minio_client = cmd.connect_to_minio_client(minio_ip_addr=minio_address, access_key=minio_access_key, secret_key=minio_secret_key)
            return minio_client

# Gets list of completed jobs
# For each image that is not in dictionary
# We will download from minio
def check_new_images_and_download(clip_server):
    while True:
        # TODO(): orchestration must provide an api
        # TODO(): that will take in num_jobs & offset
        # TODO(): so that we dont download millions of jobs each time
        clip_server.download_all_clip_vectors()

        # Sleep for 2 hours
        sleep_time_in_seconds = 2.0 * 60 * 60
        time.sleep(sleep_time_in_seconds)


@app.on_event("startup")
def startup_db_client():
    app.device = 'cuda'

    # get minio client
    app.minio_client = get_minio_client(minio_address=config["MINIO_ADDRESS"],
                                        minio_access_key=config["MINIO_ACCESS_KEY"],
                                        minio_secret_key=config["MINIO_SECRET_KEY"])
    app.clip_server = ClipServer(app.device, app.minio_client)
    app.clip_server.load_clip_model()

    # downloads all clip vectors
    app.clip_server.download_all_clip_vectors()

    # spawn a thread that will check if there are
    # new images clip_vectors & download them
    thread = threading.Thread(target=check_new_images_and_download, args=(app.clip_server,))
    thread.start()

if __name__ == "__main__":

    # get number of cores
    cores = multiprocessing.cpu_count()

    # Run the API
    uvicorn.run("clip_server.main:app", host="0.0.0.0", port=8002, workers=cores, reload=True)

