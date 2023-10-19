from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pymongo
from dotenv import dotenv_values
from orchestration.api.api_dataset import router as dataset_router
from orchestration.api.api_image import router as image_router
from orchestration.api.api_job import router as job_router
from orchestration.api.api_ranking import router as ranking_router
from orchestration.api.api_training import router as training_router
from orchestration.api.api_model import router as model_router
from orchestration.api.api_tag import router as tag_router
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

app.include_router(dataset_router)
app.include_router(image_router)
app.include_router(job_router)
app.include_router(ranking_router)
app.include_router(training_router)
app.include_router(model_router)
app.include_router(tag_router)


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
    # add creation of mongodb here for now
    app.mongodb_client = pymongo.MongoClient(config["DB_URL"])
    app.mongodb_db = app.mongodb_client["orchestration-job-db"]
    app.pending_jobs_collection = app.mongodb_db["pending-jobs"]
    app.in_progress_jobs_collection = app.mongodb_db["in-progress-jobs"]
    app.completed_jobs_collection = app.mongodb_db["completed-jobs"]
    app.failed_jobs_collection = app.mongodb_db["failed-jobs"]

    # used to store sequential ids of generated images
    app.dataset_sequential_id_collection = app.mongodb_db["dataset-sequential-id"]

    # for training jobs
    app.training_pending_jobs_collection = app.mongodb_db["training-pending-jobs"]
    app.training_in_progress_jobs_collection = app.mongodb_db["training-in-progress-jobs"]
    app.training_completed_jobs_collection = app.mongodb_db["training-completed-jobs"]
    app.training_failed_jobs_collection = app.mongodb_db["training-failed-jobs"]

    # dataset rate
    app.dataset_config_collection = app.mongodb_db["dataset_config"]

    # tags
    app.tag_definitions_collection = app.mongodb_db["tag_definitions"]
    app.image_tags_collection = app.mongodb_db["image_tags"]

    print("Connected to the MongoDB database!")

    # get minio client
    app.minio_client = get_minio_client(minio_access_key=config["MINIO_ACCESS_KEY"],
                                        minio_secret_key=config["MINIO_SECRET_KEY"])


@app.on_event("shutdown")
def shutdown_db_client():
    app.mongodb_client.close()



