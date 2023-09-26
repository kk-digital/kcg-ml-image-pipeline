from fastapi import FastAPI, Request, HTTPException
import pymongo
from dotenv import dotenv_values
from orchestration.api.add import router as add_router
from orchestration.api.count import router as count_router
from orchestration.api.delete import router as delete_router
from orchestration.api.get import router as get_router
from orchestration.api.list import router as list_router
from orchestration.api.update import router as update_router


config = dotenv_values("./orchestration/api/.env")
app = FastAPI(title="Orchestration API")
app.include_router(add_router)
app.include_router(get_router)
app.include_router(count_router)
app.include_router(list_router)
app.include_router(update_router)
app.include_router(delete_router)


@app.on_event("startup")
def startup_db_client():
    # add creation of mongodb here for now
    app.mongodb_client = pymongo.MongoClient(config["DB_URL"])
    app.mongodb_db = app.mongodb_client["orchestration-job-db"]
    app.pending_jobs_collection = app.mongodb_db["pending-jobs"]
    app.in_progress_jobs_collection = app.mongodb_db["in-progress-jobs"]
    app.completed_jobs_collection = app.mongodb_db["completed-jobs"]
    app.failed_jobs_collection = app.mongodb_db["failed-jobs"]
    print("Connected to the MongoDB database!")


@app.on_event("shutdown")
def shutdown_db_client():
    app.mongodb_client.close()



