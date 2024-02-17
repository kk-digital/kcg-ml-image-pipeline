from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pymongo
from bson.objectid import ObjectId
from fastapi.responses import JSONResponse
from .api_utils import PrettyJSONResponse, ApiResponseHandler, ErrorCode,  StandardErrorResponseV1, StandardSuccessResponse
from fastapi.exceptions import RequestValidationError
from fastapi import status, Request
from dotenv import dotenv_values
from datetime import datetime
from orchestration.api.api_clip import router as clip_router
from orchestration.api.api_dataset import router as dataset_router
from orchestration.api.api_image import router as image_router
from orchestration.api.api_job_stats import router as job_stats_router
from orchestration.api.api_job import router as job_router
from orchestration.api.api_ranking import router as ranking_router
from orchestration.api.api_training import router as training_router
from orchestration.api.api_model import router as model_router
from orchestration.api.api_tag import router as tag_router
from orchestration.api.api_dataset_settings import router as dataset_settings_router
from orchestration.api.api_users import router as user_router
from orchestration.api.api_score import router as score_router
from orchestration.api.api_sigma_score import router as sigma_score_router
from orchestration.api.api_residual import router as residual_router
from orchestration.api.api_percentile import router as percentile_router
from orchestration.api.api_residual_percentile import router as residual_percentile_router
from orchestration.api.api_image_by_rank import router as image_by_rank_router
from orchestration.api.api_queue_ranking import router as queue_ranking
from orchestration.api.api_active_learning import router as active_learning 
from orchestration.api.api_active_learning_policy import router as active_learning_policy
from orchestration.api.api_pseudo_tag import router as pseudo_tags
from orchestration.api.api_worker import router as worker
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
app.include_router(dataset_router)
app.include_router(image_router)
app.include_router(image_by_rank_router)
app.include_router(job_router)
app.include_router(job_stats_router)
app.include_router(ranking_router)
app.include_router(training_router)
app.include_router(model_router)
app.include_router(tag_router)
app.include_router(dataset_settings_router)
app.include_router(user_router)
app.include_router(score_router)
app.include_router(sigma_score_router)
app.include_router(residual_router)
app.include_router(percentile_router)
app.include_router(residual_percentile_router)
app.include_router(queue_ranking)
app.include_router(active_learning)
app.include_router(active_learning_policy)
app.include_router(pseudo_tags)
app.include_router(worker)


def get_minio_client(minio_access_key, minio_secret_key):
    # check first if minio client is available
    minio_client = None
    while minio_client is None:
        # check minio server
        if cmd.is_minio_server_accessible():
            minio_client = cmd.connect_to_minio_client(access_key=minio_access_key, secret_key=minio_secret_key)
            return minio_client


def add_models_counter():
    # add counter for models
    try:
        app.counters_collection.insert_one({"_id": "models", "seq": 0})
    except Exception as e:
        print("models counter already exists.")

    return True


def create_index_if_not_exists(collection, index_key, index_name):
    existing_indexes = collection.index_information()
    
    if index_name not in existing_indexes:
        collection.create_index(index_key, name=index_name)
        print(f"Index '{index_name}' created on collection '{collection.name}'.")
    else:
        print(f"Index '{index_name}' already exists on collection '{collection.name}'.")


# Define the exception handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    start_time = datetime.now()  # Assuming you capture this at the start of request processing
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=StandardErrorResponseV1(
            request_error_string="Validation error",
            request_error_code=ErrorCode.INVALID_PARAMS.value,  
            request_url=str(request.url),
            request_dictionary=dict(request.query_params), 
            request_method=request.method,
            request_time_total=duration,
            request_time_start=start_time,
            request_time_finished=end_time,
            request_response_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        ).dict(),
    )


@app.on_event("startup")
def startup_db_client():
    # add creation of mongodb here for now
    app.mongodb_client = pymongo.MongoClient(config["DB_URL"])
    app.mongodb_db = app.mongodb_client["orchestration-job-db"]
    app.users_collection = app.mongodb_db["users"]
    app.pending_jobs_collection = app.mongodb_db["pending-jobs"]
    app.in_progress_jobs_collection = app.mongodb_db["in-progress-jobs"]
    app.completed_jobs_collection = app.mongodb_db["completed-jobs"]

    completed_jobs_hash_index=[
    ('task_output_file_dict.output_file_hash', pymongo.ASCENDING)
    ]
    create_index_if_not_exists(app.completed_jobs_collection ,completed_jobs_hash_index, 'completed_jobs_hash_index')

    completed_jobs_createdAt_index=[
    ('task_creation_time', pymongo.ASCENDING)
    ]
    create_index_if_not_exists(app.completed_jobs_collection ,completed_jobs_createdAt_index, 'completed_jobs_createdAt_index')
    
    completed_jobs_compound_index=[
    ('task_input_dict.dataset', pymongo.ASCENDING),
    ('task_creation_time', pymongo.ASCENDING)
    ]
    create_index_if_not_exists(app.completed_jobs_collection ,completed_jobs_compound_index, 'completed_jobs_compound_index')

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
    app.tag_categories_collection = app.mongodb_db["tag_categories"]

    # pseudo tags
    app.pseudo_tag_definitions_collection = app.mongodb_db["pseudo_tag_definitions"]
    app.pseudo_image_tags_collection = app.mongodb_db["pseudo_image_tags"]
    app.pseudo_tag_categories_collection = app.mongodb_db["pseudo_tag_categories"]

    # delta score
    app.datapoints_delta_score_collection = app.mongodb_db["datapoints_delta_score"]

    # models
    app.models_collection = app.mongodb_db["models"]

    # counters
    app.counters_collection = app.mongodb_db["counters"]
    add_models_counter()

    app.uuid_tag_count_collection = app.mongodb_db["tag_count"]

    
    # scores
    app.image_scores_collection = app.mongodb_db["image-scores"]

    # active learning
    app.active_learning_policies_collection = app.mongodb_db["active-learning-policies"]
    app.active_learning_queue_pairs_collection = app.mongodb_db["queue-pairs"]

    scores_index=[
    ('model_id', pymongo.ASCENDING), 
    ('score', pymongo.ASCENDING)
    ]
    create_index_if_not_exists(app.image_scores_collection ,scores_index, 'scores_index')

    hash_index=[
    ('model_id', pymongo.ASCENDING), 
    ('image_hash', pymongo.ASCENDING)
    ]
    create_index_if_not_exists(app.image_scores_collection ,hash_index, 'score_hash_index')

    # sigma scores
    app.image_sigma_scores_collection = app.mongodb_db["image-sigma-scores"]

    sigma_scores_index = [
        ('model_id', pymongo.ASCENDING),
        ('sigma-score', pymongo.ASCENDING)
    ]
    create_index_if_not_exists(app.image_sigma_scores_collection, sigma_scores_index, 'sigma_scores_index')

    hash_index = [
        ('model_id', pymongo.ASCENDING),
        ('image_hash', pymongo.ASCENDING)
    ]
    create_index_if_not_exists(app.image_sigma_scores_collection, hash_index, 'sigma_score_hash_index')

    # residuals
    app.image_residuals_collection = app.mongodb_db["image-residuals"]

    residuals_index=[
    ('model_id', pymongo.ASCENDING), 
    ('residual', pymongo.ASCENDING)
    ]
    create_index_if_not_exists(app.image_residuals_collection ,residuals_index, 'residuals_index')
    create_index_if_not_exists(app.image_residuals_collection ,hash_index, 'residual_hash_index')

    # percentiles
    app.image_percentiles_collection = app.mongodb_db["image-percentiles"]

    percentiles_index=[
    ('model_id', pymongo.ASCENDING), 
    ('percentile', pymongo.ASCENDING)
    ]
    create_index_if_not_exists(app.image_percentiles_collection ,percentiles_index, 'percentiles_index')
    create_index_if_not_exists(app.image_percentiles_collection ,hash_index, 'percentile_hash_index')

    # residual percentiles
    app.image_residual_percentiles_collection = app.mongodb_db["image-residual-percentiles"]

    # image rank use count - the count the image is used in selection datapoint
    app.image_rank_use_count_collection = app.mongodb_db["image-rank-use-count"]

    app.image_pair_ranking_collection = app.mongodb_db["image_pair_ranking"]

    print("Connected to the MongoDB database!")

    # get minio client
    app.minio_client = get_minio_client(minio_access_key=config["MINIO_ACCESS_KEY"],
                                        minio_secret_key=config["MINIO_SECRET_KEY"])


@app.on_event("shutdown")
def shutdown_db_client():
    app.mongodb_client.close()



