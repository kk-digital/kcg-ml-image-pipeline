from fastapi import Request, APIRouter


router = APIRouter()


@router.get("/get-pending-job-count")
def get_pending_job_count(request: Request):
    count = request.app.pending_jobs_collection.count_documents({})
    return count


@router.get("/get-in-progress-job-count")
def get_in_progress_job_count(request: Request):
    count = request.app.in_progress_jobs_collection.count_documents({})
    return count


@router.get("/get-completed-job-count")
def get_completed_job_count(request: Request):
    count = request.app.completed_jobs_collection.count_documents({})
    return count


@router.get("/get-failed-job-count")
def get_failed_job_count(request: Request):
    count = request.app.failed_jobs_collection.count_documents({})
    return count