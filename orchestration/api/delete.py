from fastapi import Request, APIRouter


router = APIRouter()


@router.delete("/clear-all-pending-jobs")
def clear_all_pending_jobs(request: Request):
    request.app.pending_jobs_collection.delete_many({})

    return True


@router.delete("/clear-all-in-progress-jobs")
def clear_all_in_progress_jobs(request: Request):
    request.app.in_progress_jobs_collection.delete_many({})

    return True


@router.delete("/clear-all-failed-jobs")
def clear_all_failed_jobs(request: Request):
    request.app.failed_jobs_collection.delete_many({})

    return True
