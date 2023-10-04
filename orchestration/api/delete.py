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

@router.delete("/clear-all-completed-jobs")
def clear_all_completed_jobs(request: Request):
    request.app.completed_jobs_collection.delete_many({})

    return True


@router.delete("/clear-dataset-sequential-id")
def clear_dataset_sequential_id_jobs(request: Request):
    request.app.dataset_sequential_id_collection.delete_many({})

    return True