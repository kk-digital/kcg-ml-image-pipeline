from fastapi import Request, APIRouter


router = APIRouter()


@router.get("/get-list-pending-jobs")
def get_list_pending_jobs(request: Request):
    jobs = list(request.app.pending_jobs_collection.find({}))

    for job in jobs:
        job.pop('_id', None)

    return jobs


@router.get("/get-list-in-progress-jobs")
def get_list_in_progress_jobs(request: Request):
    jobs = list(request.app.in_progress_jobs_collection.find({}))

    for job in jobs:
        job.pop('_id', None)

    return jobs


@router.get("/get-list-completed-jobs")
def get_list_completed_jobs(request: Request):
    jobs = list(request.app.completed_jobs_collection.find({}))

    for job in jobs:
        job.pop('_id', None)

    return jobs


@router.get("/get-list-failed-jobs")
def get_list_failed_jobs(request: Request):
    jobs = list(request.app.failed_jobs_collection.find({}))

    for job in jobs:
        job.pop('_id', None)

    return jobs
