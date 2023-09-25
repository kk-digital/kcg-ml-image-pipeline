from fastapi import Request, HTTPException, APIRouter


router = APIRouter()


@router.get("/get-job")
def get_job(request: Request):
    # find
    job = request.app.pending_jobs_collection.find_one()
    if job is None:
        raise job

    # delete from pending
    request.app.pending_jobs_collection.delete_one({"uuid": job["uuid"]})
    # add to in progress
    request.app.in_progress_jobs_collection.insert_one(job)

    # remove the auto generated field
    job.pop('_id', None)

    return job
