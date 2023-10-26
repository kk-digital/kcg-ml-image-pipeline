from fastapi import Request, APIRouter, Query


router = APIRouter()

# get job stats by job type
@router.get("/job_stats/stats_by_job_type")
def get_job_stats_by_job_type(request: Request, job_type: str = Query(...)):
    pending_count = request.app.pending_jobs_collection.count_documents({
        'task_type': job_type
    })
    
    progress_count = request.app.in_progress_jobs_collection.count_documents({
        'task_type': job_type
    })

    completed_count = request.app.completed_jobs_collection.count_documents({
        'task_type': job_type
    })

    failed_count = request.app.failed_jobs_collection.count_documents({
        'task_type': job_type
    })
    return {
            'total': pending_count +  progress_count + completed_count + failed_count,
            'pending_count': pending_count,
            'progress_count': progress_count,
            'completed_count': completed_count,
            'failed_count': failed_count
    }

# get job stats by dataset
@router.get("/job_stats/stats_by_dataset")
def get_job_stats_by_job_type(request: Request, dataset: str = Query(...)):
    pending_count = request.app.pending_jobs_collection.count_documents({
        "task_input_dict.dataset": dataset
    })
    
    progress_count = request.app.in_progress_jobs_collection.count_documents({
        "task_input_dict.dataset": dataset
    })

    completed_count = request.app.completed_jobs_collection.count_documents({
        "task_input_dict.dataset": dataset
    })

    failed_count = request.app.failed_jobs_collection.count_documents({
        "task_input_dict.dataset": dataset
    })
    return {
            'total': pending_count +  progress_count + completed_count + failed_count,
            'pending_count': pending_count,
            'progress_count': progress_count,
            'completed_count': completed_count,
            'failed_count': failed_count
    }