from fastapi import Request, APIRouter, Query
from .api_utils import PrettyJSONResponse

router = APIRouter()

@router.get("/image_by_rank/image-list-sorted-by-score", response_class=PrettyJSONResponse)
def image_list_sorted_by_score(
    request: Request,
    dataset: str = Query(...),
    limit: int = 20,
    offset: int = 0,
    start_date: str = None,
    end_date: str = None,
    sort_order: str = 'asc',
    model_id: int=Query(...) ,
    min_score: float = None,
    max_score: float = None
):
    
    #Decide the sort order based on the 'order' parameter
    sort_order = -1 if sort_order == "desc" else 1
    
    #query to get all scores of the specified model and sort them
    scores_query = {"model_id": model_id}
    if min_score is not None:
        scores_query['score'] = {'$gte': min_score}
    if max_score is not None:
        scores_query['score'] = {'$lte': max_score}
    scores_data = list(request.app.image_scores_collection.find(scores_query, 
    {'_id': 0, 'image_hash': 1, 'score': 1}).sort("score", sort_order))

    images_data=[]

    # query to filter images based on dataset and date
    imgs_query={"task_input_dict.dataset": dataset}

    if start_date and end_date:
        imgs_query['task_creation_time'] = {'$gte': start_date, '$lte': end_date}
    elif start_date:
        imgs_query['task_creation_time'] = {'$gte': start_date}
    elif end_date:
        imgs_query['task_creation_time'] = {'$lte': end_date}

    #loop to get filtered list of images and their scores
    for data in scores_data:
        # adding filter based on image hash
        imgs_query['task_output_file_dict.output_file_hash']=data['image_hash']
        img=request.app.completed_jobs_collection.find_one(imgs_query)

        # only appending image to response if it is within date range
        if img is not None:
            images_data.append({
            'image_path': img['task_output_file_dict']['output_file_path'],
            'image_hash': data['image_hash'],
            'score': data['score']
            })
    
    #applying offset and limit for pagination
    images_data=images_data[offset:offset+limit]

    return images_data

@router.get("/image_by_rank/image-list-sorted-by-percentile", response_class=PrettyJSONResponse)
def image_list_sorted_by_percentile(
    request: Request,
    dataset: str = Query(...),
    limit: int = 20,
    offset: int = 0,
    start_date: str = None,
    end_date: str = None,
    sort_order: str = 'asc',
    model_id: int=Query(...) ,
    min_percentile: float = None,
    max_percentile: float = None
):
    
    #Decide the sort order based on the 'order' parameter
    sort_order = -1 if sort_order == "desc" else 1
    
    #query to get all percentiles of the specified model and sort them
    percentiles_query = {"model_id": model_id}
    if min_percentile is not None:
        percentiles_query['percentile'] = {'$gte': min_percentile}
    if max_percentile is not None:
        percentiles_query['percentile'] = {'$lte': max_percentile}
    percentiles_data = list(request.app.image_percentiles_collection.find(percentiles_query, 
    {'_id': 0, 'image_hash': 1, 'percentile': 1}).sort("percentile", sort_order))

    images_data=[]

    # query to filter images based on dataset and date
    imgs_query={"task_input_dict.dataset": dataset}

    if start_date and end_date:
        imgs_query['task_creation_time'] = {'$gte': start_date, '$lte': end_date}
    elif start_date:
        imgs_query['task_creation_time'] = {'$gte': start_date}
    elif end_date:
        imgs_query['task_creation_time'] = {'$lte': end_date}

    #loop to get filtered list of images and their percentiles
    for data in percentiles_data:
        # adding filter based on image hash
        imgs_query['task_output_file_dict.output_file_hash']=data['image_hash']
        img=request.app.completed_jobs_collection.find_one(imgs_query)

        # only appending image to response if it is within date range
        if img is not None:
            images_data.append({
            'image_path': img['task_output_file_dict']['output_file_path'],
            'image_hash': data['image_hash'],
            'percentile': data['percentile']
            })
    
    #applying offset and limit for pagination
    images_data=images_data[offset:offset+limit]

    return images_data

@router.get("/image_by_rank/image-list-sorted-by-residual", response_class=PrettyJSONResponse)
def image_list_sorted_by_residual(
    request: Request,
    dataset: str = Query(...),
    limit: int = 20,
    offset: int = 0,
    start_date: str = None,
    end_date: str = None,
    sort_order: str = 'asc',
    model_id: int=Query(...) ,
    min_residual: float = None,
    max_residual: float = None
):
    
    #Decide the sort order based on the 'order' parameter
    sort_order = -1 if sort_order == "desc" else 1
    
    #query to get all residuals of the specified model and sort them
    residuals_query = {"model_id": model_id}
    if min_residual is not None:
        residuals_query['residual'] = {'$gte': min_residual}
    if max_residual is not None:
        residuals_query['residual'] = {'$lte': max_residual}
    residuals_data = list(request.app.image_residuals_collection.find(residuals_query, 
    {'_id': 0, 'image_hash': 1, 'residual': 1}).sort("residual", sort_order))

    images_data=[]

    # query to filter images based on dataset and date
    imgs_query={"task_input_dict.dataset": dataset}

    if start_date and end_date:
        imgs_query['task_creation_time'] = {'$gte': start_date, '$lte': end_date}
    elif start_date:
        imgs_query['task_creation_time'] = {'$gte': start_date}
    elif end_date:
        imgs_query['task_creation_time'] = {'$lte': end_date}

    #loop to get filtered list of images and their residuals
    for data in residuals_data:
        # adding filter based on image hash
        imgs_query['task_output_file_dict.output_file_hash']=data['image_hash']
        img=request.app.completed_jobs_collection.find_one(imgs_query)

        # only appending image to response if it is within date range
        if img is not None:
            images_data.append({
            'image_path': img['task_output_file_dict']['output_file_path'],
            'image_hash': data['image_hash'],
            'residual': data['residual']
            })
    
    #applying offset and limit for pagination
    images_data=images_data[offset:offset+limit]

    return images_data
