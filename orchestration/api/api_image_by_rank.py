from fastapi import Request, APIRouter, Query, HTTPException
from .api_utils import PrettyJSONResponse
from datetime import datetime, timedelta


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
    model_id: int = Query(...),
    min_score: float = None,
    max_score: float = None,
    time_interval: int = Query(None, description="Time interval in minutes or hours"),
    time_unit: str = Query("minutes", description="Time unit, either 'minutes' or 'hours")
):

    # Calculate the time threshold based on the current time and the specified interval
    if time_interval is not None:
        current_time = datetime.utcnow()
        if time_unit == "minutes":
            threshold_time = current_time - timedelta(minutes=time_interval)
        elif time_unit == "hours":
            threshold_time = current_time - timedelta(hours=time_interval)
        else:
            raise HTTPException(status_code=400, detail="Invalid time unit. Use 'minutes' or 'hours'.")
    else:
        threshold_time = None

    # Decide the sort order based on the 'sort_order' parameter
    sort_order = -1 if sort_order == "desc" else 1

    # Query to get all scores of the specified model and sort them
    scores_query = {"model_id": model_id}
    if min_score and max_score:
        scores_query['score'] = {'$gte': min_score, '$lte': max_score}
    elif min_score:
        scores_query['score'] = {'$gte': min_score}
    elif max_score:
        scores_query['score'] = {'$lte': max_score}
    scores_data = list(request.app.image_scores_collection.find(scores_query, 
    {'_id': 0, 'image_hash': 1, 'score': 1}).sort("score", sort_order))

    images_data = []

    # Query to filter images based on dataset, date, and threshold_time
    imgs_query = {"task_input_dict.dataset": dataset}

    # Update the query based on provided start_date, end_date, and threshold_time
    if start_date and end_date:
        imgs_query['task_creation_time'] = {'$gte': start_date, '$lte': end_date}
    elif start_date:
        imgs_query['task_creation_time'] = {'$gte': start_date}
    elif end_date:
        imgs_query['task_creation_time'] = {'$lte': end_date}
    elif threshold_time:
        imgs_query['task_creation_time'] = {'$gte': threshold_time.strftime("%Y-%m-%dT%H:%M:%S")}

    # Loop to get filtered list of images and their scores
    for data in scores_data:
        # Adding filter based on image hash
        imgs_query['task_output_file_dict.output_file_hash'] = data['image_hash']
        img = request.app.completed_jobs_collection.find_one(imgs_query)

        # Only appending image to response if it is within date range
        if img is not None:
            images_data.append({
                'image_path': img['task_output_file_dict']['output_file_path'],
                'image_hash': data['image_hash'],
                'score': data['score']
            })
    
    # Applying offset and limit for pagination
    images_data = images_data[offset:offset+limit]

    return images_data


@router.get("/image_by_rank/image-sorted-by-score", response_class=PrettyJSONResponse)
def image_list_sorted_by_score(
    request: Request,
    dataset: str = Query(...),
    limit: int = 20,
    offset: int = 0,
    start_date: str = None,
    end_date: str = None,
    sort_order: str = 'asc',
    model_id: int = Query(...),
    min_score: float = None,
    max_score: float = None,
    time_interval: int = Query(None, description="Time interval in minutes or hours"),
    time_unit: str = Query("minutes", description="Time unit, either 'minutes' or 'hours")
):

    # Calculate the time threshold based on the current time and the specified interval
    if time_interval is not None:
        current_time = datetime.utcnow()
        if time_unit == "minutes":
            threshold_time = current_time - timedelta(minutes=time_interval)
        elif time_unit == "hours":
            threshold_time = current_time - timedelta(hours=time_interval)
        else:
            raise HTTPException(status_code=400, detail="Invalid time unit. Use 'minutes' or 'hours'.")
    else:
        threshold_time = None

    # Decide the sort order based on the 'sort_order' parameter
    sort_order = -1 if sort_order == "desc" else 1

    # Query to get all scores of the specified model and sort them
    scores_query = {"model_id": model_id}
    if min_score and max_score:
        scores_query['score'] = {'$gte': min_score, '$lte': max_score}
    elif min_score:
        scores_query['score'] = {'$gte': min_score}
    elif max_score:
        scores_query['score'] = {'$lte': max_score}
    scores_data = list(request.app.image_scores_collection.find(scores_query, 
    {'_id': 0, 'image_hash': 1, 'score': 1}).sort("score", sort_order))

    images_data = []

    # Query to filter images based on dataset, date, and threshold_time
    imgs_query = {"task_input_dict.dataset": dataset}

    # Update the query based on provided start_date, end_date, and threshold_time
    if start_date and end_date:
        imgs_query['task_creation_time'] = {'$gte': start_date, '$lte': end_date}
    elif start_date:
        imgs_query['task_creation_time'] = {'$gte': start_date}
    elif end_date:
        imgs_query['task_creation_time'] = {'$lte': end_date}
    elif threshold_time:
        imgs_query['task_creation_time'] = {'$gte': threshold_time.strftime("%Y-%m-%dT%H:%M:%S")}

    # Loop to get filtered list of images and their scores
    for data in scores_data:
        # Adding filter based on image hash
        imgs_query['task_output_file_dict.output_file_hash'] = data['image_hash']
        img = request.app.completed_jobs_collection.find_one(imgs_query)

        # Only appending image to response if it is within date range
        if img is not None:
            images_data.append({
                'image_path': img['task_output_file_dict']['output_file_path'],
                'image_hash': data['image_hash'],
                'score': data['score']
            })
    
    # Applying offset and limit for pagination
    images_data = images_data[offset:offset+limit]

    return {"images_data": images_data}

@router.get("/image_by_rank/image-list-sorted-by-percentile", response_class=PrettyJSONResponse)
def image_list_sorted_by_percentile(
    request: Request,
    dataset: str = Query(...),
    limit: int = 20,
    offset: int = 0,
    start_date: str = None,
    end_date: str = None,
    sort_order: str = 'asc',
    model_id: int = Query(...),
    min_percentile: float = None,
    max_percentile: float = None,
    time_interval: int = Query(None, description="Time interval in minutes or hours"),
    time_unit: str = Query("minutes", description="Time unit, either 'minutes' or 'hours")
):

    # Calculate the time threshold based on the current time and the specified interval
    if time_interval is not None:
        current_time = datetime.utcnow()
        if time_unit == "minutes":
            threshold_time = current_time - timedelta(minutes=time_interval)
        elif time_unit == "hours":
            threshold_time = current_time - timedelta(hours=time_interval)
        else:
            raise HTTPException(status_code=400, detail="Invalid time unit. Use 'minutes' or 'hours'.")
    else:
        threshold_time = None

    # Decide the sort order based on the 'sort_order' parameter
    sort_order = -1 if sort_order == "desc" else 1

    # Query to get all percentiles of the specified model and sort them
    percentiles_query = {"model_id": model_id}
    if min_percentile and max_percentile:
        percentiles_query['percentile'] = {'$gte': min_percentile, '$lte': max_percentile}
    elif min_percentile:
        percentiles_query['percentile'] = {'$gte': min_percentile}
    elif max_percentile:
        percentiles_query['percentile'] = {'$lte': max_percentile}

    percentiles_data = list(request.app.image_percentiles_collection.find(percentiles_query, 
    {'_id': 0, 'image_hash': 1, 'percentile': 1}).sort("percentile", sort_order))

    images_data = []

    # Query to filter images based on dataset, date, and threshold_time
    imgs_query = {"task_input_dict.dataset": dataset}

    # Update the query based on provided start_date, end_date, and threshold_time
    if start_date and end_date:
        imgs_query['task_creation_time'] = {'$gte': start_date, '$lte': end_date}
    elif start_date:
        imgs_query['task_creation_time'] = {'$gte': start_date}
    elif end_date:
        imgs_query['task_creation_time'] = {'$lte': end_date}
    elif threshold_time:
        imgs_query['task_creation_time'] = {'$gte': threshold_time}

    # Loop to get filtered list of images and their percentiles
    for data in percentiles_data:
        # Adding filter based on image hash
        imgs_query['task_output_file_dict.output_file_hash'] = data['image_hash']
        img = request.app.completed_jobs_collection.find_one(imgs_query)

        # Only appending image to response if it is within date range
        if img is not None:
            images_data.append({
                'image_path': img['task_output_file_dict']['output_file_path'],
                'image_hash': data['image_hash'],
                'percentile': data['percentile']
            })
    
    # Applying offset and limit for pagination
    images_data = images_data[offset:offset+limit]

    return images_data

@router.get("/image_by_rank/image-sorted-by-percentile", response_class=PrettyJSONResponse)
def image_list_sorted_by_percentile(
    request: Request,
    dataset: str = Query(...),
    limit: int = 20,
    offset: int = 0,
    start_date: str = None,
    end_date: str = None,
    sort_order: str = 'asc',
    model_id: int = Query(...),
    min_percentile: float = None,
    max_percentile: float = None,
    time_interval: int = Query(None, description="Time interval in minutes or hours"),
    time_unit: str = Query("minutes", description="Time unit, either 'minutes' or 'hours")
):

    # Calculate the time threshold based on the current time and the specified interval
    if time_interval is not None:
        current_time = datetime.utcnow()
        if time_unit == "minutes":
            threshold_time = current_time - timedelta(minutes=time_interval)
        elif time_unit == "hours":
            threshold_time = current_time - timedelta(hours=time_interval)
        else:
            raise HTTPException(status_code=400, detail="Invalid time unit. Use 'minutes' or 'hours'.")
    else:
        threshold_time = None

    # Decide the sort order based on the 'sort_order' parameter
    sort_order = -1 if sort_order == "desc" else 1

    # Query to get all percentiles of the specified model and sort them
    percentiles_query = {"model_id": model_id}
    if min_percentile and max_percentile:
        percentiles_query['percentile'] = {'$gte': min_percentile, '$lte': max_percentile}
    elif min_percentile:
        percentiles_query['percentile'] = {'$gte': min_percentile}
    elif max_percentile:
        percentiles_query['percentile'] = {'$lte': max_percentile}

    percentiles_data = list(request.app.image_percentiles_collection.find(percentiles_query, 
    {'_id': 0, 'image_hash': 1, 'percentile': 1}).sort("percentile", sort_order))

    images_data = []

    # Query to filter images based on dataset, date, and threshold_time
    imgs_query = {"task_input_dict.dataset": dataset}

    # Update the query based on provided start_date, end_date, and threshold_time
    if start_date and end_date:
        imgs_query['task_creation_time'] = {'$gte': start_date, '$lte': end_date}
    elif start_date:
        imgs_query['task_creation_time'] = {'$gte': start_date}
    elif end_date:
        imgs_query['task_creation_time'] = {'$lte': end_date}
    elif threshold_time:
        imgs_query['task_creation_time'] = {'$gte': threshold_time}

    # Loop to get filtered list of images and their percentiles
    for data in percentiles_data:
        # Adding filter based on image hash
        imgs_query['task_output_file_dict.output_file_hash'] = data['image_hash']
        img = request.app.completed_jobs_collection.find_one(imgs_query)

        # Only appending image to response if it is within date range
        if img is not None:
            images_data.append({
                'image_path': img['task_output_file_dict']['output_file_path'],
                'image_hash': data['image_hash'],
                'percentile': data['percentile']
            })
    
    # Applying offset and limit for pagination
    images_data = images_data[offset:offset+limit]

    return {"image_data": images_data}

@router.get("/image_by_rank/image-list-sorted-by-residual", response_class=PrettyJSONResponse)
def image_list_sorted_by_residual(
    request: Request,
    dataset: str = Query(...),
    limit: int = 20,
    offset: int = 0,
    start_date: str = None,
    end_date: str = None,
    sort_order: str = 'asc',
    model_id: int = Query(...),
    min_residual: float = None,
    max_residual: float = None,
    time_interval: int = Query(None, description="Time interval in minutes or hours"),
    time_unit: str = Query("minutes", description="Time unit, either 'minutes' or 'hours")
):

    # Calculate the time threshold based on the current time and the specified interval
    if time_interval is not None:
        current_time = datetime.utcnow()
        if time_unit == "minutes":
            threshold_time = current_time - timedelta(minutes=time_interval)
        elif time_unit == "hours":
            threshold_time = current_time - timedelta(hours=time_interval)
        else:
            raise HTTPException(status_code=400, detail="Invalid time unit. Use 'minutes' or 'hours'.")
    else:
        threshold_time = None

    # Decide the sort order based on the 'sort_order' parameter
    sort_order = -1 if sort_order == "desc" else 1

    # Query to get all residuals of the specified model and sort them
    residuals_query = {"model_id": model_id}
    if min_residual and max_residual:
        residuals_query['residual'] = {'$gte': min_residual, '$lte': max_residual}
    elif min_residual:
        residuals_query['residual'] = {'$gte': min_residual}
    elif max_residual:
        residuals_query['residual'] = {'$lte': max_residual}
    residuals_data = list(request.app.image_residuals_collection.find(residuals_query, 
    {'_id': 0, 'image_hash': 1, 'residual': 1}).sort("residual", sort_order))

    images_data = []

    # Query to filter images based on dataset, date, and threshold_time
    imgs_query = {"task_input_dict.dataset": dataset}

    # Update the query based on provided start_date, end_date, and threshold_time
    if start_date and end_date:
        imgs_query['task_creation_time'] = {'$gte': start_date, '$lte': end_date}
    elif start_date:
        imgs_query['task_creation_time'] = {'$gte': start_date}
    elif end_date:
        imgs_query['task_creation_time'] = {'$lte': end_date}
    elif threshold_time:
        imgs_query['task_creation_time'] = {'$gte': threshold_time}

    # Loop to get filtered list of images and their residuals
    for data in residuals_data:
        # Adding filter based on image hash
        imgs_query['task_output_file_dict.output_file_hash'] = data['image_hash']
        img = request.app.completed_jobs_collection.find_one(imgs_query)

        # Only appending image to response if it is within date range
        if img is not None:
            images_data.append({
                'image_path': img['task_output_file_dict']['output_file_path'],
                'image_hash': data['image_hash'],
                'residual': data['residual']
            })
    
    # Applying offset and limit for pagination
    images_data = images_data[offset:offset+limit]

    return images_data


@router.get("/image_by_rank/image-sorted-by-residual", response_class=PrettyJSONResponse)
def image_list_sorted_by_residual(
    request: Request,
    dataset: str = Query(...),
    limit: int = 20,
    offset: int = 0,
    start_date: str = None,
    end_date: str = None,
    sort_order: str = 'asc',
    model_id: int = Query(...),
    min_residual: float = None,
    max_residual: float = None,
    time_interval: int = Query(None, description="Time interval in minutes or hours"),
    time_unit: str = Query("minutes", description="Time unit, either 'minutes' or 'hours")
):

    # Calculate the time threshold based on the current time and the specified interval
    if time_interval is not None:
        current_time = datetime.utcnow()
        if time_unit == "minutes":
            threshold_time = current_time - timedelta(minutes=time_interval)
        elif time_unit == "hours":
            threshold_time = current_time - timedelta(hours=time_interval)
        else:
            raise HTTPException(status_code=400, detail="Invalid time unit. Use 'minutes' or 'hours'.")
    else:
        threshold_time = None

    # Decide the sort order based on the 'sort_order' parameter
    sort_order = -1 if sort_order == "desc" else 1

    # Query to get all residuals of the specified model and sort them
    residuals_query = {"model_id": model_id}
    if min_residual and max_residual:
        residuals_query['residual'] = {'$gte': min_residual, '$lte': max_residual}
    elif min_residual:
        residuals_query['residual'] = {'$gte': min_residual}
    elif max_residual:
        residuals_query['residual'] = {'$lte': max_residual}
    residuals_data = list(request.app.image_residuals_collection.find(residuals_query, 
    {'_id': 0, 'image_hash': 1, 'residual': 1}).sort("residual", sort_order))

    images_data = []

    # Query to filter images based on dataset, date, and threshold_time
    imgs_query = {"task_input_dict.dataset": dataset}

    # Update the query based on provided start_date, end_date, and threshold_time
    if start_date and end_date:
        imgs_query['task_creation_time'] = {'$gte': start_date, '$lte': end_date}
    elif start_date:
        imgs_query['task_creation_time'] = {'$gte': start_date}
    elif end_date:
        imgs_query['task_creation_time'] = {'$lte': end_date}
    elif threshold_time:
        imgs_query['task_creation_time'] = {'$gte': threshold_time}

    # Loop to get filtered list of images and their residuals
    for data in residuals_data:
        # Adding filter based on image hash
        imgs_query['task_output_file_dict.output_file_hash'] = data['image_hash']
        img = request.app.completed_jobs_collection.find_one(imgs_query)

        # Only appending image to response if it is within date range
        if img is not None:
            images_data.append({
                'image_path': img['task_output_file_dict']['output_file_path'],
                'image_hash': data['image_hash'],
                'residual': data['residual']
            })
    
    # Applying offset and limit for pagination
    images_data = images_data[offset:offset+limit]

    return {"image_data": images_data}
