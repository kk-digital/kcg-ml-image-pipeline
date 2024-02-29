import sys
from typing import Union

base_directory = "./"
sys.path.insert(0, base_directory)

from typing import Union

from utility.http.constants import SERVER_ADDRESS
from utility.http.http_request_utils import get_url_with_query_params, http_request


# Get request to get an available job
def http_get_job(model_type: str = None, worker_type: str = None) -> Union[dict, None]:
    """
    Construct a URL with query parameters by appending the specified parameters 'task_type' and 'model_type' to specific URL.

    Args:
    worker_type (str): The type of worker for the job.
    model_type (str): The type of model for the job.

    Returns:
    dict | None: The response from the server after making an HTTP GET request with the constructed URL and query parameters.
    """

    url = SERVER_ADDRESS + "/queue/inpainting-generation/get-job"

    query_params = {
        "task_type": worker_type,
        "model_type": model_type,
    }
    
    url = get_url_with_query_params(url, query_params)

    return http_request(url, "GET")


# Post request to add job
def http_add_job(job: dict) -> Union[dict, None]:
    """
    Make an HTTP POST request to add a job to the server at the specified endpoint.
 
    Args:
    job (dict): A dictionary containing the job information to be added.

    Returns:
    dict | None: The response from the server after adding the job, 
        or None if there is no response.
    """

    url = SERVER_ADDRESS + "/queue/inpainting-generation/add-job"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data
    
    return http_request(url, "POST", json_data=job, headers=headers)


# Post request to add job
def http_update_job_completed(job: dict) -> Union[dict, None]:
    """
    Make an HTTP PUT request to update the completion status of a job on the server at the specified endpoint.

    Args:
    job (dict): A dictionary containing the job information to update the completion status.

    Returns:
    dict | None: The response from the server after updating the job completion status, or None if there is no response.
    """

    url = SERVER_ADDRESS + "/queue/inpainting-generation/update-completed"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data
    return http_request(url, "PUT", json_data=job, headers=headers)


# Update request to update job status to failed
def http_update_job_failed(job: dict) -> Union[dict, None]:
    """
    Make an HTTP PUT request to update the failed status of a job on the server at the specified endpoint "/queue/inpainting-generation/update-failed".

    Args:
    job (dict): A dictionary containing the job information to update the failed status.

    Returns:
    dict | None: The response from the server after updating the job failed status, or None if there is no response.
    """

    url = SERVER_ADDRESS + "/queue/inpainting-generation/update-failed"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data
    return http_request(url, "PUT", json_data=job, headers=headers)