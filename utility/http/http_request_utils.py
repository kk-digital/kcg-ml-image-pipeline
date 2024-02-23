
import requests
import json
from urllib.parse import urlencode
from typing import Union

# Get the URL from base url with the query params
def get_url_with_query_params(base_url: str = 'http://localhost:8000', 
                              query_params: dict ={}) -> str:
    """
    Build a URL with query parameters by filtering out None values from the query_params dictionary.

    Args:
        base_url (str): The base URL to which the query parameters will be appended.
        query_params (dict): A dictionary containing the query parameters.

    Returns:
        str: The final URL with query parameters.
    """
    
    # Filter out None values from query_params
    query_params_filtered = {k: v for k, v in query_params.items() if v is not None}

    # Encode the filted query params into a query string
    query_string = urlencode(query_params_filtered)
    
    return base_url + '?' + query_string


# Custom Http request object 
def http_request(url: str, method: str = "GET", 
                 json_data: dict = None, 
                 headers: dict = None, 
                 params: dict = None) -> Union[dict, None]:
    """
    Make an HTTP request based on the specified method and handle the response.

    Args:
        url (str): The URL to make the request to.
        method (str): The HTTP method to use for the request (GET, POST, PUT).
        params (dict): The query parameters for the request (for GET requests).
        json_data (dict): The JSON data to send in the request body (for POST and PUT requests).
        headers (dict): The headers to include in the request.

    Returns:
        dict or None: The decoded JSON response if the request was successful, otherwise None.
    """
    
    # Initialize variables to store the response and decoded JSON response
    response = None
    decoded_response = None

    try:
        # Make an HTTP request depends on the method
        # if the method is GET then use requests.get()
        # if the method is POST then use requests.post()
        # if the method is PUT then use requests.put()
        # otherwise, return None
        if method == "GET":
            response = requests.get(url, params=params)
        elif method == "POST":
            response = requests.post(url, json=json_data, headers=headers)
        elif method == "PUT":
            response = requests.put(url, json=json_data, headers=headers)
        else :
            print(f"Method: {method} not supported")
            return decoded_response
        
        # Check if the response status code indicates a successful request (200 or 201)
        if response.status_code == 200 or response.status_code == 201:
            # Decode the JSON response if the request was successful
            decoded_response = json.loads(response.content.decode())
        else:
            print(f"Request failed with status code: {response.status_code}")

    except Exception as e:
        print('Request exception:', e)

    finally:
        if response:
            response.close()

    return decoded_response