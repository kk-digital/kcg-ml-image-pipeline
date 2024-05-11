# API guidelines

This guide includes basic guidelines about how the API endpoints are expected to work. Please read it and make sure each
new change made to the API follows it, to ensure the API is consistent and each endpoint works as expected.

- [Naming format](#naming-format)
- [HTTP methods](#http-methods)
- [Responses](#responses)
    - [Basic response format](#basic-response-format)
    - [Response content](#response-content)
    - [Delete responses](#delete-responses)
    - [Error responses](#error-responses)
- [Request params](#request-params)
- [Data validation](#data-validation)
- [Fast API documentation](#fast-api-documentation)

## Naming format

The names must be descriptive, to try to make the API a bit easier to underestand to anyone not familiar with it.

### Naming structure:

As much as possible, the API endpoints must have a naming structure similar to this one:
`{category}/{endpoint-name}`. The parts are:

- `category`: general category of the endpoint, like `datasets`, `images`, etc. The category normally references
a collection of elements, like `images`, so it should normally be in plural, but there are exceptions, like `clip`
and `queue`. All endpoints should have this part. 

- `endpoint-name`: it is the name of the endpoint and should say what it does. See the naming considerations section
for more information.

If needed, subcategories must be added after the category, but before the endpoint name.

### Naming considerations:

- The endpoint names should normally have a verb indicating the action and then the affected element. For example,
`/images/get-image-details` starts with the `get` verb, which clearly indicates that the endpoint if for obtaining data,
and then continues with the affected element. Another way to see this is adding first "what will I do" and then "with what".

- The names must be complete, something like `/images/get-details` may leave users asking "details of what". Do not asume that
having the endpoint in the  `images` categories responds that question, as the name may become ambiguous in the future if the
`/images/` category grows to include another kind of element, for any reason.

- All words must be separated with hyphens, not underscores or uppercase letters. This is `element-name`,
instead of `element_name` or `elementName`.

- If the endpoint performs a request with a special param or filter, it should be added to the name, but only if
it is a special case. For example, `/images/get-image-details-by-hash` may be a special case, different from
`/images/get-image-details`.

At this moment, we have some verbs that should be used at the start of the endpoint names in some situations:

- `get`: when getting data about an specific element.

- `list`: when getting a list with various elements.

- `count`: when getting how many elements of a specific kind there are.

- `delete`: when removing element.

Using those verbs instead of other allows to have more consistent names in the API.

## HTTP methods

The following http methods must be used:

`GET`: retrieves an element or elements.

`POST`: creates an element. It should normally return 201 as the http response code, instead of 200.

`PUT`: updates an element.

`DELETE`: removes an element.

## Responses

All API endpoints must return the responses using a standard format.

### Basic response format

There is a set of params with information about the request that all endpoints must include in the response.
The format is in the `StandardSuccessResponseV1` and `StandardErrorResponseV1` classes in `api_utils.py`.
However, this data must not be added manually, the data is added using the helper methods from `api_utils.py`.
Here is the process:

- The first line for all endpoints should be `response_handler = await ApiResponseHandlerV1.createInstance(request)`.
this creates the helper object that will keep track of the excecution stats and will allow to create the responses.

- If you are going to return a success response, use `return response_handler.create_success_response_v1(...)`, providing
to it the http status code and the response object that will be returned.

- If you are going to return a success response, but the endpoint is for deleting data, use
`return response_handler.create_success_delete_response_v1(...)` instead. See the section about delete responses for
more information.

- If you are going to return an error, use `return response_handler.create_error_response_v1(...)`. See the section about
error responses for more information.

### Response content

When using `return response_handler.create_success_response_v1(...)`, you must always provide an object as the response.
This means that if an endpoint is for returning an array, the response must not be a plain array, the array must be wrapped
in an object like `{list: []}`. The same goes for numeric responses and all other response types. This will allow to add additional
fileds to the response in the future, if needed, without introducing breaking changes.

If the endpoint is for getting a list of elements, and no element is found, return an empty array instead of error 404.

Also, the content of the response depends on what the endpoint does. Here is what each endpoint should normally return depending
on the http request method:

`GET`: the requested data.

`POST`: the contents of the created element.

`PUT`: the contents of the updated element.

`DELETE`: a special object (see below).

### Delete responses

Endpoints for deleting objects must return a success HTTP status code in the following circunstances:

- If the object was found and removed.
- If the object was not found, so there was no need to delete it.

This is because the objective of the delete operation is to make the element "not to be" in the server, and that
happens in both cases.

To return the response, use `return response_handler.create_success_delete_response_v1(...)`. This will return a
response like this:

```
{
   wasPresent: boolean
}
```

This object will allow to know if the element was in the server and was deleted (true) or if it was already not in
the server and nothing had to be done (false).

### Error responses

When there is a response with an error status code (like 4xx or 5xx), the API must always return the responses with
`return response_handler.create_error_response_v1(...)`. This function adds the error information to the responses.

In the `error_code` param you must use a value from the `ErrorCode` enum. This will help clients to know that
the error was thrown by the API code (so it was not an automatic server error) and what went wrong.

The current values are:

- `ELEMENT_NOT_FOUND` (the actual value is `2`): basically error 404. Use it when the requested element was not found.
Do not return this if a list with elements was requested and no element was found (return an empty array in this case).

- `INVALID_PARAMS` (the actual value is `3`): return this if any param sent in the request is invalid. This normally
refers to validation errors, but this response must be used also if the user sends as param the ID of an element that
can not be found (like asking to create a image in the category `2`, when there is no catgory with ID `2`). When returning
this, the http error code normally is 422.

- `OTHER_ERROR` (the actual value is `0`): Use this for any other type of error.

In the `error_string` param write small description of what when wrong, mainly for debugging purpouses.

In the `http_status_code` param set the http error code that must be returned, like 404 or 500.

## Request params

When requesting params, make sure to explicitly mark any optional param as optional. This applies to querystring
params and body params.

The params, both for query string and body, can be set as optional by code like `Optional[str] = None`.

This helps FastApi to validate the data and create the automatic documentation page.

## Data validation

All API endpoints must validate the params. The automatic features of FastAPI for validating data types must be used.
However, the basic validation FastApi provide is only for checking data types and if all the required params were added.
Please add any extra validation code that could be needed.

## Fast API documentation

In the code, the definition of all API endpoints must have:

- The endpoint name.

- The http code that is returned when the operation finishes correctly.

- An assigned FastAPI tag, to make it appear organized in the documentation. The tag is just the prefix in the endpoint name.

- A description, indicating what the endpoint does.

- A class with the structure of the response, to be able to know what the endpoint returns.

- The http error response codes the endpoint may return.

Here is an example:

```
@router.get("/datasets/list-datasets", 
            status_code=200,
            tags=["datasets"], 
            description="Gets a list with the name of all the datasets",
            response_model=StandardSuccessResponseV1[DatasetListResponse],
            responses=ApiResponseHandlerV1.listErrors([422, 500]))
```

- The first line contians the endpoint name.

- The `status_code` line indicates that the endpoint returns `200` as http response code when the operation finishes correctly.

- The `tags` line indicates that this endpoint is part of the `datasets` section. This is just the prefix of the endpoint.

- The `description` line includes a short description of what the endpoint does.

- The `response_model` line indicates that the endpoint returns a response with the structure defined in the `DatasetListResponse`
class. The value is `StandardSuccessResponseV1[DatasetListResponse]`, instead of just `DatasetListResponse`, to make sure that
all the default response fields are included in the documentation.

- The `responses` line indicates what http response codes the endpoint may return in case of error. It uses the
`ApiResponseHandlerV1.listErrors(...)` helper funtion to avoid having to write boilerplate code required by FastApi.

All those values are used to build the automatic API documentation page that FastAPI creates.
