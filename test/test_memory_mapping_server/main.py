from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tqdm import tqdm
import sys

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint to get clip-h vector for single image_global_id
@app.get("/get_clip_vector/{image_global_id}")
async def get_clip_vector(request: Request, image_global_id: int):
    
    # Retrieve the clip-h vector for the given image_global_id from the memory-mapped array
    if image_global_id < request.app.shape[0]:
        response_data = request.app.mmapped_array[image_global_id].tolist()
    else:
        raise HTTPException(status_code=404, detail="Image not found")

    return {
        "data": response_data
    }
        

# Endpoint to get clip-h vector for list of image_global_id
@app.post("/get_clip_vectors")
async def get_clip_vectors(request: Request, image_global_ids: list):
    clip_vectors = []
    for i in image_global_ids:
        if i < request.app.shape[0]:
            clip_vectors.append(request.app.mmmapped_array[i].tolist())
        else:
            raise HTTPException(status_code=404, detail="Image not found")

    response_data = clip_vectors
    return {
        "data": response_data
    }

# Endpoint to retrieve cache information
@app.get("/cache_info")
async def cache_info(request: Request):
    return {
        "data": {
            "num_clip_vectors_stored": len(request.app.mmapped_array),
            "size_of_mem_mapped_file": request.app.mmapped_array.nbytes / (1024**3),
            "count_requested": request.app.count_requested
        }
    }

# Middleware to track and increment the request count
@app.middleware("http")
async def increase_request_count(request: Request, call_next):
    # Increased the count of request when getted the request
    request.app.count_requested += 1

    # Call next function
    response = await call_next(request)
    return response

@app.on_event("startup")
def initialize():
    # set the count of request as 0
    app.count_requested = 0

    shape = (10000, 1280)
    dtype = np.float16

    # Create memory-mapped array
    filename = 'output/clip_vectors_{}.dat'.format(shape[0])

    with open(filename, 'w+b') as f:
        app.mmapped_array = np.memmap(f, dtype=dtype, mode='w+', shape=shape)

    # Initialize the memory mapped array
    for i in tqdm(range(shape[0]), desc="Initializing memory-mapped array"):
        app.mmapped_array[i, :] = np.random.rand(shape[1])

    # Add shape into app
    app.shape = shape

# Save memory mapped array
@app.on_event("shutdown")
def shutdown_db_client():
    app.mongodb_client.close()
    del app.mmapped_array