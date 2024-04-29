from flask import Flask, request, jsonify
import numpy as np
from tqdm import tqdm

app = Flask(__name__)

# CORS configuration
from flask_cors import CORS
CORS(app)

# Initialize the memory-mapped array
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

# Add request count into app
app.count_requested = 0

# Endpoint to get clip-h vector for single image_global_id
@app.route('/get_clip_vector/<int:image_global_id>', methods=['GET'])
def get_clip_vector(image_global_id):
    # Retrieve the clip-h vector for the given image_global_id from the memory-mapped array
    if image_global_id < app.shape[0]:
        response_data = app.mmapped_array[image_global_id].tolist()
    else:
        return jsonify({"detail": "Image not found"}), 404

    return jsonify({"data": response_data})

# Endpoint to get clip-h vector for list of image_global_id
@app.route('/get_clip_vectors', methods=['POST'])
def get_clip_vectors():
    image_global_ids = request.get_json()['image_global_ids']
    clip_vectors = []
    for i in image_global_ids:
        if i < app.shape[0]:
            clip_vectors.append(app.mmapped_array[i].tolist())
        else:
            return jsonify({"detail": "Image not found"}), 404

    response_data = clip_vectors
    return jsonify({"data": response_data})

# Endpoint to retrieve cache information
@app.route('/cache_info', methods=['GET'])
def cache_info():
    return jsonify({
        "data": {
            "num_clip_vectors_stored": len(app.mmapped_array),
            "size_of_mem_mapped_file": app.mmapped_array.nbytes / (1024 ** 3),
            "count_requested": app.count_requested
        }
    })

# Middleware to track and increment the request count
@app.before_request
def increase_request_count():
    # Increase the count of request when getting the request
    app.count_requested += 1

if __name__ == '__main__':
    app.run(debug=True)
