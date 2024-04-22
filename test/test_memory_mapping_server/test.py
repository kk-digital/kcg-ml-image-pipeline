import requests
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# Test script to measure "clip vectors/second" from the server
server_url = "http://localhost:8000"  # Replace with the actual server URL

elapsed_time_list = []
test_count_clip_vectors = []
# Measure the speed of serving clip vectors
for i in tqdm(range(10000, 500000, 10000), desc="Testing"):
    start_time = time.time()
    for _ in tqdm(range(i), desc="Sending request"):  # Send 1000 requests
        response = requests.get(f"{server_url}/get_clip_vector/123")  # Replace with an actual image_global_id
    end_time = time.time()

    elapsed_time = end_time - start_time

    test_count_clip_vectors.append(i)
    elapsed_time_list.append(elapsed_time)

response = requests.get(f"{server_url}/cache_info")
if response.status_code == 200:
    cache_info = response.json()["data"]

plt.figure(figsize=(10, 5))
plt.figtext(0.02, 0.7, ("num_clip_vectors_stored: {}\n"
            "size_of_mem_mapped_file: {}\n"
            "count_requested: {}\n".format(
                cache_info["num_clip_vectors_stored"], 
                cache_info["size_of_mem_mapped_file"],
                cache_info["count_requested"])))
plt.plot(test_count_clip_vectors, elapsed_time_list, marker='o')
plt.xlabel("Count of clip vectors")
plt.ylabel("Elapsed time")

plt.subplots_adjust(left=0.3)

plt.title("Test speed of loading from memory mapping file")