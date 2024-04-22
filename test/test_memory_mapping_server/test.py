import requests
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# Test script to measure "clip vectors/second" from the server
server_url = "http://localhost:8000"  # Replace with the actual server URL

def format_duration(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_time = ""
    formatted_time += f"{hours}h " if hours > 0 else ""
    formatted_time += f"{minutes}m " if minutes > 0 else ""
    formatted_time += f"{seconds}s " if seconds > 0 else "0s"
    return formatted_time

def main(increment, max_num):
    elapsed_time_list = []
    test_count_clip_vectors = list(range(increment, max_num, increment))
    # Measure the speed of serving clip vectors
    for i in tqdm(test_count_clip_vectors, desc="Testing"):
        start_time = time.time()
        for _ in tqdm(range(i), desc="Sending request"):  # Send 1000 requests
            response = requests.get(f"{server_url}/get_clip_vector/123")  # Replace with an actual image_global_id
        end_time = time.time()

        elapsed_time = end_time - start_time
        elapsed_time_list.append(elapsed_time)

    response = requests.get(f"{server_url}/cache_info")
    if response.status_code == 200:
        cache_info = response.json()["data"]

    total_elapsed_time = format_duration(int(sum(elapsed_time_list)))
    plt.figure(figsize=(10, 5))
    plt.figtext(0.02, 0.7, ("Number of clip vectors stored: {}\n"
                "Size of memory mapped file: {}(GB)\n"
                "Request Count: {}\n"
                "Total elapsed time: {}\n"
                "clip vecotors/second: {}\n"
                .format(
                    cache_info["num_clip_vectors_stored"], 
                    format(cache_info["size_of_mem_mapped_file"], ".4f"),
                    cache_info["count_requested"],
                    total_elapsed_time,
                    format(sum(test_count_clip_vectors) / sum(elapsed_time_list), ".4f")
                )), fontsize=10)
    
    plt.plot(test_count_clip_vectors, elapsed_time_list, marker='o')
    plt.xlabel("Count of clip vectors")
    plt.ylabel("Elapsed time")

    plt.subplots_adjust(left=0.4)

    plt.title("Test speed of loading from memory mapping file")

    plt.savefig("output/{}_memory_mapping_server.png".format(datetime.now()))


def parse_args():

    args = argparse.ArgumentParser()
    args.add_argument("--increment", type=int, default=10000)
    args.add_argument("--max-num", type=int, default=500000)
    return args.parse_args()

if __name__ == "__main__":

    args = parse_args()
    main(args.increment, args.max_num)