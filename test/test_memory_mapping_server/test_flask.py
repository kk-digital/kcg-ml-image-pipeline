import requests
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# Test script to measure "clip vectors/second" from the server
server_url = "http://localhost:5000"  # Replace with the actual server URL

def format_duration(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_time = ""
    formatted_time += f"{hours}h " if hours > 0 else ""
    formatted_time += f"{minutes}m " if minutes > 0 else ""
    formatted_time += f"{seconds}s " if seconds > 0 else "0s"
    return formatted_time

def get_clip_vector(url, image_global_id):

    try:
        response = requests.get(f"{url}/get_clip_vector/{image_global_id}")  # Replace with an actual image_global_id
    except Exception as e:
        print("Error in getting clip vector, ", e)
    finally:
        response.close()

def main(increment, max_num, worker_count):
    elapsed_time_list = []
    loading_clip_vector_count_list = list(range(increment, max_num + increment, increment))
    # Measure the speed of serving clip vectors
        
    for clip_vector_count in loading_clip_vector_count_list:
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = []
            for _ in range(0, clip_vector_count):  # Send 1000 requests
                futures.append(executor.submit(get_clip_vector, server_url, random.randint(0, 10000)))
            for _ in tqdm(as_completed(futures), total=len(futures)):
                pass
        end_time = time.time()

        elapsed_time = end_time - start_time
        elapsed_time_list.append(elapsed_time)

    response = requests.get(f"{server_url}/cache_info")
    if response.status_code == 200:
        cache_info = response.json()["data"]
    else:
        return None

    total_elapsed_time = format_duration(int(sum(elapsed_time_list)))
    plt.figure(figsize=(10, 5))
    plt.figtext(0.02, 0.7, ("Number of clip vectors stored: {}\n"
                "Size of memory mapped file: {}(GB)\n"
                "Request Count: {}\n"
                "Total elapsed time: {}\n"
                "clip vecotors/second: {}\n"
                "Worker count: {}\n"
                .format(
                    cache_info["num_clip_vectors_stored"], 
                    format(cache_info["size_of_mem_mapped_file"], ".4f"),
                    cache_info["count_requested"],
                    total_elapsed_time,
                    format(sum(loading_clip_vector_count_list) / sum(elapsed_time_list), ".4f"),
                    worker_count
                )), fontsize=10)
    
    plt.plot(loading_clip_vector_count_list, elapsed_time_list, marker='o')
    plt.xlabel("Count of clip vectors")
    plt.ylabel("Elapsed time")

    plt.subplots_adjust(left=0.4)

    plt.title("Test speed of loading from memory mapping file")

    plt.savefig("output/{}_memory_mapping_server.png".format(datetime.now()))


def parse_args():

    args = argparse.ArgumentParser()
    args.add_argument("--increment", type=int, default=10000)
    args.add_argument("--max-num", type=int, default=100000)
    args.add_argument("--worker-count", type=int, default=8, help="Number of workers")
    return args.parse_args()

if __name__ == "__main__":

    args = parse_args()
    main(args.increment, args.max_num, args.worker_count)