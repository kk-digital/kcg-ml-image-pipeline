import os
import json
import argparse

def load_job(dataset, image_name):
    directory = f"./{dataset}/{image_name}"
    json_path = f"{directory}/{image_name}.json"
    try:
        with open(json_path, "r") as file:
            # Check if file is empty
            if os.stat(json_path).st_size == 0:
                print(f"File {json_path} is empty. Skipping.")
                return None
            
            job = json.load(file)
            # Print in pretty JSON format
            pretty_job = json.dumps(job, ensure_ascii=False, indent=4, separators=(", ", ": "))
            print(pretty_job)
            return job
    except FileNotFoundError:
        print(f"No such file: {json_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in file: {json_path}. Error: {e}")
        return None


def load_all_jobs(dataset):
    dataset_path = f"./{dataset}/"
    job_count = 0

    # Iterate through each item in the dataset directory
    for item in sorted(os.listdir(dataset_path)):
        item_path = os.path.join(dataset_path, item)
        # Check if the item is a directory (which should correspond to an image name)
        if os.path.isdir(item_path):
            load_job(dataset, item)  # Pass the image name to load_job
            job_count += 1

    print(f"Total jobs loaded for dataset '{dataset}': {job_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load jobs for a given dataset.')
    parser.add_argument('dataset', type=str, help='The name of the dataset to load jobs from.')
    args = parser.parse_args()

    load_all_jobs(args.dataset)