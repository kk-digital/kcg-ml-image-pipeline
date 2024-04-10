from pymongo import MongoClient

def get_task_type(img_hash):
    # Connect to the MongoDB database
    client = MongoClient('mongodb://192.168.3.1:32017/')
    db = client['orchestration-job-db']

    # Access the specific collection
    comleted_jobs_collection = db["completed-jobs"]

    # Query to find the job with the given image hash
    query = {"task_output_file_dict.output_file_hash": img_hash}
    job = comleted_jobs_collection.find_one(query)

    if not job:
        print(f"Failed to fetch job data for image hash {img_hash}. Job not found.")
        return None

    # Extract the task type
    task_type = job.get("task_type")
    return task_type

if __name__ == "__main__":
    img_hash = input("Enter the image hash: ")
    task_type = get_task_type(img_hash)
    if task_type:
        print(f"The task type for image hash {img_hash} is: {task_type}")
