from pymongo import MongoClient
from bson import ObjectId
from tqdm import tqdm

# Specify the MongoDB connection details
client = MongoClient('mongodb://192.168.3.1:32017/')
db = client['orchestration-job-db']
collection = db["completed-jobs"]

# Define the original and new task types
original_task_type = "clip_calculation_task"
new_task_type = "clip_calculation_task_sd_1_5"

# Fetch documents with the specified original task type
documents = list(collection.find({"task_type": original_task_type}))

# Use tqdm to show progress
for doc in tqdm(documents, desc="Updating task types"):
    # Update the task_type to the new value
    update_result = collection.update_one(
        {'_id': ObjectId(doc['_id'])},
        {'$set': {'task_type': new_task_type}}
    )

    # Optional: print out the id of documents as they're updated for verification or tracking
    if update_result.modified_count > 0:
        print(f"Updated document ID: {doc['_id']}")

print(f"Completed updating documents from {original_task_type} to {new_task_type}.")
