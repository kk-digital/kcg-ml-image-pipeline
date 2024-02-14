from pymongo import MongoClient
from bson import ObjectId
from tqdm import tqdm

# Connect to MongoDB
client = MongoClient('mongodb://192.168.3.1:32017/')
db = client['orchestration-job-db']
collection = db["completed-jobs"]

# Specify the task types you want to update
task_types_to_update = ["image_generation_sd_1_5", "inpainting_sd_1_5"]

# Fetch documents with the specified task types
documents = list(collection.find({"task_type": {"$in": task_types_to_update}}))

# Use tqdm to show progress
for doc in tqdm(documents, desc="Updating documents"):
    # Skip documents without 'task_attributes_dict'
    if 'task_attributes_dict' not in doc:
        continue
    
    # Construct a new document with the desired field order, safely accessing 'task_attributes_dict'
    new_doc = {
        "_id": doc["_id"],
        "task_type": doc["task_type"],
        "uuid": doc["uuid"],
        "model_name": doc["model_name"],
        "model_file_name": doc["model_file_name"],
        "model_file_path": doc["model_file_path"],
        "model_hash": doc.get("model_hash"),  # Use .get() to safely access 'model_hash'
        "task_creation_time": doc["task_creation_time"],
        "task_start_time": doc["task_start_time"],
        "task_completion_time": doc["task_completion_time"],
        "task_error_str": doc.get("task_error_str"),
        "task_input_dict": doc["task_input_dict"],
        "task_input_file_dict": doc.get("task_input_file_dict"),
        "task_output_file_dict": doc["task_output_file_dict"],
        "task_attributes_dict": doc["task_attributes_dict"],  # Already checked for existence
        "prompt_generation_data": doc.get("prompt_generation_data"),
    }
    
    # Replace the old document with the new document
    collection.replace_one({'_id': ObjectId(doc['_id'])}, new_doc)

print(f"Documents of types {task_types_to_update} updated with reordered fields.")
