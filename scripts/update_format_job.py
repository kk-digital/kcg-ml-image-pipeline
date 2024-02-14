from pymongo import MongoClient
import bson

# Specify the UUID of the document you want to update
uuid_to_update = "aa7fa05b-77dc-4500-a438-8ed81370b588"

# Connect to MongoDB
client = MongoClient('mongodb://192.168.3.1:32017/')
db = client['orchestration-job-db']
collection = db["completed-jobs"]

# Fetch the document by UUID
doc = collection.find_one({"uuid": uuid_to_update})

if doc:
    # Construct a new document with the desired field order
    new_doc = {
        "_id": doc["_id"],  # Preserve the original _id
        "task_type": doc["task_type"],
        "uuid": doc["uuid"],
        "model_name": doc["model_name"],
        "model_file_name": doc["model_file_name"],
        "model_file_path": doc["model_file_path"],
        # Ensure model_hash is included and placed correctly
        "model_hash": doc.get("model_hash"),  
        "task_creation_time": doc["task_creation_time"],
        "task_start_time": doc["task_start_time"],
        "task_completion_time": doc["task_completion_time"],
        "task_error_str": doc.get("task_error_str"),
        "task_input_dict": doc["task_input_dict"],
        "task_input_file_dict": doc.get("task_input_file_dict"),
        "task_output_file_dict": doc["task_output_file_dict"],
        "task_attributes_dict": doc["task_attributes_dict"],
        "prompt_generation_data": doc["prompt_generation_data"]
    }

    # Additional fields if exist
    # Make sure to include all fields from your original document structure

    # Replace the old document with the new document
    collection.replace_one({'_id': bson.ObjectId(doc['_id'])}, new_doc)

    print(f"Document with UUID {uuid_to_update} updated successfully.")
else:
    print(f"No document found with UUID {uuid_to_update}.")



