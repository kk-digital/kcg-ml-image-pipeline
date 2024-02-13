from pymongo import MongoClient
import pymongo

# Connect to MongoDB
client = MongoClient('mongodb://192.168.3.1:32017/')
db = client['orchestration-job-db']

# Access the specific collection
collection = db["completed-jobs"]

# Fetch documents that need updating
documents = collection.find({})

for doc in documents:
    update_fields = {}
    prompt_generation_data = doc.get('prompt_generation_data', {})
    
    # Handling top_k from task_input_dict
    if 'top_k' in doc.get('task_input_dict', {}):  # Correctly access top_k within task_input_dict
        top_k_value = doc['task_input_dict']['top_k']  # Retrieve the top_k value
        prompt_generation_data['top-k'] = top_k_value  # Move it inside prompt_generation_data

    # Update prompt_generation_data with new values or structure
    update_fields['$set'] = {
        'prompt_generation_data': prompt_generation_data
    }

    # Perform the update
    collection.update_one({'_id': doc['_id']}, update_fields)

print("Documents updated successfully.")

