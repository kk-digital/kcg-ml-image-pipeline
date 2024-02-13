from pymongo import MongoClient
import pymongo

# Connect to MongoDB
client = MongoClient('mongodb://192.168.3.1:32017/')
db = client['orchestration-job-db']

# Access the specific collection
collection = db["completed-jobs"]

# Fetch documents that need updating
documents = collection.find({})

for document in documents:
    update_fields = {}
    prompt_generation_data = document.get('prompt_generation_data', {})
    
    # Handling top_k from task_input_dict
    if 'top_k' in document.get('task_input_dict', {}):
        top_k_value = document['task_input_dict']['top_k']
        prompt_generation_data['top-k'] = top_k_value  # Add top-k to prompt_generation_data
        # Prepare to remove the original top_k field from task_input_dict
        
    
    # Update prompt_generation_data with new values or structure
    update_fields['$set'] = {'prompt_generation_data': prompt_generation_data}
    
    # Perform the update on the current document
    collection.update_one({'_id': document['_id']}, update_fields)

print("Completed updating documents.")
