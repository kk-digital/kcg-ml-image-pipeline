from pymongo import MongoClient
import pymongo

# Connect to MongoDB
client = MongoClient('mongodb://192.168.3.1:32017/')
db = client['orchestration-job-db']

# Access the specific collection
collection = db["completed-jobs"]

uuid_to_update = "796a0c9e-9ffd-4f89-aff0-d26e924f2ef3"

# Fetch the document that matches the specified UUID
document = collection.find_one({"uuid": uuid_to_update})

if document:
    update_fields = {}
    prompt_generation_data = document.get('prompt_generation_data', {})
    
    # Handling top_k from task_input_dict
    if 'top_k' in document.get('task_input_dict', {}):
        top_k_value = document['task_input_dict']['top_k']
        prompt_generation_data['top-k'] = top_k_value
        # Prepare to remove the original top_k field from task_input_dict
        update_fields['$unset'] = {'task_input_dict.top_k': ""}
    
    # Update prompt_generation_data with new values or structure
    update_fields['$set'] = {
        'prompt_generation_data': prompt_generation_data
    }
    
    # Perform the update on the document with the specified UUID
    collection.update_one({'uuid': uuid_to_update}, update_fields)
    
    print(f"Document with UUID {uuid_to_update} updated successfully.")
else:
    print(f"No document found with UUID {uuid_to_update}.")

