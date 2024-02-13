from pymongo import MongoClient
import pymongo
from tqdm import tqdm

# Connect to MongoDB
client = MongoClient('mongodb://192.168.3.1:32017/')
db = client['orchestration-job-db']

# Access the specific collection
collection = db["completed-jobs"]

# Fetch documents that need updating and convert cursor to a list to use with tqdm
documents = list(collection.find({}))

# Use tqdm to wrap the documents list for progress tracking
for document in tqdm(documents, desc="Processing documents"):
    # Check if 'top_k' exists within 'task_input_dict'
    if 'top_k' in document.get('task_input_dict', {}):
        # Prepare to remove the 'top_k' field from 'task_input_dict'
        update_fields = {'$unset': {'task_input_dict.top_k': ""}}
        
        # Perform the update on the current document
        collection.update_one({'_id': document['_id']}, update_fields)

print("Completed removing top_k from task_input_dict in documents.")
