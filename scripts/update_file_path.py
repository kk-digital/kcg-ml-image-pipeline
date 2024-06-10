from pymongo import MongoClient
from tqdm import tqdm

# MongoDB configuration
MONGODB_URI = 'mongodb://192.168.3.1:32017/'
MONGODB_DB_NAME = 'orchestration-job-db'
MONGODB_COLLECTION_NAME = 'external_images'

# Initialize MongoDB client
mongo_client = MongoClient(MONGODB_URI)
mongodb = mongo_client[MONGODB_DB_NAME]
collection = mongodb[MONGODB_COLLECTION_NAME]

def update_file_paths():
    try:
        # Query to find documents with the specific dataset in the file_path
        query = {"file_path": {"$regex": "^pixel-art-dataset/"}}
        
        # Find the documents that match the query
        documents = list(collection.find(query))

        # Update each document
        for doc in tqdm(documents, desc="Updating file paths"):
            old_file_path = doc['file_path']
            new_file_path = f"external/{old_file_path}"
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"file_path": new_file_path}}
            )
            print(f"Updated file_path from {old_file_path} to {new_file_path}")

        print("File path update complete.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    update_file_paths()
