from pymongo import MongoClient

# MongoDB connection details
MONGO_URI = "mongodb://192.168.3.1:32017/"
DATABASE_NAME = "orchestration-job-db"
COLLECTION_NAME = "rank_definitions"

# Connect to MongoDB
print("Connecting to MongoDB...")
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Remove the 'rank_model_vector_index' field from all documents
try:
    update_result = collection.update_many(
        {"rank_model_vector_index": {"$exists": True}},  # Only target documents where the field exists
        {"$unset": {"rank_model_vector_index": ""}}  # Remove the field
    )
    print(f"Matched {update_result.matched_count} documents.")
    print(f"Modified {update_result.modified_count} documents.")

except Exception as e:
    print(f"An error occurred while removing 'rank_model_vector_index': {e}")

finally:
    client.close()

print("Field 'rank_model_vector_index' removed successfully from all documents.")
