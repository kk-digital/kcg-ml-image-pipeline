from pymongo import MongoClient, UpdateOne

# Connect to MongoDB
client = MongoClient('mongodb://192.168.3.1:32017/')
db = client['orchestration-job-db']  # Update with your actual database name
image_tags_collection = db['image_tags']  # Update with your actual collection name

# Fetch documents where tag_type = 0
documents = list(image_tags_collection.find({"tag_type": 0}))

# Prepare bulk update operations
bulk_operations = [
    UpdateOne({"_id": doc["_id"]}, {"$set": {"tag_type": 1}})
    for doc in documents
]

# Execute bulk update if there are operations to perform
if bulk_operations:
    result = image_tags_collection.bulk_write(bulk_operations)
    print(f"Successfully updated {result.modified_count} documents.")
else:
    print("No documents to update.")
