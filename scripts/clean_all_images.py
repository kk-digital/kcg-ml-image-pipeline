from pymongo import MongoClient

# MongoDB connection details
MONGO_URI = "mongodb://192.168.3.1:32017/"
DATABASE_NAME = "orchestration-job-db"
ALL_IMAGES_COLLECTION = "all-images"

# Connect to MongoDB
print("Connecting to MongoDB...")
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
all_images_collection = db[ALL_IMAGES_COLLECTION]

# Update all documents to remove the creation_time field
print("Removing creation_time field from all documents in all-images collection...")
result = all_images_collection.update_many(
    {},
    {"$unset": {"creation_time": ""}}
)

print(f"Modified {result.modified_count} documents.")
print("Field removal completed successfully.")
client.close()
