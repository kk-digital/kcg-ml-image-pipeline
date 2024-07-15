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

# Remove all documents from the all_images_collection
result = all_images_collection.delete_many({})
print(f"Deleted {result.deleted_count} documents from {ALL_IMAGES_COLLECTION} collection.")

# Close the MongoDB connection
client.close()
print("MongoDB connection closed.")
