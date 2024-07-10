from pymongo import MongoClient

# MongoDB connection details
MONGO_URI = "mongodb://192.168.3.1:32017/"
DATABASE_NAME = "orchestration-job-db"
COLLECTION_NAME = "image_tags"

# New value for the field
OLD_VALUE = "extracts"
NEW_VALUE = "extract_image"
FIELD_NAME = "image_source"

def update_mongodb_documents():
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    # Find all documents in the collection where image_source is "extracts"
    documents = collection.find({FIELD_NAME: OLD_VALUE})

    for document in documents:
        # Update the image_source field to "extract_image"
        document[FIELD_NAME] = NEW_VALUE

        # Update the document in the collection
        collection.replace_one({"_id": document["_id"]}, document)
        print(f"Updated document ID: {document['_id']} with {FIELD_NAME}: {document[FIELD_NAME]}")

    print("All documents updated successfully.")
    client.close()

if __name__ == "__main__":
    update_mongodb_documents()
