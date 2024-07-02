from pymongo import MongoClient

# MongoDB connection details
MONGO_URI = "mongodb://192.168.3.1:32017/"
DATABASE_NAME = "orchestration-job-db"
COLLECTION_NAME = "image_classifier_scores"

# New field to add
NEW_FIELD = "image_source"
NEW_VALUE = "generated_image"

def update_mongodb_documents():
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    # Find all documents in the collection
    documents = collection.find({})

    for document in documents:
        # Check if the new field already exists
        if NEW_FIELD in document:
            print(f"Skipping document ID: {document['_id']} as it already contains the field '{NEW_FIELD}'.")
            continue

        # Add the new field to the document
        document[NEW_FIELD] = NEW_VALUE
        
        # Update the document
        collection.replace_one({"_id": document["_id"]}, document)
        print(f"Updated document ID: {document['_id']}")

    print("All documents updated successfully.")
    client.close()

if __name__ == "__main__":
    update_mongodb_documents()
