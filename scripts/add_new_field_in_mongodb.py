from pymongo import MongoClient
from collections import OrderedDict

# MongoDB connection details
MONGO_URI = "mongodb://192.168.3.1:32017/"
DATABASE_NAME = "orchestration-job-db"
COLLECTION_NAME = "ranking_datapoints"

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

        # Create an OrderedDict with the new field
        updated_data = OrderedDict([
            ("_id", document["_id"]),
            ("file_name", document.get("file_name", "")),
            *document.items(),  # Unpack the rest of the document fields
            (NEW_FIELD, NEW_VALUE),
            ("datetime", document.get("datetime", ""))
        ])
        
        # Update the document with the ordered data
        collection.replace_one({"_id": document["_id"]}, updated_data)
        print(f"Updated document ID: {document['_id']}")

    print("All documents updated successfully.")
    client.close()

if __name__ == "__main__":
    update_mongodb_documents()