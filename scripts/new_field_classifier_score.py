import uuid
from pymongo import MongoClient

# MongoDB connection details
MONGO_URI = "mongodb://192.168.3.1:32017/"
DATABASE_NAME = "orchestration-job-db"
COLLECTION_NAME = "image_classifier_scores"

# New field to add
NEW_FIELD = "image_source"

def update_mongodb_documents():
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    # Collections to check for uuids
    completed_jobs_collection = db["completed-jobs"] 
    extracts_collection = db["extracts"]
    external_images_collection = db["external_images"]

    # Find all documents in the collection
    documents = collection.find({})

    for document in documents:
        # Check if the new field already exists
        if NEW_FIELD in document:
            print(f"Skipping document ID: {document['_id']} as it already contains the field '{NEW_FIELD}'.")
            continue

        # Determine the image source based on uuid
        document_uuid = document.get("uuid")
        if document_uuid:
            image_query = {
                '$or': [
                    {'uuid': document_uuid},
                    {'uuid': uuid.UUID(document_uuid)}
                ]
            }
            if completed_jobs_collection.find_one(image_query):
                document[NEW_FIELD] = "generated_image"
            elif extracts_collection.find_one(image_query):
                document[NEW_FIELD] = "extract_image"
            elif external_images_collection.find_one(image_query):
                document[NEW_FIELD] = "external_image"
            else:
                print(f"UUID {document_uuid} not found in any collections.")
                continue  # Skip if uuid is not found in any collection
        else:
            print(f"Document ID: {document['_id']} does not have a uuid.")
            continue  # Skip if no uuid in document

        # Update the document
        collection.replace_one({"_id": document["_id"]}, document)
        print(f"Updated document ID: {document['_id']} with image_source: {document[NEW_FIELD]}")

    print("All documents updated successfully.")
    client.close()

if __name__ == "__main__":
    update_mongodb_documents()
