from pymongo import MongoClient
from collections import defaultdict

# MongoDB connection details
MONGO_URI = "mongodb://192.168.3.1:32017/"
DATABASE_NAME = "orchestration-job-db"
COLLECTION_NAME = "image_tags"

def list_duplicates():
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    # Find all documents with image_source set to "extract_image"
    documents = collection.find({"image_source": "extract_image"})

    # Dictionary to track unique pairs of image_hash and tag_id
    unique_pairs = defaultdict(list)

    # Identify duplicates
    for document in documents:
        image_hash = document.get("image_hash")
        tag_id = document.get("tag_id")

        if image_hash and tag_id is not None:
            unique_pairs[(image_hash, tag_id)].append(document)

    # List duplicates
    duplicates = []
    for docs in unique_pairs.values():
        if len(docs) > 1:
            duplicates.append(docs)

    # Print duplicates
    if duplicates:
        print("Found duplicates:")
        for duplicate_set in duplicates:
            for doc in duplicate_set:
                print(doc)
            print("---")
    else:
        print("No duplicates found.")

    client.close()

if __name__ == "__main__":
    list_duplicates()
