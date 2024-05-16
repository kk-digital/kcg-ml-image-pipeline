from pymongo import MongoClient
from tqdm import tqdm

# Replace with your MongoDB connection string
client = MongoClient('mongodb://localhost:27017/')

# Replace with your database name
db = client['orchestration-job-db']

# Replace with your collections
source_collection = db['image_pair_ranking']
destination_collection = db['ranking_datapoints']

# Query to find documents where dataset="mech"
query = {"dataset": "mech"}

# Fetch documents from source collection
documents = list(source_collection.find(query))

total_documents = len(documents)
print(f"Total documents to migrate: {total_documents}")

# Iterate over documents and insert into the destination collection
for doc in tqdm(documents, desc="Migrating documents"):
    # Prepare the new document
    new_doc = {
        "file_name": doc["file_name"],
        "rank_model_id": 4,
        "task": doc["task"],
        "username": doc["username"],
        "image_1_metadata": doc["image_1_metadata"],
        "image_2_metadata": doc["image_2_metadata"],
        "selected_image_index": doc["selected_image_index"],
        "selected_image_hash": doc["selected_image_hash"],
        "training_mode": doc["training_mode"],
        "rank_active_learning_policy_id": doc["rank_active_learning_policy_id"],
        "datetime": doc["datetime"]
    }

    # Insert the new document into the destination collection
    destination_collection.insert_one(new_doc)

print("Migration completed successfully!")
