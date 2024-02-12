from pymongo import MongoClient
import matplotlib.pyplot as plt

# Specify your database name and MongoDB connection details
db_name = 'orchestration-job-db'
client = MongoClient('mongodb://192.168.3.1:32017/')
db = client[db_name]

# Access the specific collection
collection = db.image_pair_ranking_collection

print("Connected to the database and collection successfully.")

# Initialize lists to hold the delta scores for 'linear' and 'elm-v1'
linear_scores = []
elm_v1_scores = []

document_count = 0  # To count how many documents are processed

# Iterate over documents in the collection
for doc in collection.find():
    document_count += 1
    # Check if 'delta_score' exists in the document
    delta_score = doc.get('delta_score')
    if delta_score:  # Proceed only if delta_score is not None
        # Append the scores, use a default of 0 if the specific model score is not found
        linear_score = delta_score.get('linear', 0)
        elm_v1_score = delta_score.get('elm-v1', 0)
        linear_scores.append(linear_score)
        elm_v1_scores.append(elm_v1_score)
    else:
        print(f"Document ID: has no delta_score field.")

print(f"Processed {document_count} documents.")

# Check if any scores were collected
if not linear_scores and not elm_v1_scores:
    print("No delta scores were found. Check if the documents in the collection have the 'delta_score' field.")

# Plotting for Linear model
plt.figure(figsize=(6, 4))
plt.hist(linear_scores, bins=30, alpha=0.7, label='Linear')
plt.xlabel('Delta Score')
plt.ylabel('Frequency')
plt.title('Distribution of Delta Scores for Linear')
plt.tight_layout()
plt.savefig('linear_delta_score_distribution.png')
plt.close()

# Plotting for ELM-v1 model
plt.figure(figsize=(6, 4))
plt.hist(elm_v1_scores, bins=30, alpha=0.7, label='ELM-v1', color='orange')
plt.xlabel('Delta Score')
plt.ylabel('Frequency')
plt.title('Distribution of Delta Scores for ELM-v1')
plt.tight_layout()
plt.savefig('elm_v1_delta_score_distribution.png')
plt.close()

# Close the client connection
client.close()
print("Disconnected from the database.")
