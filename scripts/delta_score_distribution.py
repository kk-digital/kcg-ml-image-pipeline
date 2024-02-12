from pymongo import MongoClient
import matplotlib.pyplot as plt

# Specify your database name and MongoDB connection details
db_name = 'orchestration-job-db'
client = MongoClient('mongodb://192.168.3.1:32017/')
db = client[db_name]

# Access the specific collection
collection = db.image_pair_ranking_collection

# Initialize lists to hold the delta scores for 'linear' and 'elm-v1'
linear_scores = []
elm_v1_scores = []

# Iterate over documents in the collection
for doc in collection.find():
    # Check if 'delta_score' exists in the document
    delta_score = doc.get('delta_score')
    if delta_score:  # Proceed only if delta_score is not None
        # Append the scores, use a default of 0 if the specific model score is not found
        linear_scores.append(delta_score.get('linear', 0))
        elm_v1_scores.append(delta_score.get('elm-v1', 0))

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
