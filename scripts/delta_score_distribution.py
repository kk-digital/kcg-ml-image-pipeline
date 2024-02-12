from pymongo import MongoClient
import matplotlib.pyplot as plt

# Replace 'your_database_name' with the actual name of your database
db_name = 'orchestration-job-db'

# Connect to the MongoDB database
client = MongoClient('mongodb://192.168.3.1:32017/')
db = client[db_name]

# Access the specific collection
collection = db.image_pair_ranking_collection

# Retrieve delta scores for 'linear' and 'elm-v1'
linear_scores = []
elm_v1_scores = []

for doc in collection.find():
    delta_score = doc.get('delta_score', {})
    linear_scores.append(delta_score.get('linear', 0))
    elm_v1_scores.append(delta_score.get('elm-v1', 0))

# Plotting for Linear
plt.figure(figsize=(6, 4))
plt.hist(linear_scores, bins=30, alpha=0.7, label='Linear')
plt.xlabel('Delta Score')
plt.ylabel('Frequency')
plt.title('Distribution of Delta Scores for Linear')
plt.tight_layout()
plt.savefig('linear_delta_score_distribution.png')  # Save the figure for Linear
plt.close()  # Close the plot to avoid displaying it in the notebook

# Plotting for ELM-v1
plt.figure(figsize=(6, 4))
plt.hist(elm_v1_scores, bins=30, alpha=0.7, label='ELM-v1', color='orange')
plt.xlabel('Delta Score')
plt.ylabel('Frequency')
plt.title('Distribution of Delta Scores for ELM-v1')
plt.tight_layout()
plt.savefig('elm_v1_delta_score_distribution.png')  # Save the figure for ELM-v1
plt.close()  # Close the plot to avoid displaying it in the notebook

# Close the client connection to clean up resources
client.close()
