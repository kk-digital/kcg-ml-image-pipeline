import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient

# Connect to the MongoDB database
client = MongoClient('mongodb://192.168.3.1:32017/')
db = client['orchestration-job-db']

# Access the specific collection
collection = db["image_pair_ranking"]

# Initialize lists to hold the delta scores for 'linear' and 'elm-v1'
linear_scores = []
elm_v1_scores = []

# Iterate over documents in the collection
for doc in collection.find():
    delta_score = doc.get('delta_score')
    if delta_score:
        linear_score = delta_score.get('linear')
        elm_v1_score = delta_score.get('elm-v1')
        if linear_score is not None:  # Ensure there is a score
            linear_scores.append(linear_score)
        if elm_v1_score is not None:  # Ensure there is a score
            elm_v1_scores.append(elm_v1_score)

# Plotting for Linear model
plt.figure(figsize=(18, 10))
plt.hist(linear_scores, bins=100, alpha=0.7, label='Linear')
plt.xlabel('Delta Score')
plt.ylabel('Frequency')
plt.title('Distribution of Delta Scores for Linear')
xmin, xmax = plt.xlim()  # Get the min and max of the current x-axis range
plt.xticks(np.arange(start=xmin, stop=xmax, step=0.5))
plt.tight_layout()
plt.savefig('linear_delta_score_distribution.png')
plt.close()

# Plotting for ELM-v1 model
plt.figure(figsize=(6, 4))
plt.hist(elm_v1_scores, bins=100, alpha=0.7, label='ELM-v1', color='orange')
plt.xlabel('Delta Score')
plt.ylabel('Frequency')
plt.title('Distribution of Delta Scores for ELM-v1')
xmin, xmax = plt.xlim()  # Get the min and max of the current x-axis range
plt.xticks(np.arange(start=xmin, stop=xmax, step=0.5))
plt.tight_layout()
plt.savefig('elm_v1_delta_score_distribution.png')
plt.close()

# Close the client connection
client.close()
print("Disconnected from the database.")
