from pymongo import MongoClient

# Connect to the MongoDB database
client = MongoClient('mongodb://192.168.3.1:32017/')
db = client['orchestration-job-db']

# Access the specific collection
collection = db["image_pair_ranking"]

# Initialize counters for delta scores in the range -10 to 0
linear_count_in_range = 0
elm_v1_count_in_range = 0

# Initialize total counters
total_linear_scores = 0
total_elm_v1_scores = 0

# Iterate over documents in the collection
for doc in collection.find():
    delta_score = doc.get('delta_score')
    if delta_score:
        linear_score = delta_score.get('linear')
        elm_v1_score = delta_score.get('elm-v1')
        if linear_score is not None:  # Ensure there is a score
            total_linear_scores += 1
            if -10 <= linear_score <= 0:
                linear_count_in_range += 1
        if elm_v1_score is not None:  # Ensure there is a score
            total_elm_v1_scores += 1
            if -10 <= elm_v1_score <= 0:
                elm_v1_count_in_range += 1

# Calculate percentages
linear_percentage = (linear_count_in_range / total_linear_scores * 100) if total_linear_scores > 0 else 0
elm_v1_percentage = (elm_v1_count_in_range / total_elm_v1_scores * 100) if total_elm_v1_scores > 0 else 0

# Print out the results
print(f"Percentage of Linear model delta scores from -10 to 0: {linear_percentage:.2f}%")
print(f"Percentage of ELM-v1 model delta scores from -10 to 0: {elm_v1_percentage:.2f}%")

# Close the client connection
client.close()
