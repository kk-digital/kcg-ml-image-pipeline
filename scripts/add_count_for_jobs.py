from pymongo import MongoClient
from bson import ObjectId
from tqdm import tqdm

# Configuration for MongoDB connection
client = MongoClient('mongodb://192.168.3.1:32017/')
db = client['orchestration-job-db']

def update_completed_jobs():
    completed_jobs = db['completed-jobs']
    tags_collection = db['image_tags']
    ranking_collection = db['image_pair_ranking']

    # Retrieve all jobs from completed_jobs_collection that do not have 'safe_to_delete' field
    all_jobs = list(completed_jobs.find({"safe_to_delete": {"$exists": False}}))

    # Iterate over each job in completed_jobs_collection with a progress bar
    for job in all_jobs:
        output_file_hash = job.get('task_output_file_dict', {}).get('output_file_hash', '')

        print(output_file_hash)

        # Count occurrences in image_tags_collection
        tag_count = tags_collection.count_documents({'image_hash': output_file_hash})

        # Count occurrences in image_pair_ranking_collection
        ranking_count = ranking_collection.count_documents(
            {'$or': [{'image_1_metadata.file_hash': output_file_hash}, 
                     {'image_2_metadata.file_hash': output_file_hash}]}
        )

        # Determine if the image is safe to delete
        safe_to_delete = tag_count == 0 and ranking_count == 0

        # Update the job document
        completed_jobs.update_one(
            {'_id': ObjectId(job['_id'])},
            {'$set': {
                'ranking_count': ranking_count,
                'safe_to_delete': safe_to_delete,
                'tag_count': tag_count,
            }}
        )

if __name__ == "__main__":
    update_completed_jobs()
