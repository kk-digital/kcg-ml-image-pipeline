import time
from pymongo import MongoClient
from bson import ObjectId

# Configuration for MongoDB connection
client = MongoClient('mongodb://192.168.3.1:32017/')
db = client['orchestration-job-db']

def update_completed_jobs():
    completed_jobs = db['completed-jobs']
    tags_collection = db['image_tags']
    ranking_collection = db['image_pair_ranking']

    # Retrieve all jobs from completed_jobs_collection that do not have 'safe_to_delete' field
    start_time = time.time()
    all_jobs = list(completed_jobs.find({"safe_to_delete": {"$exists": False}}))
    print("Time to fetch jobs: {:.2f} seconds".format(time.time() - start_time))

    for job in all_jobs:
        output_file_hash = job.get('task_output_file_dict', {}).get('output_file_hash', '')

        # Measure time to get output_file_hash
        start_time = time.time()
        print(output_file_hash)
        print("Time to get and print hash: {:.2f} seconds".format(time.time() - start_time))

        # Count occurrences in image_tags_collection
        start_time = time.time()
        tag_count = tags_collection.count_documents({'image_hash': output_file_hash})
        print("Time to count tags: {:.2f} seconds".format(time.time() - start_time))

        # Count occurrences in image_pair_ranking_collection
        start_time = time.time()
        count1 = ranking_collection.count_documents({'image_1_metadata.file_hash': output_file_hash})
        count2 = ranking_collection.count_documents({'image_2_metadata.file_hash': output_file_hash})
        ranking_count = count1 + count2
        print("Time to count rankings: {:.2f} seconds".format(time.time() - start_time))

        # Determine if the image is safe to delete
        start_time = time.time()
        safe_to_delete = tag_count == 0 and ranking_count == 0
        print("Time to determine safety for deletion: {:.2f} seconds".format(time.time() - start_time))

        # Update the job document
        start_time = time.time()
        completed_jobs.update_one(
            {'_id': ObjectId(job['_id'])},
            {'$set': {
                'ranking_count': ranking_count,
                'safe_to_delete': safe_to_delete,
                'tag_count': tag_count,
            }}
        )
        print("Time to update job: {:.2f} seconds".format(time.time() - start_time))

if __name__ == "__main__":
    update_completed_jobs()
