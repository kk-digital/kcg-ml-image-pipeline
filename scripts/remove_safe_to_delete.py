from pymongo import MongoClient

# Configuration for MongoDB connection
client = MongoClient('mongodb://192.168.3.1:32017/')
db = client['orchestration-job-db']

def remove_safe_to_delete_fields():
    completed_jobs = db['completed-jobs']

    # Remove "safe_to_delete" and "safe-to-delete" fields from all documents
    result = completed_jobs.update_many(
        {},  # The empty query object {} means "match all documents"
        {'$unset': {'safe_to_delete': "", 'safe-to-delete': ""}}
    )

    # Output the result of the operation
    print(f"Fields removed in {result.modified_count} documents.")

if __name__ == "__main__":
    remove_safe_to_delete_fields()
