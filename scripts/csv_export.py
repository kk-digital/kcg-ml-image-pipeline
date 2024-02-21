import csv
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://192.168.3.1:32017/')
db = client['orchestration-job-db']  # Change this to your actual database name

# Access collections
tag_defs = db['tag_definitions']
tag_cats = db['tag_categories']
image_tags = db['image_tags']
completed_jobs = db['completed-jobs']

# Prepare CSV file for output
csv_file = 'tags_export.csv'
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Updated the header row to include the Tag Type field
    writer.writerow(['Tag ID', 'Tag String', 'Tag Type', 'Tag Category ID', 'Tag Category String', 'Image Hash', 'Image File Path', 'Prompt Generation Policy'])

    # Iterate over image tags
    for image_tag in image_tags.find():
        # Retrieve tag definition
        tag_def = tag_defs.find_one({'tag_id': image_tag['tag_id']})
        
        # Skip if tag definition is not found
        if not tag_def:
            continue

        # Retrieve tag category string and ID
        tag_cat = tag_cats.find_one({'tag_category_id': tag_def.get('tag_category_id')})
        tag_category_string = tag_cat['tag_category_string'] if tag_cat else 'N/A'
        tag_category_id = tag_cat['tag_category_id'] if tag_cat else 'N/A'
        
        # Retrieve prompt generation policy using image hash from completed jobs collection
        job = completed_jobs.find_one({'task_output_file_dict.output_file_hash': image_tag['image_hash']})
        prompt_generation_policy = job.get('prompt_generation_data', {}).get('prompt_generation_policy', 'N/A') if job else 'N/A'
        
        # Added tag_type retrieval from image_tag
        tag_type = image_tag.get('tag_type', 'N/A')
        
        # Write to CSV including the new Tag Type column
        writer.writerow([tag_def['tag_id'], tag_def['tag_string'], tag_type, tag_category_id, tag_category_string, image_tag['image_hash'], image_tag['file_path'], prompt_generation_policy])

print(f"CSV file '{csv_file}' has been created.")
