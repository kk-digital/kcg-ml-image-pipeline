import csv
from pymongo import MongoClient
import sys

# Assuming the base directory is correctly set to include your custom modules
base_directory = "./"
sys.path.insert(0, base_directory)

# Importing your custom schema class for TagDefinition
from orchestration.api.mongo_schemas import TagDefinition

# MongoDB connection setup
client = MongoClient("mongodb://localhost:27017/")  # Update this with your actual MongoDB URI
db = client['orchestration-job-db']  # Update this with your actual database name

# Collections
tag_definitions_collection = db['tag_definitions']
image_tags_collection = db['image_tags']

# Fetch tag definitions
tag_definitions_cursor = tag_definitions_collection.find({})
tag_definitions = [TagDefinition(**tag_data).to_dict() for tag_data in tag_definitions_cursor]

# Count the number of images per tag
tag_image_count = {}
for tag_data in image_tags_collection.find({}, {"tag_id": 1}):
    tag_id = tag_data.get('tag_id')
    if tag_id is not None:
        tag_image_count[tag_id] = tag_image_count.get(tag_id, 0) + 1

# Combine tag definitions with image counts
export_data = []
for tag in tag_definitions:
    tag_id = tag['tag_id']
    tag['number_of_images_tagged'] = tag_image_count.get(tag_id, 0)
    export_data.append(tag)

# Export to CSV
csv_columns = ['tag_id', 'tag_string', 'tag_category', 'tag_description', 'tag_vector_index', 'user_who_created', 'creation_time', 'number_of_images_tagged']
csv_file = "TagsExport.csv"

try:
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in export_data:
            writer.writerow(data)
    print("Data exported successfully to", csv_file)
except IOError as e:
    print("I/O error:", e)
