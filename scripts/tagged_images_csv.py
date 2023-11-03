import requests
import csv

# Base URL for the API calls
BASE_URL = "http://123.176.98.90:8764"

# Fetching all tagged images
response = requests.get(f"{BASE_URL}/tags/get_all_tagged_images")
response.raise_for_status()
tagged_images = response.json()

# Prepare CSV data
csv_data = []

for image in tagged_images:
    tag_id = image['tag_id']
    image_path = image['file_path']
    image_hash = image['image_hash']
    date_tagged = image['creation_time']
    user_who_tagged = image['user_who_created']
    
    # Fetch the tag string using tag_id
    tag_response = requests.get(f"{BASE_URL}/tags/list_tag_definition")
    tag_response.raise_for_status()
    tags = tag_response.json()
    
    tag_string = next((tag['tag_string'] for tag in tags if tag['tag_id'] == tag_id), None)
    
    csv_data.append([tag_id, image_path, image_hash, tag_string, date_tagged, user_who_tagged])

# Write CSV data to a file
with open('tagged_images.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Tag ID', 'Image Path', 'Image Hash', 'Tag String', 'Date Tagged', 'User Who Tagged'])
    csvwriter.writerows(csv_data)

print("CSV file has been created as 'tagged_images.csv'.")
